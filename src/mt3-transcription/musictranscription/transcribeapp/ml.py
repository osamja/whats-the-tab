import functools
import os

import numpy as np
import tensorflow.compat.v2 as tf

import functools
import gin
import jax
import librosa
import note_seq
import seqio
import t5
import t5x

import nest_asyncio
nest_asyncio.apply()

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WhisperModel, WhisperConfig, WhisperProcessor
import transformers

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

import matplotlib.pyplot as plt
from mido import MidiFile, MidiTrack
from pydub import AudioSegment

SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'
MODEL = "mt3" #@param["ismir2021", "mt3"]
mt3_path = 'checkpoints'

checkpoint_path = f'{mt3_path}/{MODEL}/'

print(checkpoint_path)

class InferenceModel(object):
  """Wrapper of T5X model for music transcription."""

  def __init__(self, checkpoint_path, model_type='mt3'):

    # Model Constants.
    if model_type == 'ismir2021':
      num_velocity_bins = 127
      self.encoding_spec = note_sequences.NoteEncodingSpec
      self.inputs_length = 512
    elif model_type == 'mt3':
      num_velocity_bins = 1
      self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
      self.inputs_length = 256
    else:
      raise ValueError('unknown model_type: %s' % model_type)

    gin_files = ['mt3/gin/model.gin',
                 f'mt3/gin/{model_type}.gin']

    self.batch_size = 8
    self.outputs_length = 1024
    self.sequence_length = {'inputs': self.inputs_length,
                            'targets': self.outputs_length}

    self.partitioner = t5x.partitioning.PjitPartitioner(
        num_partitions=1)

    # Build Codecs and Vocabularies.
    self.spectrogram_config = spectrograms.SpectrogramConfig()
    self.codec = vocabularies.build_codec(
        vocab_config=vocabularies.VocabularyConfig(
            num_velocity_bins=num_velocity_bins))
    self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
    self.output_features = {
        'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
        'targets': seqio.Feature(vocabulary=self.vocabulary),
    }

    # Create a T5X model.
    self._parse_gin(gin_files)
    self.model = self._load_model()

    # Restore from checkpoint.
    self.restore_from_checkpoint(checkpoint_path)

  @property
  def input_shapes(self):
    return {
          'encoder_input_tokens': (self.batch_size, self.inputs_length),
          'decoder_input_tokens': (self.batch_size, self.outputs_length)
    }

  def _parse_gin(self, gin_files):
    """Parse gin files used to train the model."""
    gin_bindings = [
        'from __gin__ import dynamic_registration',
        'from mt3 import vocabularies',
        'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
        'vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS'
    ]
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          gin_files, gin_bindings, finalize_config=False)

  def _load_model(self):
    """Load up a T5X `Model` after parsing training gin config."""
    model_config = gin.get_configurable(network.T5Config)()
    module = network.Transformer(config=model_config)
    return models.ContinuousInputsEncoderDecoderModel(
        module=module,
        input_vocabulary=self.output_features['inputs'].vocabulary,
        output_vocabulary=self.output_features['targets'].vocabulary,
        optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
        input_depth=spectrograms.input_depth(self.spectrogram_config))


  def restore_from_checkpoint(self, checkpoint_path):
    """Restore training state from checkpoint, resets self._predict_fn()."""
    train_state_initializer = t5x.utils.TrainStateInitializer(
      optimizer_def=self.model.optimizer_def,
      init_fn=self.model.get_initial_variables,
      input_shapes=self.input_shapes,
      partitioner=self.partitioner)

    restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
        path=checkpoint_path, mode='specific', dtype='float32')

    train_state_axes = train_state_initializer.train_state_axes
    self._predict_fn = self._get_predict_fn(train_state_axes)
    self._train_state = train_state_initializer.from_checkpoint_or_scratch(
        [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

  @functools.lru_cache()
  def _get_predict_fn(self, train_state_axes):
    """Generate a partitioned prediction function for decoding."""
    def partial_predict_fn(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decoder_params={'decode_rng': None})
    return self.partitioner.partition(
        partial_predict_fn,
        in_axis_resources=(
            train_state_axes.params,
            t5x.partitioning.PartitionSpec('data',), None),
        out_axis_resources=t5x.partitioning.PartitionSpec('data',)
    )

  def predict_tokens(self, batch, seed=0):
    """Predict tokens from preprocessed dataset batch."""
    prediction, _ = self._predict_fn(
        self._train_state.params, batch, jax.random.PRNGKey(seed))
    return self.vocabulary.decode_tf(prediction).numpy()

  def __call__(self, audio):
    """Infer note sequence from audio samples.

    Args:
      audio: 1-d numpy array of audio samples (16kHz) for a single example.

    Returns:
      A note_sequence of the transcribed audio.
    """
    ds = self.audio_to_dataset(audio)
    ds = self.preprocess(ds)

    model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
        ds, task_feature_lengths=self.sequence_length)
    model_ds = model_ds.batch(self.batch_size)

    # inferences = (tokens for batch in model_ds.as_numpy_iterator()
    #               for tokens in self.predict_tokens(batch))

    def process_inferences(model_ds):
      for batch in model_ds.as_numpy_iterator():
          tokens_batch = self.predict_tokens(batch)
          for tokens in tokens_batch:
              yield tokens

    inferences = process_inferences(model_ds)


    predictions = []
    for example, tokens in zip(ds.as_numpy_iterator(), inferences):
      predictions.append(self.postprocess(tokens, example))

    result = metrics_utils.event_predictions_to_ns(
        predictions, codec=self.codec, encoding_spec=self.encoding_spec)
    return result['est_ns']

  def audio_to_dataset(self, audio):
    """Create a TF Dataset of spectrograms from input audio."""
    frames, frame_times = self._audio_to_frames(audio)
    return tf.data.Dataset.from_tensors({
        'inputs': frames,
        'input_times': frame_times,
    })

  def _audio_to_frames(self, audio):
    """Compute spectrogram frames from audio."""
    frame_size = self.spectrogram_config.hop_width
    padding = [0, frame_size - len(audio) % frame_size]
    audio = np.pad(audio, padding, mode='constant')
    frames = spectrograms.split_audio(audio, self.spectrogram_config)
    num_frames = len(audio) // frame_size
    times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
    return frames, times

  def preprocess(self, ds):
    pp_chain = [
        functools.partial(
            t5.data.preprocessors.split_tokens_to_inputs_length,
            sequence_length=self.sequence_length,
            output_features=self.output_features,
            feature_key='inputs',
            additional_feature_keys=['input_times']),
        # Cache occurs here during training.
        preprocessors.add_dummy_targets,
        functools.partial(
            preprocessors.compute_spectrograms,
            spectrogram_config=self.spectrogram_config)
    ]
    for pp in pp_chain:
      ds = pp(ds)
    return ds

  def postprocess(self, tokens, example):
    tokens = self._trim_eos(tokens)
    start_time = example['input_times'][0]
    # Round down to nearest symbolic token step.
    start_time -= start_time % (1 / self.codec.steps_per_second)
    return {
        'est_tokens': tokens,
        'start_time': start_time,
        # Internal MT3 code expects raw inputs, not used here.
        'raw_inputs': []
    }

  @staticmethod
  def _trim_eos(tokens):
    tokens = np.array(tokens, np.int32)
    if vocabularies.DECODED_EOS_ID in tokens:
      tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
    return tokens

def split_mp3(audio, chunk_length_ms=2000, num_chunks=5):
    output_dir = 'content'
    # Load the mp3 file
    audio = AudioSegment.from_mp3(audio.path)
    split_filenames = []

    # Length of the audio in milliseconds
    length_ms = len(audio)

    # Start and end points for slicing
    start_ms = 0
    end_ms = chunk_length_ms

    file_counter = 0

    # Splitting the audio
    chunks = []

    while start_ms < length_ms and len(chunks) < num_chunks:
        # Extract the chunk
        chunk = audio[start_ms:end_ms]

        # Save the chunk as a separate file
        chunk_name = f"{output_dir}/{file_counter}.mp3"
        chunk.export(chunk_name, format="mp3")
        split_filenames.append(chunk_name)

        # Append the chunk to the list
        chunks.append(chunk)

        # Move to the next chunk
        start_ms = end_ms
        end_ms += chunk_length_ms
        file_counter += 1

    return chunks, split_filenames

def transcribe_audio(audio, inference_model):
  est_ns = inference_model(audio)


  note_seq.play_sequence(est_ns, synth=note_seq.fluidsynth,
                        sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
  # note_seq.plot_sequence(est_ns)
  return est_ns

def download_midi(est_ns, download_path='transcription.midi'):
  note_seq.sequence_proto_to_midi_file(est_ns, download_path)
  # files.download('/tmp/transcribed.mid')

def transcribe_and_download(split_audio, split_audio_filenames, inference_model):
  download_filenames = []

  for (audio_chunk, audio_filename) in zip(split_audio, split_audio_filenames):
    audio, sr = librosa.load(audio_filename, sr=SAMPLE_RATE, mono=True)
    est_ns = transcribe_audio(audio, inference_model)
    download_filename = audio_filename.rsplit('.', 1)[0] + '.midi'
    download_midi(est_ns, download_filename)
    download_filenames.append(download_filename)

  return download_filenames

def plot_note_on_times(midi_file_path):
    midi_file = MidiFile(midi_file_path)

    # Initialize lists to store note_on times for each track
    note_on_times = []

    for i, track in enumerate(midi_file.tracks[1:]):  # Skip the first track
        cumulative_time = 0
        track_times = []

        for msg in track:
            cumulative_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:  # Check for actual note_on event
                track_times.append(cumulative_time)

        print(f"Cumulative time: {cumulative_time} for track {i}")
        note_on_times.append(track_times)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, times in enumerate(note_on_times):
        plt.plot(times, [i+1] * len(times), 'o', label=f'Track {i+2} Note On Events')  # i+2 because we skipped the first track

    plt.yticks(range(1, len(note_on_times) + 1), [f'Track {i+2}' for i in range(len(note_on_times))])
    plt.xlabel('Time (ticks)')
    plt.title('Note On Event Times for Tracks (Excluding First Track)')
    plt.legend()
    plt.grid(True)
    plt.show()


def copy_acoustic_guitar_events(midi_files, output_file):
    # Create a new MIDI file for the output
    output_midi = MidiFile()

    metadata_track = MidiTrack()
    acoustic_track = MidiTrack()
    output_midi.tracks.append(metadata_track)

    end_of_track_msg = None

    # Process the first MIDI file to keep its tempo and time signature
    first_midi = MidiFile(midi_files[0])
    for msg in first_midi.tracks[0]:
        if msg.is_meta:
            metadata_track.append(msg.copy())

    # set the ticks per beat in output midi to be the same as first midi
    output_midi.ticks_per_beat = first_midi.ticks_per_beat

    for file_path in midi_files:
        midi_file = MidiFile(file_path)

        for track_idx, track in enumerate(midi_file.tracks):
            # Flag to track whether we are currently copying events
            copy_events = False

            for msg_idx, msg in enumerate(track):
                # Check for program change messages
                if msg.type == 'program_change' and msg.program == 24:
                    # Start copying when program 24 (acoustic guitar) is encountered
                    copy_events = True
                elif msg.type == 'program_change' and msg.program != 24:
                    # Stop copying if a different program is set
                    copy_events = False

                if copy_events:
                  # Check if acoustic track is empty
                  if len(acoustic_track) == 0 and msg.type == 'program_change':
                    acoustic_track.append(msg.copy())
                    continue

                  # Add end of track message if last message
                  if msg_idx == len(track) - 1 and msg.type == 'end_of_track':
                    end_of_track_msg = msg.copy()

                  # Ignore end of track messages and program change events
                  if msg.type == 'end_of_track' or msg.type == 'program_change':
                      continue

                  # Copy the event to the new track
                  acoustic_track.append(msg.copy())

    # Add the end of track message to the acoustic track
    if end_of_track_msg is not None:
        acoustic_track.append(end_of_track_msg)

    output_midi.tracks.append(acoustic_track)
    # Save the output MIDI file
    output_midi.save(output_file)
    return output_midi


def delete_midi_and_mp3s():
  # delete all the midi and mp3 files stored in /content such as /content/0.mp3, /content/0.midi, etc..
  for file in os.listdir('/content'):
    # do not delete 3_why_georgia or why_georgia-30.mp3
    if file.endswith('.mp3'):
      os.remove(os.path.join('/content', file))
    elif file.endswith('.midi'):
      os.remove(os.path.join('/content', file))

def generate_midi_from_audio(audio_id, audio, num_transcription_segments=100):
  inference_model = InferenceModel(checkpoint_path, MODEL)

  # mp3 is split into N segments of audio chunk length.
  # To transcribe entire mp3, num_transcription_segments = len(audio) / audio_chunk_length
  # To transcribe the first 2 seconds of an mp3, set NUM_TRANSCRIPTION_SEGMENTS to 1 assuming length is 2 seconds
  NUM_TRANSCRIPTION_SEGMENTS = int(num_transcription_segments)
  AUDIO_CHUNK_LENGTH = 2000
  split_audio, split_audio_filenames = split_mp3(audio, AUDIO_CHUNK_LENGTH, NUM_TRANSCRIPTION_SEGMENTS)

  # audio = upload_audio(sample_rate=SAMPLE_RATE)
  # log_event('uploadAudioComplete', {'value': round(len(audio) / SAMPLE_RATE)})
 
  # note_seq.notebook_utils.colab_play(audio, sample_rate=SAMPLE_RATE)

  midi_files = transcribe_and_download(split_audio, split_audio_filenames, inference_model)

  # Replace with the path to your MIDI file
  midi_file_path = midi_files[0]
  plot_note_on_times(midi_file_path)

  # Copy acoustic guitar events to a new MIDI file
  output_file = 'acoustic_guitar_only.midi'
  acoustic_guitar_midi = copy_acoustic_guitar_events(midi_files, output_file)

  print(f"Acoustic guitar events copied to '{output_file}'")

  return acoustic_guitar_midi, output_file


def sayHi():
    print("Hi from ml.py")

