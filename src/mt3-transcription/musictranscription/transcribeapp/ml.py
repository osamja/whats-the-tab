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

import torch.nn.functional as F

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

from .models import AudioMIDI, AudioChunk, MIDIChunk
import dramatiq
import io
from django.core.files import File

from midi2audio import FluidSynth

SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'

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
    padding = [0, frame_size - len(audio) % frame_size]   # we'll always pad to the next frame. [0, pad_width] means pad nothing at beginning, but pad pad_width to end of array with default value 0
    audio = np.pad(audio, padding, mode='constant')   
    frames = spectrograms.split_audio(audio, self.spectrogram_config)
    num_frames = len(audio) // frame_size
    times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
    return frames, times

  # def preprocess(self, ds):
  #   pp_chain = [
  #       functools.partial(
  #           t5.data.preprocessors.split_tokens_to_inputs_length,
  #           sequence_length=self.sequence_length,
  #           output_features=self.output_features,
  #           feature_key='inputs',
  #           additional_feature_keys=['input_times']),
  #       # Cache occurs here during training.
  #       preprocessors.add_dummy_targets,
  #       functools.partial(
  #           preprocessors.compute_spectrograms,
  #           spectrogram_config=self.spectrogram_config)
  #   ]
  #   for pp in pp_chain:
  #     ds = pp(ds)
  #   return ds

  def preprocess(self, ds):
    # Define the preprocessing functions with all arguments
    def split_tokens(ds, sequence_length, output_features, feature_key, additional_feature_keys):
        return t5.data.preprocessors.split_tokens_to_inputs_length(
            ds, 
            sequence_length=sequence_length,
            output_features=output_features,
            feature_key=feature_key,
            additional_feature_keys=additional_feature_keys
        )

    def add_dummy_targets(ds):
        return preprocessors.add_dummy_targets(ds)

    def compute_spectrograms(ds, spectrogram_config):
        return preprocessors.compute_spectrograms(ds, spectrogram_config=spectrogram_config)

    # Apply the preprocessing steps sequentially
    ds = split_tokens(
        ds,
        sequence_length=self.sequence_length,
        output_features=self.output_features,
        feature_key='inputs',
        additional_feature_keys=['input_times']
    )

    ds = add_dummy_targets(ds)

    ds = compute_spectrograms(ds, spectrogram_config=self.spectrogram_config)
    
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

def split_audio_segments(audio_midi, chunk_length_ms=2000, num_chunks=5):
    """Split the audio file into segments of chunk_length_ms and save them as AudioChunk objects."""
    audio = audio_midi.audio_file

    # if audio is mp3, load the mp3 file
    if audio.name.endswith('.mp3'):
      audio = AudioSegment.from_mp3(audio.path)
    elif audio.name.endswith('.wav'):
      audio = AudioSegment.from_wav(audio.path)
    elif audio.name.endswith('.mp4'):
      audio = AudioSegment.from_file(audio.path, format='mp4')
    split_filenames = []

    # Length of the audio in milliseconds
    length_ms = len(audio)

    # Start and end points for slicing
    start_ms = 0
    end_ms = chunk_length_ms

    file_counter = 0

    tmp_chunk_names = []

    # Delete existing chunks
    AudioChunk.objects.filter(audio_midi=audio_midi).delete()

    while start_ms < length_ms and file_counter < num_chunks:
        # Extract the chunk
        chunk = audio[start_ms:end_ms]
        # Save the chunk as a separate file
        chunk_name = f"{audio_midi.id}-{file_counter}.wav"
        chunk.export(chunk_name, format="wav")
        tmp_chunk_names.append(chunk_name)

        # Create a file-like object from the chunk data
        with open(chunk_name, 'rb') as chunk_file:
            chunk_file_obj = File(chunk_file)

            # Create AudioChunk object
            audio_chunk = AudioChunk.objects.create(
                audio_midi=audio_midi,
                chunk_file=chunk_file_obj,
                segment_index=file_counter
            )

        # Move to the next chunk
        start_ms = end_ms
        end_ms += chunk_length_ms
        file_counter += 1

    # delete the temporary chunk files
    for chunk_name in tmp_chunk_names:
        os.remove(chunk_name)

def transcribe_audio(audio, inference_model, play=False):
  est_ns = inference_model(audio)

  if play:
    note_seq.play_sequence(est_ns, synth=note_seq.fluidsynth,
                           sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)

  return est_ns

def download_midi(est_ns, download_path='transcription.midi'):
  note_seq.sequence_proto_to_midi_file(est_ns, download_path)

def transcribe_and_download(audio_midi, split_filenames, inference_model):
    # Delete existing midi chunks
    MIDIChunk.objects.filter(audio_midi=audio_midi).delete()

    for i, audio_filename in enumerate(split_filenames):
        audio, sr = librosa.load(audio_filename, sr=SAMPLE_RATE, mono=True)
        est_ns = transcribe_audio(audio, inference_model)

        # Create a MIDI file name
        midi_filename = 'midi_chunks/' + audio_filename.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.midi'
        download_midi(est_ns, midi_filename)

        # Save the MIDI file to a new MIDIChunk instance
        with open(midi_filename, 'rb') as midi_file:
            midi_chunk = MIDIChunk(
                audio_midi=audio_midi,
                midi_file=File(midi_file, name=os.path.basename(midi_filename)),
                segment_index=i
            )
            midi_chunk.save()

        # Update current segment
        audio_midi.current_segment = i + 1
        audio_midi.save()


