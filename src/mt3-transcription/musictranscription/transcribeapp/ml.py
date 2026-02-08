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

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

import matplotlib.pyplot as plt
from mido import MidiFile, MidiTrack

from .models import AudioMIDI, MIDIChunk
import dramatiq
import io
from django.core.files import File

from midi2audio import FluidSynth

SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'

class InferenceModel(object):
  """Wrapper of T5X model for music transcription."""

  @staticmethod
  def _get_device_platform():
    """Detect the platform of the first available device."""
    try:
        devices = jax.devices()
        if devices:
            return devices[0].platform
    except Exception as e:
        print(f"Warning: Could not detect JAX device: {e}")
    return 'cpu'  # fallback

  @staticmethod
  def _patch_t5x_for_gpu():
    """Monkey-patch T5X to support GPU devices.

    T5X's bounds_from_last_device assumes TPU-specific attributes.
    We patch it to handle GPU/CPU devices gracefully.
    """
    import t5x.partitioning as t5x_partitioning

    # Store original function
    original_bounds_from_last_device = t5x_partitioning.bounds_from_last_device

    def gpu_compatible_bounds_from_last_device(last_device):
      """Return device bounds, handling non-TPU devices."""
      # For GPU/CPU, return a simple 1D mesh bound
      if hasattr(last_device, 'core_on_chip'):
        # TPU device - use original function
        return original_bounds_from_last_device(last_device)
      else:
        # GPU or CPU device - return simple 1x1x1x1 bounds
        # This creates a minimal mesh for single-device inference
        return (1, 1, 1, 1)

    # Apply the patch
    t5x_partitioning.bounds_from_last_device = gpu_compatible_bounds_from_last_device

  @staticmethod
  def _create_partitioner(device_platform):
    """Create appropriate partitioner based on device platform."""
    if device_platform == 'gpu':
        # Patch T5X to handle GPU devices before creating partitioner
        InferenceModel._patch_t5x_for_gpu()

    # Now use standard partitioner for all platforms
    return t5x.partitioning.PjitPartitioner(num_partitions=1)

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

    # Use absolute paths for gin files
    # __file__ is in musictranscription/transcribeapp/ml.py
    # We need to go to mt3-transcription/mt3/gin/
    mt3_base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    gin_files = [os.path.join(mt3_base_path, 'mt3/gin/model.gin'),
                 os.path.join(mt3_base_path, f'mt3/gin/{model_type}.gin')]

    self.batch_size = 8
    self.outputs_length = 1024
    self.sequence_length = {'inputs': self.inputs_length,
                            'targets': self.outputs_length}

    # Detect device and create appropriate partitioner
    self.device_platform = self._get_device_platform()
    print(f"MT3 InferenceModel initializing on device: {self.device_platform}")
    self.partitioner = self._create_partitioner(self.device_platform)

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
    print(f"Restoring checkpoint on {self.device_platform} device...")
    print(f"Available JAX devices: {jax.devices()}")

    try:
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

        print(f"Successfully restored checkpoint on {self.device_platform}")
    except Exception as e:
        print(f"Error during checkpoint restoration: {e}")
        print(f"Device platform: {self.device_platform}")
        print(f"Partitioner config: {self.partitioner}")
        raise

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

# Note: split_audio_segments has been moved to audio_utils.py
# Import it from there: from .audio_utils import split_audio_segments

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


