"""PyTorch-based ML inference for MT3 music transcription.

This module provides a PyTorch implementation that's compatible with the
existing Django API interface.
"""

import os
import numpy as np
import torch
import librosa
import note_seq

# Import existing MT3 utilities for vocabulary and encoding
from mt3 import metrics_utils, vocabularies, note_sequences

# Import our PyTorch implementation
import sys
pytorch_mt3_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pytorch_mt3')
sys.path.insert(0, pytorch_mt3_path)

from pytorch_model import MT3Model, MT3Config
from pytorch_spectrograms import SpectrogramConfig, audio_to_frames
from download_checkpoint import ensure_checkpoint

SAMPLE_RATE = 16000


class PyTorchInferenceModel:
    """PyTorch wrapper for MT3 music transcription.

    This class provides the same interface as the JAX InferenceModel
    but uses PyTorch for inference.
    """

    def __init__(self, checkpoint_path, model_type='mt3'):
        """Initialize PyTorch MT3 model.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (e.g., 'pytorch_mt3/mt3_pytorch_checkpoint.pt')
            model_type: Model type ('mt3' or 'ismir2021')
        """
        print(f"Initializing PyTorch MT3 model: {model_type}")

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model constants
        if model_type == 'ismir2021':
            self.num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == 'mt3':
            self.num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError(f'unknown model_type: {model_type}')

        self.batch_size = 8
        self.outputs_length = 1024
        self.model_type = model_type

        # Build codec and vocabulary
        self.spectrogram_config = SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=self.num_velocity_bins
            )
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)

        # Create PyTorch model
        config = MT3Config(
            vocab_size=self.vocabulary.vocab_size,
            emb_dim=512,
            num_heads=6,
            num_encoder_layers=8,
            num_decoder_layers=8,
            head_dim=64,
            mlp_dim=1024,
            dropout_rate=0.1,
            max_encoder_length=self.inputs_length,
            max_decoder_length=self.outputs_length,
            input_depth=self.spectrogram_config.num_mel_bins,
        )

        self.model = MT3Model(config).to(self.device)
        self.model.eval()

        # Load checkpoint if provided
        if checkpoint_path:
            self.restore_from_checkpoint(checkpoint_path)

        print(f"PyTorch MT3 model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def restore_from_checkpoint(self, checkpoint_path):
        """Load model weights from PyTorch checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
        """
        print(f"Loading checkpoint from: {checkpoint_path}")

        try:
            # Handle both full path and relative path
            if not os.path.exists(checkpoint_path):
                # Try relative to project root
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                checkpoint_path = os.path.join(base_path, checkpoint_path)

            # Ensure checkpoint exists, downloading if necessary
            checkpoint_path = str(ensure_checkpoint(checkpoint_path))

            state_dict = torch.load(checkpoint_path, map_location=self.device)

            # Load with strict=False since we don't have LayerNorm params in JAX checkpoint
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            print(f"✓ Checkpoint loaded successfully")
            print(f"  Loaded: {len(state_dict)} parameters")
            print(f"  Missing: {len(missing_keys)} (LayerNorm/positional - using defaults)")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def __call__(self, audio):
        """Infer note sequence from audio samples.

        Args:
            audio: 1-d numpy array of audio samples (16kHz) for a single example.

        Returns:
            A note_sequence of the transcribed audio.
        """
        print(f"  Processing audio: {len(audio)/16000:.2f}s")

        # Convert audio to torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        audio = audio.to(self.device)

        # Compute spectrogram frames
        frames, frame_times = audio_to_frames(audio, self.spectrogram_config)
        print(f"  Computed spectrogram: {frames.shape[0]} frames")

        # Process in chunks
        predictions = []
        chunk_size = self.inputs_length
        num_frames = frames.size(0)
        num_chunks = (num_frames + chunk_size - 1) // chunk_size

        print(f"  Processing {num_chunks} chunk(s)...")

        for chunk_idx, start_idx in enumerate(range(0, num_frames, chunk_size)):
            end_idx = min(start_idx + chunk_size, num_frames)
            chunk_frames = frames[start_idx:end_idx]
            chunk_times = frame_times[start_idx:end_idx]

            # Pad if necessary
            if chunk_frames.size(0) < chunk_size:
                import torch.nn.functional as F
                padding = chunk_size - chunk_frames.size(0)
                chunk_frames = F.pad(chunk_frames, (0, 0, 0, padding))

            # Add batch dimension
            chunk_frames = chunk_frames.unsqueeze(0)

            # Generate tokens
            with torch.no_grad():
                tokens = self.model.generate(
                    chunk_frames,
                    max_length=self.outputs_length,
                    start_token_id=0,
                    eos_token_id=self.vocabulary.eos_id,
                    temperature=1.0,
                )

            # Convert to numpy and remove batch dimension
            tokens = tokens.squeeze(0).cpu().numpy()

            # Trim EOS token
            tokens = self._trim_eos(tokens)

            # Create prediction dict
            start_time = chunk_times[0].item() if len(chunk_times) > 0 else 0.0
            # Round down to nearest symbolic token step
            start_time -= start_time % (1 / self.codec.steps_per_second)

            predictions.append({
                'est_tokens': tokens,
                'start_time': start_time,
                'raw_inputs': []
            })

            print(f"    Chunk {chunk_idx + 1}/{num_chunks}: {len(tokens)} tokens")

        # Convert predictions to note sequence using MT3's built-in decoder
        print(f"  Decoding {len(predictions)} predictions to note sequence...")
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )

        est_ns = result['est_ns']
        print(f"  ✓ Generated note sequence with {len(est_ns.notes)} notes")

        return est_ns

    @staticmethod
    def _trim_eos(tokens):
        """Trim tokens at EOS token."""
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens


# Helper functions for Django integration

def transcribe_audio(audio, inference_model, play=False):
    """Transcribe audio using inference model.

    Args:
        audio: Audio samples (numpy array)
        inference_model: PyTorchInferenceModel or JAX InferenceModel
        play: Whether to play the result (not implemented for PyTorch)

    Returns:
        note_seq.NoteSequence
    """
    est_ns = inference_model(audio)

    if play:
        # Note: Playing is not implemented in PyTorch version yet
        print("Warning: Audio playback not implemented for PyTorch model")

    return est_ns


def download_midi(est_ns, download_path='transcription.midi'):
    """Save note sequence to MIDI file.

    Args:
        est_ns: note_seq.NoteSequence
        download_path: Output path for MIDI file
    """
    note_seq.sequence_proto_to_midi_file(est_ns, download_path)


def transcribe_and_download(audio_midi, split_filenames, inference_model):
    """Transcribe audio chunks and save MIDI files.

    This function is compatible with both JAX and PyTorch inference models.

    Args:
        audio_midi: AudioMIDI Django model instance
        split_filenames: List of audio chunk file paths
        inference_model: PyTorchInferenceModel or JAX InferenceModel
    """
    from .models import MIDIChunk
    from django.core.files import File

    # Delete existing MIDI chunks
    MIDIChunk.objects.filter(audio_midi=audio_midi).delete()

    for i, audio_filename in enumerate(split_filenames):
        print(f"Transcribing chunk {i+1}/{len(split_filenames)}: {audio_filename}")

        # Load audio
        audio, sr = librosa.load(audio_filename, sr=SAMPLE_RATE, mono=True)

        # Transcribe
        est_ns = transcribe_audio(audio, inference_model)

        # Create MIDI filename
        midi_filename = 'midi_chunks/' + audio_filename.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.midi'
        download_midi(est_ns, midi_filename)

        # Save to database
        with open(midi_filename, 'rb') as midi_file:
            midi_chunk = MIDIChunk(
                audio_midi=audio_midi,
                midi_file=File(midi_file, name=os.path.basename(midi_filename)),
                segment_index=i
            )
            midi_chunk.save()

        # Update progress
        audio_midi.current_segment = i + 1
        audio_midi.save()

        print(f"✓ Chunk {i+1} complete")
