"""PyTorch inference wrapper for MT3 model.

This module provides a high-level interface for music transcription inference
using the PyTorch implementation of MT3.
"""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np

# Import from existing MT3 modules (these work with both JAX and PyTorch)
try:
    import note_seq
    from mt3 import vocabularies
    from mt3 import metrics_utils
    from mt3 import note_sequences
    HAS_MT3_DEPS = True
except ImportError:
    HAS_MT3_DEPS = False
    print("Warning: MT3 dependencies not fully available. Some features may be limited.")

# Import our PyTorch modules
from mt3.pytorch_model import MT3Model, MT3Config
from mt3.pytorch_spectrograms import (
    SpectrogramConfig,
    SpectrogramExtractor,
    load_audio,
    audio_to_frames,
)


class PyTorchMT3Inference:
    """PyTorch-based MT3 inference wrapper.

    This class provides a simple interface for music transcription
    that mirrors the JAX InferenceModel API but uses PyTorch.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = 'mt3',
        device: Optional[str] = None,
        vocab_config: Optional[Any] = None,
    ):
        """Initialize PyTorch MT3 inference model.

        Args:
            checkpoint_path: Path to model checkpoint (optional, for future use)
            model_type: Model type ('mt3' or 'ismir2021')
            device: Device to run on ('cuda', 'cpu', or None for auto)
            vocab_config: Vocabulary configuration (optional)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Model type configuration
        self.model_type = model_type
        if model_type == 'ismir2021':
            self.num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec if HAS_MT3_DEPS else None
            self.inputs_length = 512
            self.outputs_length = 1024
        elif model_type == 'mt3':
            self.num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec if HAS_MT3_DEPS else None
            self.inputs_length = 256
            self.outputs_length = 1024
        else:
            raise ValueError(f'Unknown model_type: {model_type}')

        # Create spectrogram config
        self.spectrogram_config = SpectrogramConfig()
        self.sample_rate = self.spectrogram_config.sample_rate

        # Build vocabulary and codec
        if HAS_MT3_DEPS and vocab_config is None:
            self.codec = vocabularies.build_codec(
                vocab_config=vocabularies.VocabularyConfig(
                    num_velocity_bins=self.num_velocity_bins
                )
            )
            self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
            vocab_size = self.vocabulary.vocab_size
        else:
            # Fallback for when dependencies aren't available
            vocab_size = 1536  # Approximate vocab size
            self.codec = None
            self.vocabulary = None

        # Create model config
        self.config = MT3Config(
            vocab_size=vocab_size,
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

        # Create model
        self.model = MT3Model(self.config).to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Create spectrogram extractor
        self.spectrogram_extractor = SpectrogramExtractor(self.spectrogram_config)

        print(f"Model initialized with vocab_size={vocab_size}")
        print(f"Model has {self._count_parameters():,} parameters")

        # Checkpoint loading (placeholder for future implementation)
        if checkpoint_path is not None:
            print(f"Checkpoint path provided: {checkpoint_path}")
            print("Note: Checkpoint loading from JAX format not yet implemented.")
            print("Model is initialized with random weights.")

    def _count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @torch.no_grad()
    def transcribe_audio_file(
        self,
        audio_path: str,
        return_note_sequence: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe an audio file to MIDI.

        Args:
            audio_path: Path to audio file
            return_note_sequence: If True, return a note_seq.NoteSequence object

        Returns:
            Dictionary containing transcription results
        """
        # Load audio
        print(f"Loading audio from {audio_path}...")
        audio = load_audio(audio_path, sample_rate=self.sample_rate, mono=True)

        # Transcribe
        result = self.transcribe(audio, return_note_sequence=return_note_sequence)

        return result

    @torch.no_grad()
    def transcribe(
        self,
        audio: torch.Tensor,
        return_note_sequence: bool = True,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Transcribe audio samples to MIDI.

        Args:
            audio: Audio tensor [num_samples]
            return_note_sequence: If True, return note_seq.NoteSequence
            temperature: Sampling temperature for generation

        Returns:
            Dictionary containing transcription results
        """
        print(f"Transcribing audio of length {len(audio)/self.sample_rate:.2f}s...")

        # Ensure audio is on correct device
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.to(self.device)

        # Compute spectrogram
        frames, frame_times = audio_to_frames(audio, self.spectrogram_config)
        frames = frames.to(self.device)

        # Split into chunks if necessary (MT3 has max input length)
        num_frames = frames.size(0)
        max_frames = self.inputs_length

        all_tokens = []
        chunk_start_times = []

        # Process in chunks
        num_chunks = (num_frames + max_frames - 1) // max_frames
        print(f"Processing {num_chunks} chunk(s)...")

        for i in range(num_chunks):
            start_idx = i * max_frames
            end_idx = min(start_idx + max_frames, num_frames)
            chunk = frames[start_idx:end_idx]

            # Pad if necessary
            if chunk.size(0) < max_frames:
                padding = max_frames - chunk.size(0)
                chunk = F.pad(chunk, (0, 0, 0, padding))

            # Add batch dimension
            chunk = chunk.unsqueeze(0)  # [1, seq_len, input_depth]

            # Generate tokens
            tokens = self.model.generate(
                chunk,
                max_length=self.outputs_length,
                start_token_id=0,
                eos_token_id=1 if self.vocabulary is None else self.vocabulary.eos_id,
                temperature=temperature,
            )

            # Remove batch dimension and convert to list
            tokens = tokens.squeeze(0).cpu().numpy().tolist()
            all_tokens.extend(tokens)

            # Track start time for this chunk
            chunk_start_time = frame_times[start_idx].item() if start_idx < len(frame_times) else 0
            chunk_start_times.append(chunk_start_time)

        result = {
            'tokens': all_tokens,
            'chunk_start_times': chunk_start_times,
            'num_chunks': num_chunks,
        }

        # Convert to note sequence if requested and dependencies available
        if return_note_sequence and HAS_MT3_DEPS and self.codec is not None:
            try:
                # This would require implementing the token-to-note conversion
                # For now, return the tokens
                print("Note: Token to note sequence conversion requires additional implementation")
            except Exception as e:
                print(f"Warning: Could not convert to note sequence: {e}")

        return result

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'vocab_size': self.config.vocab_size,
                'emb_dim': self.config.emb_dim,
                'num_heads': self.config.num_heads,
                'num_encoder_layers': self.config.num_encoder_layers,
                'num_decoder_layers': self.config.num_decoder_layers,
                'head_dim': self.config.head_dim,
                'mlp_dim': self.config.mlp_dim,
                'dropout_rate': self.config.dropout_rate,
                'max_encoder_length': self.config.max_encoder_length,
                'max_decoder_length': self.config.max_decoder_length,
                'input_depth': self.config.input_depth,
            },
            'model_type': self.model_type,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None):
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded PyTorchMT3Inference instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        config_dict = checkpoint['config']
        model_type = checkpoint.get('model_type', 'mt3')

        # Create instance
        instance = cls(
            checkpoint_path=None,
            model_type=model_type,
            device=device,
        )

        # Update config from checkpoint
        for key, value in config_dict.items():
            if hasattr(instance.config, key):
                setattr(instance.config, key, value)

        # Recreate model with loaded config
        instance.model = MT3Model(instance.config).to(instance.device)

        # Load weights
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()

        print(f"Checkpoint loaded from {path}")
        return instance


# Example usage
if __name__ == "__main__":
    print("PyTorch MT3 Inference Module")
    print("=" * 50)

    # Create model
    model = PyTorchMT3Inference(model_type='mt3', device='cuda' if torch.cuda.is_available() else 'cpu')

    print("\nModel created successfully!")
    print(f"Ready for inference on {model.device}")

    # Test with synthetic audio
    print("\nTesting with synthetic audio...")
    duration = 2.0  # seconds
    num_samples = int(model.sample_rate * duration)
    test_audio = torch.randn(num_samples)

    print(f"Synthetic audio: {duration}s, {num_samples} samples")

    # Note: This will run inference with random weights since no checkpoint is loaded
    # result = model.transcribe(test_audio, return_note_sequence=False)
    # print(f"Generated {len(result['tokens'])} tokens")
