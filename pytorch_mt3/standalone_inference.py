"""Standalone PyTorch MT3 inference with minimal dependencies.

This module provides inference without requiring full JAX/T5X installation.
"""

import os
import sys
import torch
import numpy as np

# Import our PyTorch implementation.
from .pytorch_model import MT3Model, MT3Config
from .pytorch_spectrograms import SpectrogramConfig, audio_to_frames, load_audio
from .download_checkpoint import ensure_checkpoint


class StandaloneMT3:
    """Standalone PyTorch MT3 for inference without JAX dependencies."""

    def __init__(self, checkpoint_path, device=None):
        """Initialize standalone MT3 model.

        Args:
            checkpoint_path: Path to PyTorch checkpoint
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Standalone MT3 initializing on {self.device}")

        # Create model config
        self.config = MT3Config(
            vocab_size=1536,
            emb_dim=512,
            num_heads=6,
            num_encoder_layers=8,
            num_decoder_layers=8,
            head_dim=64,
            mlp_dim=1024,
            dropout_rate=0.1,
            max_encoder_length=256,
            max_decoder_length=1024,
            input_depth=512,
        )

        # Create and load model
        self.model = MT3Model(self.config).to(self.device)

        if checkpoint_path:
            # Ensure checkpoint exists, downloading if necessary
            checkpoint_path = str(ensure_checkpoint(checkpoint_path))
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded {len(state_dict)} parameters (missing: {len(missing)})")

        self.model.eval()

        # Spectrogram config
        self.spectrogram_config = SpectrogramConfig()

        print(f"✓ Model ready with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def transcribe_file(self, audio_path, output_midi_path=None):
        """Transcribe audio file to tokens.

        Args:
            audio_path: Path to audio file
            output_midi_path: Optional path to save MIDI (requires note-seq)

        Returns:
            Dictionary with tokens and metadata
        """
        print(f"\nTranscribing: {audio_path}")

        # Load audio
        audio = load_audio(audio_path, sample_rate=16000, mono=True)
        print(f"  Audio: {len(audio)/16000:.2f}s")

        # Transcribe
        result = self.transcribe(audio)

        # Save MIDI if requested and note-seq is available
        if output_midi_path and result.get('note_sequence'):
            try:
                import note_seq
                note_seq.sequence_proto_to_midi_file(result['note_sequence'], output_midi_path)
                print(f"  ✓ MIDI saved: {output_midi_path}")
            except ImportError:
                print("  ⚠ note-seq not available, skipping MIDI save")

        return result

    def transcribe(self, audio):
        """Transcribe audio samples to tokens.

        Args:
            audio: Audio tensor or numpy array [num_samples]

        Returns:
            Dictionary with tokens and timing info
        """
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        audio = audio.to(self.device)

        # Compute spectrogram
        frames, frame_times = audio_to_frames(audio, self.spectrogram_config)
        print(f"  Spectrogram: {frames.shape[0]} frames")

        # Process in chunks
        all_tokens = []
        all_start_times = []

        chunk_size = self.config.max_encoder_length
        num_frames = frames.size(0)
        num_chunks = (num_frames + chunk_size - 1) // chunk_size

        print(f"  Processing {num_chunks} chunk(s)...")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
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
                    max_length=self.config.max_decoder_length,
                    start_token_id=0,
                    eos_token_id=1,
                    temperature=1.0,
                )

            # Convert to list
            tokens = tokens.squeeze(0).cpu().tolist()
            tokens = self._trim_generated_prefix(tokens)

            # Record chunk
            start_time = chunk_times[0].item() if len(chunk_times) > 0 else 0.0
            all_tokens.append(tokens)
            all_start_times.append(start_time)

            print(f"    Chunk {chunk_idx + 1}/{num_chunks}: {len(tokens)} tokens")

        # Try to decode with note-seq if available
        note_sequence = None
        try:
            note_sequence = self._decode_to_note_sequence(all_tokens, all_start_times)
        except ImportError:
            print("  ℹ note-seq not available for MIDI decoding")
        except Exception as e:
            print(f"  ⚠ Note sequence decoding failed: {e}")

        return {
            'tokens': all_tokens,
            'start_times': all_start_times,
            'num_chunks': num_chunks,
            'note_sequence': note_sequence,
        }

    @staticmethod
    def _trim_generated_prefix(tokens):
        """Drop leading generation-only special tokens.

        MT3 training/inference paths operate in decoded vocabulary space and do
        not include decoder start/PAD tokens as musical events.
        """
        if not tokens:
            return tokens

        # Generated sequences can begin with repeated decoder start/pad ids (0).
        idx = 0
        while idx < len(tokens) and tokens[idx] == 0:
            idx += 1
        return tokens[idx:]

    def _decode_to_note_sequence(self, token_chunks, start_times):
        """Decode tokens to note sequence using MT3 vocabulary.

        This requires note-seq and MT3 vocabulary to be available.
        """
        # Import vendored MT3 decoding modules (no TF/seqio/t5 needed).
        from .mt3_decoding import vocabularies, metrics_utils, note_sequences

        # Build codec and vocabulary
        codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=1)
        )
        encoding_spec = note_sequences.NoteEncodingWithTiesSpec

        # Build vocabulary to convert model tokens to codec indices
        vocab = vocabularies.vocabulary_from_codec(codec)

        # Create predictions in MT3 format
        predictions = []
        for tokens, start_time in zip(token_chunks, start_times):
            # Trim EOS
            tokens = np.array(tokens, np.int32)
            if vocabularies.DECODED_EOS_ID in tokens:
                tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]

            # Convert from model vocabulary space to codec indices
            # vocab.decode handles special tokens (PAD/EOS/UNK -> DECODED_INVALID_ID)
            # and subtracts the special token offset for regular tokens
            tokens = np.array(vocab.decode(tokens), np.int32)

            # Regular MT3 decode path does not emit a leading invalid special
            # token. Drop any leading decoded invalid markers for parity.
            if tokens.size:
                first_valid = np.argmax(tokens >= 0)
                if tokens[first_valid] >= 0:
                    tokens = tokens[first_valid:]
                else:
                    tokens = np.array([], np.int32)

            # Round start time
            start_time -= start_time % (1 / codec.steps_per_second)

            predictions.append({
                'est_tokens': tokens,
                'start_time': start_time,
                'raw_inputs': []
            })

        # Decode to note sequence
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=codec, encoding_spec=encoding_spec
        )

        return result['est_ns']


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python standalone_inference.py <audio_file> [output.midi]")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_midi = sys.argv[2] if len(sys.argv) > 2 else None

    # Initialize model
    checkpoint = 'mt3_pytorch_checkpoint.pt'
    model = StandaloneMT3(checkpoint)

    # Transcribe
    result = model.transcribe_file(audio_file, output_midi)

    print(f"\n✓ Transcription complete!")
    print(f"  Total chunks: {result['num_chunks']}")
    print(f"  Total tokens: {sum(len(t) for t in result['tokens'])}")

    if result['note_sequence']:
        print(f"  Notes: {len(result['note_sequence'].notes)}")
