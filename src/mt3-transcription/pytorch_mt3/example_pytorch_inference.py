"""Simple example of using PyTorch MT3 for music transcription.

This script demonstrates how to use the PyTorch implementation of MT3
for basic music transcription inference.
"""

import torch
import sys
import os

# Add mt3 to path
sys.path.insert(0, os.path.dirname(__file__))

# Import using direct module loading to avoid dependency issues
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load PyTorch modules
base = os.path.join(os.path.dirname(__file__), 'mt3')
pytorch_model = load_module('pytorch_model', os.path.join(base, 'pytorch_model.py'))
pytorch_spec = load_module('pytorch_spectrograms', os.path.join(base, 'pytorch_spectrograms.py'))

MT3Model = pytorch_model.MT3Model
MT3Config = pytorch_model.MT3Config
load_audio = pytorch_spec.load_audio
SpectrogramExtractor = pytorch_spec.SpectrogramExtractor
SpectrogramConfig = pytorch_spec.SpectrogramConfig


def transcribe_audio_file(audio_path, device='cuda'):
    """Transcribe an audio file to tokens.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        device: 'cuda' or 'cpu'

    Returns:
        Generated token sequence
    """
    print(f"Transcribing: {audio_path}")
    print(f"Using device: {device}")

    # Create model
    config = MT3Config(
        vocab_size=1536,
        emb_dim=512,
        num_heads=6,
        num_encoder_layers=8,
        num_decoder_layers=8,
        max_encoder_length=256,
        max_decoder_length=1024,
    )

    model = MT3Model(config).to(device)
    model.eval()

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load audio
    print("\nLoading audio...")
    audio = load_audio(audio_path, sample_rate=16000, mono=True)
    print(f"Audio length: {len(audio)/16000:.2f}s")

    # Extract spectrogram
    print("Computing spectrogram...")
    spec_config = SpectrogramConfig()
    extractor = SpectrogramExtractor(spec_config).to(device)

    # Process in chunks (MT3 has max input length of 256 frames)
    import torch.nn.functional as F

    # Use the already-loaded module's audio_to_frames
    frames, times = pytorch_spec.audio_to_frames(audio.to(device), spec_config)
    print(f"Spectrogram shape: {frames.shape}")

    # Chunk into 256-frame segments
    max_frames = 256
    num_frames = frames.size(0)
    num_chunks = (num_frames + max_frames - 1) // max_frames

    all_tokens = []

    print(f"\nProcessing {num_chunks} chunk(s)...")

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * max_frames
            end = min(start + max_frames, num_frames)
            chunk = frames[start:end]

            # Pad if needed
            if chunk.size(0) < max_frames:
                pad = max_frames - chunk.size(0)
                chunk = F.pad(chunk, (0, 0, 0, pad))

            chunk = chunk.unsqueeze(0)  # Add batch dim

            # Generate
            tokens = model.generate(
                chunk,
                max_length=1024,
                start_token_id=0,
                eos_token_id=1,
                temperature=1.0,
            )

            chunk_tokens = tokens.squeeze(0).cpu().tolist()
            all_tokens.extend(chunk_tokens)
            print(f"  Chunk {i+1}/{num_chunks}: Generated {len(chunk_tokens)} tokens")

    print(f"\nTotal tokens generated: {len(all_tokens)}")
    print(f"First 50 tokens: {all_tokens[:50]}")

    return all_tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Transcribe audio with PyTorch MT3')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--create-test-audio', action='store_true',
                        help='Create a test audio file instead of transcribing')

    args = parser.parse_args()

    # Check if creating test audio
    if args.create_test_audio:
        import torchaudio
        print("Creating test audio file...")
        # 5 seconds of synthetic audio
        audio = torch.randn(16000 * 5)
        torchaudio.save('test_audio.wav', audio.unsqueeze(0), 16000)
        print("Created test_audio.wav (5 seconds of random noise)")
        sys.exit(0)

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Check file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        print("\nTo create a test audio file, run:")
        print("  python example_pytorch_inference.py --create-test-audio test.wav")
        sys.exit(1)

    # Transcribe
    print("=" * 60)
    print("PyTorch MT3 Transcription Example")
    print("=" * 60)
    print("\nNOTE: Model is initialized with RANDOM weights!")
    print("For actual transcription, you need to load trained weights.")
    print("=" * 60)

    tokens = transcribe_audio_file(args.audio_file, device=device)

    print("\n" + "=" * 60)
    print("Transcription complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Convert JAX checkpoint to PyTorch")
    print("2. Load trained weights into model")
    print("3. Decode tokens to MIDI using vocabulary")
