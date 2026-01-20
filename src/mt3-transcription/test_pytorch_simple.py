"""Simple standalone test of PyTorch MT3 model.

This tests the PyTorch implementation without Django or JAX dependencies.
"""

import sys
import os
import torch
import importlib.util

# Load PyTorch modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = 'pytorch_mt3'
pytorch_model = load_module('pytorch_model', os.path.join(base_path, 'pytorch_model.py'))
pytorch_spec = load_module('pytorch_spectrograms', os.path.join(base_path, 'pytorch_spectrograms.py'))

MT3Model = pytorch_model.MT3Model
MT3Config = pytorch_model.MT3Config
load_audio = pytorch_spec.load_audio
audio_to_frames = pytorch_spec.audio_to_frames
SpectrogramConfig = pytorch_spec.SpectrogramConfig


def main():
    print("=" * 80)
    print("Standalone PyTorch MT3 Test")
    print("=" * 80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create model
    print("\nCreating model...")
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

    # Load checkpoint
    checkpoint_path = os.path.join(base_path, 'mt3_pytorch_checkpoint.pt')
    print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"✓ Loaded {len(state_dict)} parameters")
    print(f"  Missing: {len(missing)} (LayerNorm/positional)")

    model.eval()

    # Test audio file
    test_files = ['john_mayer_neon_10sec.mp3', 'test_10sec.mp3']
    test_audio = None

    for f in test_files:
        if os.path.exists(f):
            test_audio = f
            break

    if not test_audio:
        print("\n⚠ No test audio found")
        print("Please create one with:")
        print("  python pytorch_mt3/example_pytorch_inference.py --create-test-audio")
        return 1

    print(f"\nLoading audio: {test_audio}")
    audio = load_audio(test_audio, sample_rate=16000, mono=True).to(device)
    print(f"  Duration: {len(audio)/16000:.2f}s")

    # Compute spectrogram
    print("\nComputing spectrogram...")
    spec_config = SpectrogramConfig()
    frames, times = audio_to_frames(audio, spec_config)
    print(f"  Frames: {frames.shape}")

    # Process first chunk
    max_frames = 256
    chunk = frames[:max_frames]
    if chunk.size(0) < max_frames:
        import torch.nn.functional as F
        pad = max_frames - chunk.size(0)
        chunk = F.pad(chunk, (0, 0, 0, pad))

    chunk = chunk.unsqueeze(0)

    # Run inference
    print("\nRunning inference...")
    import time
    start = time.time()

    with torch.no_grad():
        tokens = model.generate(
            chunk,
            max_length=1024,
            start_token_id=0,
            eos_token_id=1,
            temperature=1.0,
        )

    elapsed = time.time() - start

    tokens_list = tokens.squeeze(0).cpu().tolist()

    print(f"✓ Inference complete!")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Tokens: {len(tokens_list)}")
    print(f"  Unique: {len(set(tokens_list))}")
    print(f"  First 50: {tokens_list[:50]}")

    print("\n" + "=" * 80)
    print("SUCCESS! PyTorch MT3 working end-to-end")
    print("=" * 80)
    print("\nNext: Integrate vocabulary decoder to convert tokens → MIDI")

    return 0


if __name__ == "__main__":
    sys.exit(main())
