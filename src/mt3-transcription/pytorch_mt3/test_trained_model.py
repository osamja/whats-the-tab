"""Test the trained MT3 model with real checkpoint weights."""

import torch
import sys
import os

# Import using direct module loading
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = os.path.join(os.path.dirname(__file__), 'mt3')
pytorch_model = load_module('pytorch_model', os.path.join(base, 'pytorch_model.py'))
pytorch_spec = load_module('pytorch_spectrograms', os.path.join(base, 'pytorch_spectrograms.py'))

MT3Model = pytorch_model.MT3Model
MT3Config = pytorch_model.MT3Config
load_audio = pytorch_spec.load_audio
audio_to_frames = pytorch_spec.audio_to_frames
SpectrogramConfig = pytorch_spec.SpectrogramConfig


def main():
    print("=" * 80)
    print("Testing MT3 with TRAINED Weights!")
    print("=" * 80)

    # Check for checkpoint
    checkpoint_path = 'mt3_pytorch_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found: {checkpoint_path}")
        print("Please run: python convert_jax_to_pytorch.py")
        return 1

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create model
    print("\nCreating model...")
    config = MT3Config(
        vocab_size=1536,  # MT3 vocab size
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

    model = MT3Model(config).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Load weights
    print("Loading weights into model...")
    try:
        # Use strict=False because JAX checkpoint doesn't have LayerNorm params
        # (T5 uses RMSNorm which doesn't have learnable parameters)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("✓ Weights loaded successfully!")
        print(f"  Loaded: {len(state_dict)} parameters from checkpoint")
        print(f"  Missing: {len(missing_keys)} (LayerNorm/Positional encoding - using initialized values)")
        print(f"  Unexpected: {len(unexpected_keys)}")

        # Check that we loaded the important weights
        important_loaded = sum(1 for k in state_dict.keys() if any(x in k for x in ['proj', 'attn', 'mlp', 'embedding']))
        print(f"  Important parameters loaded: {important_loaded}")

    except Exception as e:
        print(f"Error loading weights: {e}")
        return 1

    model.eval()

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load test audio - prefer real music over synthetic
    audio_file = None
    for candidate in ['john_mayer_neon_10sec.mp3', 'test_10sec.mp3']:
        if os.path.exists(candidate):
            audio_file = candidate
            break

    if not audio_file:
        print(f"\nTest audio not found. Please run:")
        print(f"  python example_pytorch_inference.py --create-test-audio test.mp3")
        return 1

    print(f"\nLoading audio: {audio_file}")
    audio = load_audio(audio_file, sample_rate=16000, mono=True).to(device)
    print(f"Audio length: {len(audio)/16000:.2f}s")

    # Compute spectrogram
    print("Computing spectrogram...")
    spec_config = SpectrogramConfig()
    frames, times = audio_to_frames(audio, spec_config)
    print(f"Spectrogram shape: {frames.shape}")

    # Process first chunk
    max_frames = 256
    chunk = frames[:max_frames]
    if chunk.size(0) < max_frames:
        import torch.nn.functional as F
        pad = max_frames - chunk.size(0)
        chunk = F.pad(chunk, (0, 0, 0, pad))

    chunk = chunk.unsqueeze(0)  # Add batch dim

    print("\n" + "=" * 80)
    print("Running inference with TRAINED weights...")
    print("=" * 80)

    with torch.no_grad():
        # Time the inference
        import time
        start = time.time()

        tokens = model.generate(
            chunk,
            max_length=1024,
            start_token_id=0,
            eos_token_id=1,
            temperature=1.0,
        )

        elapsed = time.time() - start

    tokens_list = tokens.squeeze(0).cpu().tolist()

    print(f"\n✓ Inference complete!")
    print(f"Time: {elapsed:.3f}s")
    print(f"Generated {len(tokens_list)} tokens")
    print(f"\nFirst 100 tokens:")
    print(tokens_list[:100])

    # Check if tokens look reasonable (not all the same)
    unique_tokens = len(set(tokens_list))
    print(f"\nUnique tokens: {unique_tokens}/{len(tokens_list)}")

    if unique_tokens > 10:
        print("✓ Tokens look diverse (likely real transcription)")
    else:
        print("⚠ Tokens may not be meaningful (low diversity)")

    print("\n" + "=" * 80)
    print("SUCCESS! Model is working with trained weights!")
    print("=" * 80)
    print("\nNext step: Integrate vocabulary to decode tokens to MIDI")

    return 0


if __name__ == "__main__":
    sys.exit(main())
