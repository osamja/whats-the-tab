"""Test script for PyTorch MT3 implementation.

This script verifies that the PyTorch implementation works correctly.
"""

import torch
import sys
import os

# Add mt3 to path
sys.path.insert(0, os.path.dirname(__file__))

# Import directly from the module files to avoid __init__.py dependency issues
import importlib.util

def load_module_from_path(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load our PyTorch modules
base_path = os.path.join(os.path.dirname(__file__), 'mt3')
pytorch_model = load_module_from_path('pytorch_model', os.path.join(base_path, 'pytorch_model.py'))
pytorch_spectrograms = load_module_from_path('pytorch_spectrograms', os.path.join(base_path, 'pytorch_spectrograms.py'))

MT3Model = pytorch_model.MT3Model
MT3Config = pytorch_model.MT3Config
SpectrogramExtractor = pytorch_spectrograms.SpectrogramExtractor
SpectrogramConfig = pytorch_spectrograms.SpectrogramConfig
load_audio = pytorch_spectrograms.load_audio

# For inference, we need to mock some dependencies
class PyTorchMT3InferenceSimple:
    """Simplified inference class for testing without full dependencies."""

    def __init__(self, model_type='mt3', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_type = model_type
        self.sample_rate = 16000
        self.inputs_length = 256 if model_type == 'mt3' else 512
        self.outputs_length = 1024

        self.spectrogram_config = SpectrogramConfig()

        self.config = MT3Config(
            vocab_size=1536,
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

        self.model = MT3Model(self.config).to(self.device)
        self.model.eval()

        self.spectrogram_extractor = SpectrogramExtractor(self.spectrogram_config).to(self.device)

        print(f"Model initialized on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")

    @torch.no_grad()
    def transcribe(self, audio, return_note_sequence=False, temperature=1.0):
        """Simple transcription for testing."""
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.to(self.device)

        # Compute spectrogram using already-loaded module
        frames, frame_times = pytorch_spectrograms.audio_to_frames(audio, self.spectrogram_config)
        frames = frames.to(self.device)

        # Take first chunk
        max_frames = self.inputs_length
        if frames.size(0) > max_frames:
            frames = frames[:max_frames]
        elif frames.size(0) < max_frames:
            import torch.nn.functional as F
            padding = max_frames - frames.size(0)
            frames = F.pad(frames, (0, 0, 0, padding))

        frames = frames.unsqueeze(0)

        # Generate
        tokens = self.model.generate(
            frames,
            max_length=min(100, self.outputs_length),  # Shorter for testing
            start_token_id=0,
            eos_token_id=1,
            temperature=temperature,
        )

        return {
            'tokens': tokens.squeeze(0).cpu().numpy().tolist(),
            'num_chunks': 1,
        }

    def save_checkpoint(self, path):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': vars(self.config),
            'model_type': self.model_type,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, device=None):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        instance = cls(model_type=checkpoint['model_type'], device=device)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()
        return instance

PyTorchMT3Inference = PyTorchMT3InferenceSimple


def test_model_forward():
    """Test basic forward pass of the model."""
    print("\n" + "=" * 60)
    print("Test 1: Model Forward Pass")
    print("=" * 60)

    # Create config
    config = MT3Config(
        vocab_size=1536,
        emb_dim=512,
        num_heads=6,
        num_encoder_layers=2,  # Smaller for testing
        num_decoder_layers=2,
        head_dim=64,
        mlp_dim=1024,
        max_encoder_length=256,
        max_decoder_length=512,
        input_depth=512,
    )

    # Create model
    model = MT3Model(config)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    enc_len = 128
    dec_len = 64

    encoder_inputs = torch.randn(batch_size, enc_len, config.input_depth)
    decoder_inputs = torch.randint(0, config.vocab_size, (batch_size, dec_len))

    # Forward pass
    print(f"Encoder inputs shape: {encoder_inputs.shape}")
    print(f"Decoder inputs shape: {decoder_inputs.shape}")

    with torch.no_grad():
        logits = model(encoder_inputs, decoder_inputs)

    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {dec_len}, {config.vocab_size}]")

    assert logits.shape == (batch_size, dec_len, config.vocab_size), "Output shape mismatch!"
    print("✓ Forward pass successful!")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")


def test_generation():
    """Test autoregressive generation."""
    print("\n" + "=" * 60)
    print("Test 2: Autoregressive Generation")
    print("=" * 60)

    config = MT3Config(
        vocab_size=1536,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_encoder_length=128,
        max_decoder_length=256,
    )

    model = MT3Model(config)
    model.eval()

    # Create dummy encoder input
    batch_size = 1
    enc_len = 64
    encoder_inputs = torch.randn(batch_size, enc_len, config.input_depth)

    print(f"Encoder inputs shape: {encoder_inputs.shape}")
    print("Generating with max_length=50...")

    with torch.no_grad():
        generated = model.generate(
            encoder_inputs,
            max_length=50,
            start_token_id=0,
            eos_token_id=1,
        )

    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens (first 20): {generated[0, :20].tolist()}")
    print("✓ Generation successful!")


def test_spectrogram():
    """Test spectrogram extraction."""
    print("\n" + "=" * 60)
    print("Test 3: Spectrogram Extraction")
    print("=" * 60)

    config = SpectrogramConfig()
    extractor = SpectrogramExtractor(config)

    # Create synthetic audio
    sample_rate = 16000
    duration = 2.0
    num_samples = int(sample_rate * duration)
    audio = torch.randn(num_samples)

    print(f"Audio shape: {audio.shape}")
    print(f"Duration: {duration}s at {sample_rate}Hz")

    # Extract spectrogram
    spectrogram = extractor(audio)

    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Expected time frames: ~{int(duration * config.frames_per_second)}")
    print(f"Actual time frames: {spectrogram.shape[0]}")
    print(f"Mel bins: {spectrogram.shape[1]}")

    assert spectrogram.shape[1] == config.num_mel_bins, "Mel bins mismatch!"
    print("✓ Spectrogram extraction successful!")


def test_inference_model():
    """Test the inference wrapper."""
    print("\n" + "=" * 60)
    print("Test 4: Inference Model")
    print("=" * 60)

    # Create inference model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = PyTorchMT3Inference(model_type='mt3', device=device)

    # Create synthetic audio
    duration = 1.0
    num_samples = int(model.sample_rate * duration)
    audio = torch.randn(num_samples)

    print(f"Transcribing {duration}s of synthetic audio...")

    # Transcribe (with random weights, so output won't be meaningful)
    result = model.transcribe(audio, return_note_sequence=False, temperature=1.0)

    print(f"Number of generated tokens: {len(result['tokens'])}")
    print(f"Number of chunks processed: {result['num_chunks']}")
    print("✓ Inference model successful!")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("\n" + "=" * 60)
    print("Test 5: Checkpoint Save/Load")
    print("=" * 60)

    # Create and save model
    model1 = PyTorchMT3Inference(model_type='mt3', device='cpu')
    checkpoint_path = '/tmp/test_mt3_checkpoint.pt'

    print(f"Saving checkpoint to {checkpoint_path}...")
    model1.save_checkpoint(checkpoint_path)

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    model2 = PyTorchMT3Inference.load_checkpoint(checkpoint_path, device='cpu')

    # Verify weights match
    for (name1, param1), (name2, param2) in zip(
        model1.model.named_parameters(),
        model2.model.named_parameters()
    ):
        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
        assert torch.allclose(param1, param2), f"Parameter {name1} values don't match!"

    print("✓ Checkpoint save/load successful!")

    # Clean up
    os.remove(checkpoint_path)
    print(f"Cleaned up temporary checkpoint file")


def test_gpu_if_available():
    """Test GPU execution if CUDA is available."""
    print("\n" + "=" * 60)
    print("Test 6: GPU Execution (if available)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    print("CUDA is available!")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Create model on GPU
    model = PyTorchMT3Inference(model_type='mt3', device='cuda')

    # Create audio on GPU
    duration = 0.5
    num_samples = int(model.sample_rate * duration)
    audio = torch.randn(num_samples, device='cuda')

    print(f"Running inference on GPU...")

    # Time the inference
    import time
    start_time = time.time()

    result = model.transcribe(audio, return_note_sequence=False)

    elapsed = time.time() - start_time
    print(f"Inference time: {elapsed:.3f}s")
    print(f"Generated {len(result['tokens'])} tokens")
    print("✓ GPU execution successful!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PyTorch MT3 Implementation Tests")
    print("=" * 60)

    try:
        test_model_forward()
        test_generation()
        test_spectrogram()
        test_inference_model()
        test_checkpoint_save_load()
        test_gpu_if_available()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Convert JAX checkpoint weights to PyTorch format")
        print("2. Load real checkpoint and test on actual audio")
        print("3. Compare output with JAX implementation")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
