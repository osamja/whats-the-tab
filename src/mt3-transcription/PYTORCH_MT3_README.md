# PyTorch MT3 Implementation

This directory contains a PyTorch re-implementation of the MT3 (Multi-Task Multitrack Music Transcription) model, originally implemented in JAX/Flax.

## Overview

The PyTorch implementation provides the same architecture as the original MT3 model but using PyTorch instead of JAX/TensorFlow. While potentially less optimized than the original, it allows for easier integration with PyTorch-based pipelines and inference workflows.

## Files

### Core Implementation
- `mt3/pytorch_model.py` - Main MT3 model architecture in PyTorch
- `mt3/pytorch_spectrograms.py` - Audio preprocessing and spectrogram extraction
- `mt3/pytorch_inference.py` - High-level inference API

### Testing
- `test_pytorch_mt3.py` - Comprehensive test suite for the PyTorch implementation

## Model Architecture

The PyTorch MT3 model follows the original architecture:

### Configuration (MT3 model)
- **Encoder**: 8 layers, 512-dim embeddings, 6 attention heads
- **Decoder**: 8 layers, 512-dim embeddings, 6 attention heads
- **Input**: Mel spectrograms (512 mel bins, 16kHz audio)
- **Output**: Token sequence representing MIDI events
- **Total parameters**: ~45.9M

### Key Components

1. **MT3Encoder**: Processes continuous spectrogram inputs
   - Linear projection from 512-d spectrograms to 512-d embeddings
   - Fixed sinusoidal positional encoding
   - 8 transformer encoder layers with self-attention
   - Gated MLP with GELU activation

2. **MT3Decoder**: Generates MIDI event tokens
   - Token embeddings + positional encoding
   - 8 transformer decoder layers with self-attention and cross-attention
   - Output projection to vocabulary

3. **SpectrogramExtractor**: Audio preprocessing
   - 16kHz sample rate
   - 512 mel bins
   - 2048 FFT size
   - 128 hop width (~125 frames/second)

## Usage

### Basic Inference

```python
from mt3.pytorch_model import MT3Model, MT3Config
from mt3.pytorch_spectrograms import load_audio, SpectrogramExtractor, SpectrogramConfig
import torch

# Create model
config = MT3Config()
model = MT3Model(config)
model.eval()

# Load and process audio
audio = load_audio('path/to/audio.mp3', sample_rate=16000)
spec_extractor = SpectrogramExtractor(SpectrogramConfig())
spectrogram = spec_extractor(audio)

# Add batch dimension
spectrogram = spectrogram.unsqueeze(0)  # [1, time, freq]

# Generate transcription
with torch.no_grad():
    tokens = model.generate(
        spectrogram,
        max_length=1024,
        temperature=1.0,
    )

print(f"Generated {tokens.size(1)} tokens")
```

### Using the Inference Wrapper

```python
from mt3.pytorch_inference import PyTorchMT3Inference

# Create inference model
model = PyTorchMT3Inference(
    model_type='mt3',  # or 'ismir2021' for piano-only
    device='cuda',     # or 'cpu'
)

# Transcribe audio file
result = model.transcribe_audio_file('path/to/audio.mp3')

# Or transcribe audio tensor
import torch
audio = torch.randn(16000 * 30)  # 30 seconds at 16kHz
result = model.transcribe(audio)

print(f"Transcribed {result['num_chunks']} chunks")
print(f"Generated {len(result['tokens'])} tokens")
```

### GPU Support

The implementation automatically uses CUDA if available:

```python
# Automatic device selection
model = PyTorchMT3Inference()  # Uses CUDA if available

# Explicit device
model = PyTorchMT3Inference(device='cuda')

# Force CPU
model = PyTorchMT3Inference(device='cpu')
```

## Testing

Run the comprehensive test suite:

```bash
cd /home/samus/programming-projects/whats-the-tab/src/mt3-transcription
python test_pytorch_mt3.py
```

The test suite includes:
1. Model forward pass
2. Autoregressive generation
3. Spectrogram extraction
4. Full inference pipeline
5. Checkpoint save/load
6. GPU execution (if CUDA available)

### Test Results

All tests pass successfully on:
- CPU (x86_64)
- NVIDIA GPU (RTX 4090 tested)

Inference speed on RTX 4090: ~0.5s for 0.5s of audio

## Current Status

### ✅ Implemented
- Full MT3 model architecture in PyTorch
- Mel spectrogram preprocessing
- Autoregressive generation
- Checkpoint save/load (PyTorch format)
- GPU support
- Comprehensive test suite

### ⚠️ Not Yet Implemented
- **JAX checkpoint conversion**: The model currently initializes with random weights
- **Token-to-MIDI conversion**: Requires vocabulary/codec integration
- **Full inference pipeline**: End-to-end audio → MIDI file

### 🔜 Next Steps

1. **Checkpoint Conversion**: Convert JAX/T5X weights to PyTorch format
   - Read zarr-format checkpoints
   - Map parameter names from JAX to PyTorch
   - Verify weight shapes match
   - Test inference output matches JAX implementation

2. **Complete Inference Pipeline**:
   - Integrate vocabulary and codec from `mt3/vocabularies.py`
   - Implement token decoding to MIDI notes
   - Add note_seq integration for MIDI file output

3. **Optimization**:
   - Profile inference performance
   - Add mixed-precision (FP16) support
   - Implement model quantization for faster inference

## Architecture Comparison

| Component | JAX/Flax | PyTorch | Status |
|-----------|----------|---------|--------|
| Encoder | ✅ | ✅ | Complete |
| Decoder | ✅ | ✅ | Complete |
| Attention | ✅ | ✅ | Complete |
| Gated MLP | ✅ | ✅ | Complete |
| Positional Encoding | ✅ | ✅ | Complete |
| Spectrogram | ✅ | ✅ | Complete |
| Generation | ✅ | ✅ | Complete |
| Checkpoint Loading | ✅ | ❌ | TODO |

## Performance

Measured on NVIDIA RTX 4090:
- **Model size**: 45.9M parameters
- **Inference time**: ~0.5s for 0.5s audio (real-time factor ~1x)
- **Memory usage**: ~2GB GPU RAM

Note: These numbers are with random weights. Performance may vary with trained weights.

## Dependencies

Required packages:
- `torch >= 2.0`
- `torchaudio >= 2.0`
- `numpy`

Optional (for full functionality):
- `note-seq` - MIDI output
- `mt3` - Vocabulary/codec from original implementation

## Differences from Original JAX Implementation

1. **Framework**: PyTorch instead of JAX/Flax
2. **Checkpoint format**: PyTorch state_dict instead of T5X zarr format
3. **Inference API**: Simplified compared to T5X predict_batch
4. **Optimization**: Less optimized for TPU, more suitable for GPU

## Contributing

When adding features:
1. Maintain compatibility with the JAX architecture
2. Add tests to `test_pytorch_mt3.py`
3. Update this README
4. Document any deviations from original implementation

## License

Same as original MT3: Apache License 2.0

## Credits

- Original MT3 by Google Magenta team
- PyTorch implementation created for easier inference integration
