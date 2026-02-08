# PyTorch MT3 Implementation

This directory contains the PyTorch implementation of the MT3 model, separated from the gitignored `mt3` directory.

## Files

### Core Implementation
- `pytorch_model.py` - MT3 model architecture in PyTorch
- `pytorch_spectrograms.py` - Audio preprocessing and spectrogram extraction
- `pytorch_inference.py` - High-level inference API (standalone)
- `__init__.py` - Package initialization

### Checkpoint & Conversion
- `mt3_pytorch_checkpoint.pt` - Converted model weights (175 MB, auto-downloaded from GitHub Releases)
- `download_checkpoint.py` - Auto-downloads checkpoint if missing
- `utils/convert_jax_to_pytorch.py` - Script to convert JAX checkpoints

**Note:** The checkpoint is automatically downloaded on first use. To download manually: `python download_checkpoint.py`

### Testing & Examples
- `test_pytorch_mt3.py` - Comprehensive test suite
- `test_trained_model.py` - Test with trained weights
- `example_pytorch_inference.py` - Usage examples

## Django Integration

The PyTorch backend is integrated with the Django API via:
- `musictranscription/transcribeapp/ml_pytorch.py` - PyTorch inference for Django
- `musictranscription/transcribeapp/tasks.py` - Updated to support PyTorch
- `musictranscription/.env` - Configuration (USE_PYTORCH=True)

## Quick Start

### Standalone Usage

```python
from pytorch_mt3 import MT3Model, MT3Config, load_audio, audio_to_frames
import torch

# Load model with trained weights
config = MT3Config(vocab_size=1536)
model = MT3Model(config)
checkpoint = torch.load('pytorch_mt3/mt3_pytorch_checkpoint.pt')
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Load and transcribe audio
audio = load_audio('song.mp3', sample_rate=16000)
# ... (see example_pytorch_inference.py for full example)
```

### Django API

```bash
# Configure to use PyTorch (in musictranscription/.env)
USE_PYTORCH=True

# Test via Django
cd musictranscription
python manage.py runserver

# Or test directly
python test_django_pytorch.py
```

## Performance

- **Model Size**: 45.9M parameters
- **Checkpoint**: 175 MB
- **GPU Memory**: ~2 GB
- **Inference Speed**: ~0.5x real-time on RTX 4090

## Files Not in Git

This directory is NOT gitignored, so all files will be tracked.

The `mt3` directory (parent level) IS gitignored, which is why we moved files here.

## Next Steps

1. Test Django API with PyTorch backend
2. Compare output quality with JAX implementation
3. Optimize inference speed
4. Add batch processing support
