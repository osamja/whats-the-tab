# PyTorch MT3 Implementation

This directory contains the PyTorch implementation of the MT3 model used by the
GPU transcription worker.

## Files

### Core implementation

- `pytorch_model.py` — MT3 model architecture in PyTorch
- `pytorch_spectrograms.py` — audio preprocessing and spectrogram extraction
- `standalone_inference.py` — standalone inference wrapper and vocabulary decoding
- `download_checkpoint.py` — downloads/prepares the converted checkpoint
- `convert_jax_to_pytorch.py` — converts the original T5X checkpoint
- `prepare_checkpoint.py` — downloads the public JAX checkpoint and converts it
- `mt3_decoding/` — vendored MT3 decoding modules (no TensorFlow/seqio/t5 deps)

### Checkpoint

`mt3_pytorch_checkpoint.pt` is the converted model weights (about 175 MB). It is
not committed to git and is prepared on first use if missing.

The original pre-converted GitHub release is no longer published. If that
download fails, `download_checkpoint.py` falls back to downloading the public
T5X checkpoint from `gs://mt3/checkpoints/mt3/` and converting it locally with
`convert_jax_to_pytorch.py`. For production workers, bake the checkpoint into
the image instead of doing this at request time.

## Standalone usage

```python
from pytorch_mt3 import MT3Model, MT3Config, load_audio, audio_to_frames
import torch

config = MT3Config(vocab_size=1536)
model = MT3Model(config)
checkpoint = torch.load("pytorch_mt3/mt3_pytorch_checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
model.eval()

audio = load_audio("song.mp3", sample_rate=16000)
frames, times = audio_to_frames(audio)
```

See `standalone_inference.py` for the full encode/decode pipeline.

## Django integration

- `transcribeapp/ml_pytorch.py` — adapter used by the worker
- `transcribeapp/tasks.py` — loads audio, runs inference, writes MIDI
- `transcribeapp/management/commands/runworker.py` — Redis-queue worker

## Performance

- Model size: ~45.9M parameters
- Checkpoint: ~175 MB
- GPU memory: ~2 GB
- Inference speed: about 0.5x real-time on an RTX 4090

## Not in git

The checkpoint is gitignored (`*.pt`) and hosted as a GitHub release. The
legacy `mt3/` directory is also gitignored.
