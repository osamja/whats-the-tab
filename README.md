# whats-the-tab

A music transcription API that converts audio to MIDI, built on a from-scratch PyTorch reimplementation of Google's [MT3](https://github.com/magenta/mt3) (Multi-Task Multitrack Music Transcription). The original MT3 is written in JAX/T5X; this project rewrites the model and inference pipeline in pure PyTorch and wraps it in a Django REST API.

## How it works

1. Upload an audio file (MP3, WAV, etc.) or provide a YouTube URL
2. The MT3 model processes the audio spectrogram through a T5-style encoder-decoder transformer
3. The decoder autoregressively generates MIDI note events
4. Download the resulting MIDI file

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- GPU with 8GB+ VRAM (GTX 1080 or better) for inference
- PyTorch model checkpoint: `pytorch_mt3/mt3_pytorch_checkpoint.pt`

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

At minimum you need `SECRET_KEY` set. See `.env.example` for all options.

### Local development

```bash
uv sync --frozen --no-dev --no-install-project
uv run python manage.py migrate
USE_PYTORCH=True IS_ASYNC=False uv run python manage.py runserver 0.0.0.0:8008
```

### Docker

```bash
docker compose up --build
```

The app is available at `http://127.0.0.1:8008` and Redis runs in a sidecar container.

### Production deployment

```bash
uv run python manage.py migrate
uv run gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --worker-class gthread --timeout 300 musictranscription.wsgi
```

- `gthread` with 2 threads allows `/transcribe/status/` polling while `/transcribe/generate/` runs GPU inference.
- CUDA releases the GIL during kernel execution, so the second thread can handle status polls concurrently.
- 1 worker keeps GPU memory usage low (model is loaded once).

For deploying to a vast.ai GPU instance behind the `pyaar.ai/transcribe/*` nginx proxy (the production setup), see [`docs/vast-deploy.md`](docs/vast-deploy.md). vast instances are containers themselves and cannot run Docker, so the runbook covers launching `redis-server`, `gunicorn`, and the dramatiq worker directly under `uv` with `IS_ASYNC=True`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/transcribe/` | GET | Health check |
| `/transcribe/upload/` | POST | Upload an MP3 file |
| `/transcribe/upload_from_yt/` | POST | Upload via YouTube URL |
| `/transcribe/generate/` | POST | Start MIDI transcription |
| `/transcribe/status/<id>/` | GET | Poll transcription progress (`current_chunk`/`total_chunks`) |
| `/transcribe/midi/<id>/` | GET | Get MIDI file (binary) |
| `/transcribe/download_midi/<id>/` | GET | Download MIDI as attachment |

### Example

```bash
# Upload audio
curl -F "audio=@song.mp3" http://127.0.0.1:8008/transcribe/upload/

# Start transcription (use audio_midi_id from upload response)
curl -X POST -d "audio_midi_id=<ID>" http://127.0.0.1:8008/transcribe/generate/

# Check progress
curl http://127.0.0.1:8008/transcribe/status/<ID>/

# Download MIDI
curl -o output.midi http://127.0.0.1:8008/transcribe/download_midi/<ID>/
```

## Architecture

- **ML backend:** `transcribeapp/ml_pytorch.py` — PyTorch MT3 inference
- **Model:** `pytorch_mt3/pytorch_model.py` — T5 1.1 encoder-decoder (8 layers, 512 dim, 6 heads)
- **Spectrogram:** `pytorch_mt3/pytorch_spectrograms.py` — log-mel spectrogram matching MT3's spectral_ops
- **Decoding:** `pytorch_mt3/mt3_decoding/` — vendored from MT3, TF/seqio/t5 dependencies removed
- **Checkpoint conversion:** `pytorch_mt3/convert_jax_to_pytorch.py` — JAX T5X to PyTorch state dict
- **Async tasks:** Dramatiq + Redis (optional, controlled by `IS_ASYNC` env var)

## Acknowledgments

This project is a derivative work of [MT3](https://github.com/magenta/mt3) by Google's Magenta team. The model architecture, training methodology, and decoding logic are based on their work. Vendored decoding modules retain the original copyright headers.

If you use this project in research, please cite the original MT3 paper:

```bibtex
@inproceedings{gardner2022mt3,
  title={MT3: Multi-Task Multitrack Music Transcription},
  author={Gardner, Josh and Simon, Ian and Manilow, Ethan and Hawthorne, Curtis and Engel, Jesse},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  url={https://openreview.net/forum?id=iMSjopcOn0p}
}
```

## License

[Apache License 2.0](LICENSE)
