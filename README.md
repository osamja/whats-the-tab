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
DEBUG=True uv run python manage.py runserver 0.0.0.0:8008
```

Uploaded audio and generated MIDI files are stored under `media/`.

Transcription jobs require Redis (`REDIS_URL`) and a worker process:
`uv run python manage.py runworker`. A Redis server is included in the Docker
Compose setup; for bare-metal development, start one locally or point
`REDIS_URL` at a managed instance.

### Docker

```bash
docker compose up --build
```

The app is available at `http://127.0.0.1:8008` and Redis runs in a sidecar container.

### Media storage

By default, audio and MIDI files live under `media/` (Docker Compose uses a
named volume). For a web tier and GPU workers on different hosts, configure an
S3-compatible bucket such as Cloudflare R2 with `AWS_STORAGE_BUCKET_NAME` and
the related settings in `.env.example`. Files are served with time-limited
presigned URLs, so workers never need shared filesystem access.

### Production deployment

```bash
uv run python manage.py migrate
uv run python manage.py collectstatic --noinput
uv run gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --worker-class gthread --timeout 300 musictranscription.wsgi
```

- The web process only handles HTTP, status polling, and media; it does not perform GPU inference.
- The GPU worker runs separately: `uv run python manage.py runworker`.
- `WORKER_SHARED_SECRET` must be identical in the web and worker environments; it authenticates worker result callbacks.
- Django does not serve `/media/` unless `DEBUG=True` or `SERVE_MEDIA=True`. In production, serve media through nginx or object storage; only set `SERVE_MEDIA=True` for simple deployments that need Django to serve it.
- `runworker` polls Redis for jobs, writes real-time progress, reaps stale jobs from dead workers, and retries interrupted jobs up to `TASK_MAX_RETRIES`.

For deploying to a vast.ai GPU instance behind the `pyaar.ai/transcribe/*` nginx proxy, see [`docs/vast-deploy.md`](docs/vast-deploy.md). Vast instances are containers themselves and cannot run Docker, so the runbook covers launching `redis-server`, `gunicorn`, and the `runworker` process directly under `uv`.

### Retention

Delete completed jobs and their audio/MIDI files after a retention window:

```bash
uv run python manage.py prune_jobs --days 7
```

Run this from cron/systemd or a scheduled job. Use `--dry-run` first to see how
many records would be removed.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/transcribe/` | GET | Liveness check |
| `/transcribe/ready/` | GET | Readiness check (DB, Redis, worker heartbeats) |
| `/transcribe/upload/` | POST | Upload an MP3 file |
| `/transcribe/upload_from_yt/` | POST | Upload via YouTube URL |
| `/transcribe/generate/` | POST | Start MIDI transcription |
| `/transcribe/status/<id>/` | GET | Poll transcription progress (`current_chunk`/`total_chunks`) |
| `/transcribe/midi/<id>/` | GET | Get MIDI file (binary) |
| `/transcribe/download_midi/<id>/` | GET | Download MIDI as attachment |
| `/transcribe/_result/` | POST | Internal worker result callback (requires `X-Worker-Token`) |
| `/transcribe/metrics/` | GET | Internal queue/Redis metrics |

If `TRANSCRIBE_API_KEY` is set, every client-facing endpoint requires either
`X-API-Key: <key>` or `Authorization: Bearer <key>`. The health/readiness
endpoints and the internal worker callback remain open/protected separately.

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
- **Async tasks:** Redis-backed queue + `runworker` management command

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
