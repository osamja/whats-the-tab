# whats-the-tab

Django app for transcribing music from audio files to MIDI using a PyTorch rewrite of Google's MT3.

## Local setup with uv

```bash
uv sync --frozen --no-dev --no-install-project
uv run python manage.py migrate
USE_PYTORCH=True IS_ASYNC=False uv run python manage.py runserver 0.0.0.0:8008
```

## Run with Docker

```bash
docker compose up --build
```

The app is available at `http://127.0.0.1:8008` and Redis runs in a sidecar container.

## Production deployment (Vast.ai)

```bash
uv run python manage.py migrate
uv run gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --worker-class gthread --timeout 300 musictranscription.wsgi
```

- `gthread` with 2 threads allows the `/transcribe/status/` polling endpoint to be served while a `/transcribe/generate/` request is running inference on the GPU.
- CUDA releases the GIL during kernel execution, so the second thread can handle status polls concurrently.
- 1 worker keeps GPU memory usage low (model is loaded once).

## Django transcription backends

The transcription app runs on the PyTorch MT3 backend.

Backend selection is controlled by `USE_PYTORCH`:
- `USE_PYTORCH=True` uses `transcribeapp/ml_pytorch.py`

## PyTorch checkpoint

Expected at: `pytorch_mt3/mt3_pytorch_checkpoint.pt` (~8GB, needs GTX 1080+).

`PyTorchInferenceModel` resolves relative checkpoint paths against the project root.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/transcribe/` | GET | Health check |
| `/transcribe/upload/` | POST | Upload an MP3 file |
| `/transcribe/upload_from_yt/` | POST | Upload via YouTube URL |
| `/transcribe/generate/` | POST | Start MIDI transcription |
| `/transcribe/status/<id>/` | GET | Poll transcription progress (`current_chunk`/`total_chunks`) |
| `/transcribe/midi/<id>/` | GET | Get MIDI file (binary) |
| `/transcribe/download_midi/<id>/` | GET | Download MIDI as attachment |

## API smoke test

```bash
# 1) Upload audio
curl -F "audio=@dataset/in-the-morning-jcole.mp3" http://127.0.0.1:8008/transcribe/upload/

# 2) Generate MIDI (use audio_midi_id from upload response)
curl -X POST -d "audio_midi_id=<ID>" http://127.0.0.1:8008/transcribe/generate/

# 3) Check status
curl http://127.0.0.1:8008/transcribe/status/<ID>/

# 4) Download MIDI
curl -o output.midi http://127.0.0.1:8008/transcribe/download_midi/<ID>/
```
