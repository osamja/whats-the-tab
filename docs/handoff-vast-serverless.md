# Handoff: Vast.ai Serverless Deployment

This is the starting point for an agent continuing the Vast.ai Serverless
implementation. Read this before changing the deployment architecture. For the
broader pre-deployment review, see [`pre-deploy-design.md`](pre-deploy-design.md).

This document records the state at handoff. For the implementation and current
deployment steps, see [`vast-serverless.md`](vast-serverless.md).

## Current state

- Main branch includes the pre-deployment hardening work (PR #4) and the
  architecture diagrams (PR #5).
- PR #6 documents Linode Object Storage and is the reference for the current
  bucket configuration.
- Tests: `uv run python manage.py test` passes (23 tests).
- Local Docker works for the web stack:
  - `docker compose up -d postgres redis web`
  - `GET /transcribe/` and `GET /transcribe/ready/` return 200.
- The worker has been tested end-to-end in Docker on CPU.
- Host GPU inference works on the local RTX 4090.
- The Docker NVIDIA runtime is **not** configured locally, so the worker
  container cannot see the GPU yet.
- The worker image builds successfully, including the checkpoint conversion
  fallback.
- A local `.env` exists with Linode Object Storage credentials. It is
  gitignored and must remain uncommitted. Do not print or copy its secrets.
- No Vast endpoint, model server, PyWorker, or Vast client exists yet.

## Target shape

```text
Browser
  -> Cloudflare / nginx
  -> Django on Linode
       -> PostgreSQL
       -> Redis
       -> Linode Object Storage (audio + MIDI)
       -> Dispatcher (runworker)
            -> Vast.ai Serverless GPU worker
```

See [`architecture.md`](architecture.md) for the diagram and request lifecycle.

The web tier, database, Redis, and object storage stay on an always-on host.
Vast runs only the GPU inference function. Do not try to host Django, Postgres,
or Redis on Vast serverless.

## Recommended implementation path

Start with **Vast Deployments (beta)** if it supports the model size and request
duration. It handles image packaging, worker provisioning, `@context` model
loading, and autoscaling with much less platform-specific code than a custom
PyWorker.

If Deployments hits a hard limit (request timeout, package size, image control,
or beta instability), fall back to a **custom PyWorker**:

- a FastAPI/uvicorn model server on the GPU instance,
- a `worker.py` using `WorkerConfig`/`HandlerConfig`,
- a Vast template that starts both.

Either way, keep the existing Redis queue and `runworker` as the dispatcher.
This preserves retries, task state, progress, and result callbacks.

## Work items

### 1. Add the Vast inference adapter

Create something like `deploy/vast/deploy.py`:

- Define a deployment named `whats-the-tab`.
- Use a PyTorch/CUDA image with `ffmpeg` and the project dependencies.
- Load the MT3 model once per worker via an `@context` class or model server.
- Expose a remote function, e.g.:

```python
async def transcribe(audio_url: str) -> dict:
    # download audio from the presigned URL
    # enforce MAX_AUDIO_DURATION_SECONDS
    # run StandaloneMT3 / transcribe
    # return {"midi_base64": "...", "notes": N}
```

- Add a benchmark using a short audio sample.
- Configure autoscaling conservatively: `max_workers=1`, `cold_workers=0`
  initially. Measure cold-start latency before enabling a cold worker.
- Keep the checkpoint in the image or download it once at worker startup; do
  not download 175 MB per request.
- Do not give the Vast worker database, Redis, or Django credentials.

### 2. Add a Vast client for the dispatcher

Create `transcribeapp/vast_client.py`:

- Call the Vast deployment with the presigned audio URL.
- Decode the returned MIDI base64 into a temp file.
- Return the temp path and any metadata.
- Read `VAST_ENDPOINT_NAME` / `VAST_DEPLOYMENT_NAME` and `VAST_API_KEY` from
  settings or environment. The Vast SDK can also read `VAST_API_KEY`.
- Keep the call synchronous from the worker's perspective
  (`asyncio.run(...)` is acceptable in `runworker`).

Add `vastai` as an optional dependency (for example a `serverless` extra) or
document a separate dispatcher requirements file.

### 3. Wire the dispatcher

In `transcribeapp/management/commands/runworker.py`:

- If Vast is configured, use `state["payload"]["audio_url"]` directly and call
  `transcribe_via_vast`.
- Otherwise keep the existing local path:
  - `_download_audio(...)`
  - `transcribe_audio(...)`
- Continue using `_upload_result(...)` and `mark_completed`/`mark_failed`.
- Preserve `WORKER_SHARED_SECRET` on the result callback.

Progress through a synchronous Vast call is limited. Either:

- publish a simple `0/1` and `1/1` progress update, or
- use Vast sessions and a status route if per-chunk progress is required.

Do not block the web tier on a long Vast call; the dispatcher owns that wait.

### 4. Deploy the web tier

On the Linode web host:

- Run Django + gunicorn, the Redis subscriber, Redis, and Postgres.
- Configure Linode Object Storage using [`object-storage.md`](object-storage.md).
- Set `SERVE_MEDIA=False`.
- Set `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `SITE_DOMAIN`, and the HTTPS
  settings.
- Set a strong `SECRET_KEY` and a different `WORKER_SHARED_SECRET`.
- Optionally set `TRANSCRIBE_API_KEY` to keep the client API private.
- Run migrations and `collectstatic`.

Current Docker caveat: `docker-compose.yml` does not pass the `AWS_*` variables
into the web container. Either run the web tier directly with `uv`, or add the
variables to the Compose service / an `env_file`.

### 5. End-to-end validation

1. Upload a short 4-second melody.
2. Generate a transcription.
3. Confirm the dispatcher calls Vast and the status becomes `completed`.
4. Download the MIDI and parse it with `note_seq`; compare note output with the
   local GPU path if possible.
5. Repeat with a real song and record latency and estimated Vast cost.
6. Verify the bucket remains private and all media access is via presigned URLs.

## Configuration reference

### Vast

```bash
VAST_API_KEY=<vast-api-key>
VAST_ENDPOINT_NAME=whats-the-tab
```

Do not put the Vast API key in the browser or frontend.

### Linode Object Storage

See [`object-storage.md`](object-storage.md). The important values:

```bash
AWS_STORAGE_BUCKET_NAME=...
AWS_S3_ENDPOINT_URL=https://<cluster>.linodeobjects.com
AWS_S3_REGION_NAME=us-east-1
AWS_S3_ADDRESSING_STYLE=virtual
AWS_QUERYSTRING_EXPIRE=3600
AWS_REQUEST_CHECKSUM_CALCULATION=when_required
AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
SERVE_MEDIA=False
```

The presigned URL must remain valid long enough for the Vast worker to download
the audio. Increase `AWS_QUERYSTRING_EXPIRE` if cold starts are slow.

## Gotchas

- Vast serverless workers are ephemeral and cannot host the web app or database.
- Vast's SDK request timeout is roughly 600 seconds. Cap audio duration or use
  sessions for longer jobs.
- Loading time is billed. Keep the image lean and the checkpoint in the image or
  a fast object-storage download.
- `torch 2.6 + cu124` does not support Blackwell GPUs. Pin workers to
  Ampere/Ada GPUs or upgrade the CUDA/PyTorch stack.
- The local worker image currently starts `manage.py runworker`, not an HTTP
  model server. Vast needs a model server/PyWorker wrapper.
- The old pre-converted checkpoint GitHub release is gone. Do not remove
  `prepare_checkpoint.py`; it rebuilds from the public T5X checkpoint.
- `generate()` currently supports batch size 1. Run one Vast request per GPU
  worker or add batching deliberately.
- Keep the worker callback authenticated with `X-Worker-Token`.
- `.env` is ignored and contains real credentials. Never commit or paste it.

## Open decisions

- Vast Deployments beta vs custom PyWorker.
- Synchronous Vast request vs Vast sessions for progress/long jobs.
- Maximum audio duration (currently 600 seconds).
- Whether to keep the client API open or require `TRANSCRIBE_API_KEY`.
- Cold-worker count and acceptable first-request latency.
- Whether the web tier runs under Docker Compose or directly with `uv` on the
  Linode.

## Definition of done

- A presigned audio URL can be sent to Vast and a usable MIDI file comes back.
- The dispatcher uses Vast when configured and still supports local fallback.
- `/status/` reaches `completed`; MIDI is downloadable.
- Retries, stale-task reaping, and result callback auth still work.
- No secrets are committed.
- README/docs explain how to deploy and how to switch between local and Vast.
- Local tests pass, and an end-to-end Vast transcription has been recorded.

## Useful local commands

```bash
# Tests
uv run python manage.py test

# Web stack
docker compose up -d postgres redis web
curl http://127.0.0.1:8008/transcribe/ready/

# Local GPU inference
SECRET_KEY=dev uv run python -c "from transcribeapp.tasks import transcribe_audio; ..."

# Worker image
docker build -f Dockerfile -t whats-the-tab-worker:local .
```
