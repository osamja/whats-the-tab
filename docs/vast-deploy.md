# Deploying whats-the-tab to a vast.ai instance

This is the runbook for getting the API running on a fresh vast.ai GPU instance, behind the Linode nginx proxy at `pyaar.ai/transcribe/*`.

## Why vast.ai (and not Docker)

vast.ai instances are themselves containers — there is no Docker daemon inside. You cannot `docker compose up`. You must run `redis-server`, `gunicorn`, and the dramatiq worker directly on the host with `uv`.

## Important gotchas

- **`/app` is not a git repo.** Code is scp'd at instance creation time and drifts from `main` over time. Always sync from your local clone before launching, or you'll be running stale code (we hit this — the chunk-progress commit was missing for weeks).
- **No process manager.** Nothing restarts services on reboot or crash. If the instance is rebooted, you must manually restart redis + gunicorn + dramatiq. `onstart.sh` is empty by default — populate it if you want auto-start.
- **Port mapping.** Gunicorn binds to `8008` inside the container. vast maps it to a random external port (visible in the vast UI under "Open Ports"). Use the external mapped port everywhere outside the instance, including in nginx upstream config.
- **Sync gunicorn (1 worker) blocks `/transcribe/status/`.** While `/transcribe/generate/` is running on the single sync worker, status polls queue behind it and nginx returns 504. Async mode (Dramatiq + Redis) is the fix — see below.

## First-time setup

Connect: `ssh -p <ssh_port> root@<vast_ip>`

### 1. Install Redis

```bash
apt-get update -qq
apt-get install -y redis-server
redis-server --daemonize yes --logfile /var/log/redis-server.log
redis-cli ping   # should return PONG
```

(systemd is not available inside vast containers, so the apt postinstall can't auto-start it — `--daemonize yes` is required.)

### 2. Sync code from your local clone

From your laptop, in the whats-the-tab repo:

```bash
scp -P <ssh_port> -r \
  transcribeapp pytorch_mt3 musictranscription manage.py \
  root@<vast_ip>:/app/
```

Or sync only changed files if you know what they are. Check the running version against `git log` on your local clone if anything looks stale.

### 3. Run migrations

```bash
cd /app
IS_ASYNC=True REDIS_URL=redis://localhost:6379 \
  uv run python manage.py migrate
```

### 4. Start gunicorn (async mode)

```bash
cd /app
DEBUG=False USE_PYTORCH=True IS_ASYNC=True REDIS_URL=redis://localhost:6379 \
  nohup uv run gunicorn \
    --bind 0.0.0.0:8008 \
    --timeout 600 \
    musictranscription.wsgi \
  > /app/gunicorn.log 2>&1 &
disown
```

### 5. Start the dramatiq worker

```bash
cd /app
DEBUG=False USE_PYTORCH=True IS_ASYNC=True REDIS_URL=redis://localhost:6379 \
  nohup uv run python manage.py rundramatiq \
    --processes 1 --threads 1 \
  > /app/dramatiq.log 2>&1 &
disown
```

### 6. Verify

```bash
ps -ef | grep -E "redis|gunicorn|dramatiq" | grep -v grep
# Should see: redis-server, gunicorn master+worker, dramatiq master+worker
```

From outside the instance (using the vast-mapped external port for `8008`):

```bash
curl -H "Host: pyaar.ai" http://<vast_ip>:<ext_port_for_8008>/transcribe/upload/
# {"error": "Failed to upload file"}   ← endpoint reachable, expected GET-on-POST
```

The `Host: pyaar.ai` header is required because direct vast IPs are not in `ALLOWED_HOSTS` — without it Django returns 400. Through the Linode nginx proxy this is automatic.

## Wiring up the Linode nginx proxy

On the Linode (`ssh -p 44444 sammy@173.255.217.39`), edit `~/workspace/nginx_sammyjaved_proxy/nginx.conf` and update the `vast-ai-backend` upstream to the new instance:

```nginx
upstream vast-ai-backend {
    server <vast_ip>:<ext_port_for_8008>;
    keepalive 16;
}
```

Then **fully restart** the nginx container — `nginx -s reload` is not sufficient when the upstream IP changes (see [`nginx_sammyjaved_proxy/NOTES.md`](https://github.com/osamja/nginx_sammyjaved_proxy) on the Linode for details):

```bash
cd ~/workspace/nginx_sammyjaved_proxy
docker compose restart my-site-proxy
```

Verify end-to-end through the public domain:

```bash
curl https://pyaar.ai/transcribe/upload/
# {"error": "Failed to upload file"}
```

## Debugging

| Symptom | Likely cause |
|---|---|
| `https://pyaar.ai/transcribe/*` hangs/504 after upstream change | nginx held stale upstream state. `docker compose restart my-site-proxy` on the Linode. |
| `400 Bad Request` from a direct vast curl | Missing `Host: pyaar.ai` header (vast IP isn't in `ALLOWED_HOSTS`). |
| `/transcribe/status/<id>/` returns 504 mid-transcription | Running in sync mode with one gunicorn worker. Switch to `IS_ASYNC=True` and start dramatiq. |
| `current_chunk`/`total_chunks` always 0 in status response | Code on `/app` is older than commit `9788335` ("Add real-time transcription progress tracking"). Re-sync `transcribeapp/` and `pytorch_mt3/standalone_inference.py`, run migrations, restart workers. |
| `ModuleNotFoundError` after restart | Likely uv venv drift; `uv sync --frozen --no-dev --no-install-project` from `/app`. |

## Process layout reference

| Process | Pid pattern | Purpose |
|---|---|---|
| `redis-server *:6379` | 1 | Dramatiq broker |
| `gunicorn ... musictranscription.wsgi` | master + 1 worker | HTTP API on `:8008` |
| `dramatiq ... transcribeapp.tasks` | master + 1 worker | Long-running transcription jobs |

GPU memory usage is dominated by the dramatiq worker (which loads the MT3 PyTorch model on each call). The gunicorn worker only handles fast DB-backed endpoints (`/upload/`, `/status/`) and does not need GPU memory.
