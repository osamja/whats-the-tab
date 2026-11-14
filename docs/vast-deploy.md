# Deploying whats-the-tab to a vast.ai instance

This is the runbook for running the API on a fresh vast.ai GPU instance, behind
the Linode nginx proxy at `pyaar.ai/transcribe/*`.

## Why vast.ai (and not Docker)

vast.ai instances are themselves containers — there is no Docker daemon inside.
You cannot `docker compose up` on a standard instance. You must run
`redis-server`, `gunicorn`, and the transcription worker directly on the host
with `uv`.

> vast.ai also offers VM instances that support Docker. This runbook covers the
> container-style instance, which is what the production proxy setup uses.

## Important gotchas

- **`/app` is not a git repo.** Code is scp'd at instance creation time and
  drifts from `main`. Sync from your local clone before launching, or you'll be
  running stale code.
- **No process manager.** Nothing restarts services on reboot or crash. If the
  instance is rebooted, restart redis + gunicorn + the worker manually.
  `onstart.sh` is empty by default — populate it if you want auto-start.
- **Port mapping.** Gunicorn binds to `8008` inside the container. vast maps it
  to a random external port (visible in the vast UI under "Open Ports"). Use the
  external mapped port everywhere outside the instance, including in nginx
  upstream config.
- **Worker callbacks need a shared secret.** Set the same
  `WORKER_SHARED_SECRET` for gunicorn and the worker. If it is unset,
  `/transcribe/_result/` rejects all callbacks and every job fails.
- **Media is not served by default when `DEBUG=False`.** Set
  `SERVE_MEDIA=True` if the instance is serving `/media/` directly (this
  runbook does), or put nginx/object storage in front of it and leave
  `SERVE_MEDIA=False`.

## First-time setup

Connect: `ssh -p <ssh_port> root@<vast_ip>`

### 1. Install Redis

```bash
apt-get update -qq
apt-get install -y redis-server
redis-server --daemonize yes --logfile /var/log/redis-server.log
redis-cli ping   # should return PONG
```

(systemd is not available inside vast containers, so the apt postinstall can't
auto-start it — `--daemonize yes` is required.)

### 2. Sync code from your local clone

From your laptop, in the whats-the-tab repo:

```bash
scp -P <ssh_port> -r \
  transcribeapp pytorch_mt3 musictranscription accounts manage.py \
  pyproject.toml uv.lock \
  root@<vast_ip>:/app/
```

Or sync only changed files if you know what they are. Check the running version
against `git log` on your local clone if anything looks stale.

### 3. Install dependencies

On the instance:

```bash
cd /app
uv sync --frozen --no-dev --no-install-project --extra gpu
```

### 4. Configure the environment

Create `/app/.env` (or export these variables before each command). Keep the
file readable only by root:

```bash
cat > /app/.env <<'EOF'
DEBUG=False
SECRET_KEY=<generate-a-new-secret>
DATABASE_URL=<postgres-url-or-leave-unset-for-sqlite>
REDIS_URL=redis://localhost:6379
WEB_URL=http://127.0.0.1:8008
WORKER_SHARED_SECRET=<generate-a-second-secret>
SERVE_MEDIA=True
ALLOWED_HOSTS=pyaar.ai,127.0.0.1,localhost
EOF
chmod 600 /app/.env
```

`SECRET_KEY` and `WORKER_SHARED_SECRET` must be different. Generate each with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

### 5. Run migrations

```bash
cd /app
uv run python manage.py migrate --noinput
uv run python manage.py collectstatic --noinput
```

### 6. Start gunicorn

```bash
cd /app
nohup uv run gunicorn \
  --bind 0.0.0.0:8008 \
  --workers 1 \
  --threads 2 \
  --worker-class gthread \
  --timeout 300 \
  musictranscription.wsgi \
  > /app/gunicorn.log 2>&1 &
disown
```

The web process handles HTTP, status polling, and media. It does not load the
MT3 model.

### 7. Start the transcription worker

```bash
cd /app
nohup uv run python manage.py runworker \
  > /app/worker.log 2>&1 &
disown
```

The worker polls the Redis queue, downloads audio from `WEB_URL`, runs GPU
inference, posts the result back with `X-Worker-Token`, and publishes progress.
It also reaps tasks left behind by dead workers.

### 8. Verify

```bash
ps -ef | grep -E "redis|gunicorn|runworker" | grep -v grep
# Should see: redis-server, gunicorn master+worker, runworker
```

From outside the instance (using the vast-mapped external port for `8008`):

```bash
curl -H "Host: pyaar.ai" http://<vast_ip>:<ext_port_for_8008>/transcribe/
# {"status": "ok"}
```

The `Host: pyaar.ai` header is required because direct vast IPs are not in
`ALLOWED_HOSTS`. Through the Linode nginx proxy this is automatic.

## Wiring up the Linode nginx proxy

On the Linode (`ssh -p 44444 sammy@173.255.217.39`), edit
`~/workspace/nginx_sammyjaved_proxy/nginx.conf` and update the
`vast-ai-backend` upstream to the new instance:

```nginx
upstream vast-ai-backend {
    server <vast_ip>:<ext_port_for_8008>;
    keepalive 16;
}
```

Then **fully restart** the nginx container — `nginx -s reload` is not
sufficient when the upstream IP changes (see
[`nginx_sammyjaved_proxy/NOTES.md`](https://github.com/osamja/nginx_sammyjaved_proxy)
on the Linode for details):

```bash
cd ~/workspace/nginx_sammyjaved_proxy
docker compose restart my-site-proxy
```

Verify end-to-end through the public domain:

```bash
curl https://pyaar.ai/transcribe/
```

## Debugging

| Symptom | Likely cause |
|---|---|
| `https://pyaar.ai/transcribe/*` hangs/504 after upstream change | nginx held stale upstream state. `docker compose restart my-site-proxy` on the Linode. |
| `400 Bad Request` from a direct vast curl | Missing `Host: pyaar.ai` header (vast IP isn't in `ALLOWED_HOSTS`). |
| Every job fails with `403` at `_result` | `WORKER_SHARED_SECRET` is missing or does not match between web and worker. |
| `/media/...` returns 404 | `SERVE_MEDIA=False` and no nginx/object-storage layer is serving media. Set `SERVE_MEDIA=True` or configure media serving. |
| `/transcribe/status/<id>/` never leaves `queued` | Worker is not running, or cannot reach Redis. Check `/app/worker.log` and `redis-cli ping`. |
| `current_chunk`/`total_chunks` always 0 in status response | Code on `/app` is older than commit `9788335` ("Add real-time transcription progress tracking"). Re-sync `transcribeapp/` and `pytorch_mt3/standalone_inference.py`, run migrations, restart workers. |
| `ModuleNotFoundError` after restart | Likely uv venv drift; `uv sync --frozen --no-dev --no-install-project --extra gpu` from `/app`. |

## Process layout reference

| Process | Pid pattern | Purpose |
|---|---|---|
| `redis-server *:6379` | 1 | Queue and pub/sub broker |
| `gunicorn ... musictranscription.wsgi` | master + 1 worker | HTTP API on `:8008` |
| `python manage.py runworker` | 1 | Long-running transcription jobs, progress, result upload |

GPU memory usage is dominated by the worker. The gunicorn worker only handles
fast DB-backed endpoints (`/upload/`, `/status/`) and does not need GPU memory.
