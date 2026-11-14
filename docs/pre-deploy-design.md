# Pre-Deployment Design Review: whats-the-tab + Vast.ai Serverless

Status: draft for discussion  
Scope: improvements to make before hosting the service again, with a target of using Vast.ai Serverless for GPU inference  
Related files: `README.md`, `docs/vast-deploy.md`, `docs/system-design.md`, `musictranscription/settings.py`, `transcribeapp/`, `pytorch_mt3/`

## Implementation status

Completed in the `pre-deploy-hardening` branch:

- [x] Scope `MEDIA_ROOT` to `media/`, gate Django media serving behind
  `SERVE_MEDIA`/`DEBUG`, add a persistent Docker media volume.
- [x] Require `X-Worker-Token` on `_result`, with a shared
  `WORKER_SHARED_SECRET`.
- [x] Fix queue claim/result attribution: `claim_task()` now returns the
  claimed task ID and the worker uses `BRPOPLPUSH`.
- [x] Add stale-task reaping, retry limit, task-hash TTLs, dead-letter cap, and
  resilient heartbeat.
- [x] Add upload size/content-type limits and audio-duration checks.
- [x] Add per-IP Redis rate limits for uploads, transcriptions, and YouTube
  downloads, plus a `/transcribe/ready/` readiness endpoint.
- [x] Add `prune_jobs` for retention cleanup.
- [x] Fix the `youtube_audio_downloaded` status length, use `TextChoices`, and
  switch YouTube downloads to `yt-dlp`.
- [x] Load `.env`, add `django.contrib.sites` and a default site migration,
  environment-driven hosts/origins, HTTPS toggles, WhiteNoise/static handling,
  console logging, and a migration toggle.
- [x] Update the README, Vast runbook, system design, and PyTorch README.
- [x] Add `transcribeapp` tests for media scoping, callback auth, upload
  limits, status handling, queue claim attribution, and reaping.
- [ ] Vast serverless worker image, dispatcher, and R2/S3 media.
- [x] Cache the inference model per worker process.
- [x] Pad the final chunk with the log-mel silence floor instead of 0.0.
- [ ] KV cache, chunk batching, and precision tuning.
- [x] Optional API-key gate for client-facing endpoints.
- [ ] Per-user job ownership.
- [ ] Account-test cleanup and CI worker-image build.

---

## 1. Key architectural constraint

Vast.ai Serverless is a GPU-worker product. It is not a suitable host for Django, Postgres, Redis, or durable media storage.

Vast serverless workers:

- are ephemeral,
- are request-routed through Vast's serverless engine,
- stop when idle,
- are billed while loading and while ready,
- cannot serve as a stable home for the web application's state.

The target architecture should keep the always-on pieces off Vast and use Vast only for the expensive GPU inference step.

```text
Browser
  → Django on Linode (or another small always-on host)
       → Postgres
       → S3/R2 for audio + MIDI
       → queue / dispatcher
            → Vast serverless GPU worker (MT3 only)
```

This preserves the current web/GPU split while moving the GPU worker to a platform that can scale to zero between jobs.

---

## 2. P0 — fix before exposing the service again

These are the issues that should be addressed before a public URL points at the service.

### 2.1 `/media/` currently exposes the whole project directory

`musictranscription/settings.py:242` sets:

```python
MEDIA_ROOT = BASE_DIR
```

`musictranscription/urls.py:59` serves any path under that root:

```python
re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT})
```

Consequences:

- In a non-Docker deployment, `/media/.env` can serve the `.env` file, including `SECRET_KEY`, database credentials, Redis URL, and Mailgun credentials.
- `/media/db.sqlite3` is reachable.
- Source files under the project root are reachable.
- In Docker `.env` is excluded, but source code is still exposed.

Fix:

- Set `MEDIA_ROOT = BASE_DIR / "media"`.
- Move uploads under `media/audios/` and `media/midi/`.
- Only use Django's `serve` view when `DEBUG=True`.
- In production, serve media through nginx or S3/R2 presigned URLs.
- Keep `.env` outside the repository and never in `MEDIA_ROOT`.

### 2.2 The worker result endpoint is unauthenticated

`/transcribe/_result/` (`transcribeapp/views.py:113-152`) accepts any `audio_midi_id` and file, with no authentication. Anyone can:

- overwrite another job's MIDI,
- replace the audio for a YouTube job,
- set an arbitrary status string,
- upload arbitrary files.

Fix:

- Add a shared worker secret, e.g. `WORKER_SHARED_SECRET`, checked with `hmac.compare_digest` against an `X-Worker-Token` header.
- Keep `_result` off the public route if possible.
- Better: when moving to Vast, use object storage instead of an HTTP callback and delete the callback endpoint.

`/transcribe/metrics/` should also be internal-only or protected.

### 2.3 Queue can attribute results to the wrong job

In `transcribeapp/management/commands/runworker.py:59-81`, the task ID comes from the pub/sub message:

```python
task_id = message["data"].decode()
state = claim_task()
...
self._process_task(task_id, state)
```

But `claim_task()` (`transcribeapp/queue.py:34-56`) pops whatever is at the front of `task:queue`, which may not be the announced task.

Failure modes:

- Two workers receive the same `task:new` broadcast and claim different tasks; one saves its result against the other worker's task ID.
- A worker restarts with a backlog; a new pub/sub message causes it to claim an old task but report it against the new task ID.
- MIDI for song A is stored on the `AudioMIDI` row for song B.

Additionally, pub/sub is fire-and-forget. Tasks enqueued while the worker is down are never claimed until another `task:new` message is published.

Fix:

- Make `claim_task()` return `(task_id, state)` and use the claimed ID everywhere.
- Replace the pub/sub-only worker loop with a blocking `BRPOP`/`BLPOP` loop.
- Optionally use pub/sub only as a low-latency wake-up signal; always drain the queue list.
- Add an integration test with two workers or a backlog to prove result attribution.

### 2.4 Add task reaping, retries, and TTLs

`TASK_PROCESSING_TIMEOUT` exists in `musictranscription/settings.py:129` but is never used. If a worker dies mid-job, the task stays in `task:processing` forever.

Add:

- a reaper that requeues or marks failed any task older than `TASK_PROCESSING_TIMEOUT`,
- a max retry count (the `retries` field exists in the task hash but is never incremented),
- TTLs on completed task hashes,
- a capped dead-letter list,
- non-blocking shutdown: replace `pubsub.listen()` with `get_message(timeout=1)` so SIGTERM can be handled promptly.

### 2.5 Uploads and MIDI files disappear on container recreate

`docker-compose.yml` has no media volume for the web container. `MEDIA_ROOT` lives inside the container layer. `docker compose down`, a redeploy, or a container rebuild loses files while the Postgres rows remain.

Fix for the Linode phase:

- Add named volumes for the audio/MIDI directories.

Fix for Vast:

- Move media to S3-compatible object storage.
- Use `django-storages[s3]` with Cloudflare R2 (S3-compatible, no egress fees) or S3.
- Use presigned GET URLs for workers and presigned PUT URLs for results.
- Add a retention/lifecycle policy so old audio and MIDI are deleted.

### 2.6 Add abuse and cost controls

Currently anyone can upload and start transcriptions without auth, rate limits, size limits, or duration limits. On Vast this means anyone can spend the account's GPU credits.

Add:

- max upload size, enforced in Django and at the proxy,
- allowed MIME types,
- max audio duration (e.g. 10 minutes) checked with `ffprobe` or `torchaudio.info`,
- per-IP or per-token throttling,
- optional API key or login-gated jobs,
- job ownership checks,
- a UUID public job ID instead of enumerable sequential integer IDs.

If this is a public demo, Cloudflare Turnstile plus a strict per-IP daily quota is likely sufficient.

### 2.7 Fix the YouTube path before enabling it publicly

Concrete bugs:

1. `AudioMIDI.status` is `max_length=20` (`transcribeapp/models.py:10`, migration `0004`), but `runworker.py:119` writes `"youtube_audio_downloaded"` (24 characters).
   - PostgreSQL raises `DataError` when `views.py:143-144` saves the status.
   - SQLite silently accepts it.
   - Result: the YouTube flow can work locally and fail in production.
   - Fix the field length / use `TextChoices`, add a migration, and validate allowed statuses.

2. `pytube` is brittle and frequently breaks on current YouTube.
   - Switch to `yt-dlp`.
   - Enforce a duration limit.
   - Reject live streams and playlists.
   - Add a copyright/ToS warning and consider gating the feature.

3. `/generate/` can be called before a YouTube download completes.
   - Guard on status and return a clear error rather than downloading `/media/` and failing.

### 2.8 Production settings and documentation

- `.env` is never loaded. `python-dotenv` is a dependency but no code calls `load_dotenv()`, despite `README.md` instructing users to copy `.env.example` to `.env`.
  - Either load it explicitly in `settings.py` / `manage.py`, or remove the dependency and document shell exports.

- `django.contrib.sites` is not in `INSTALLED_APPS`, but `SITE_ID = 1` is configured for allauth.
  - Password reset and email flows may hit a missing `django_site` table.
  - Add `django.contrib.sites` and migrate, or remove allauth if accounts are not used.

- No `collectstatic` / WhiteNoise / nginx static setup.
  - Admin CSS will 404 with `DEBUG=False`.
  - Add WhiteNoise or configure nginx to serve `/static/`.

- `entrypoint.sh` runs `python manage.py migrate` on every web container start.
  - This races if more than one web replica is running.
  - Move migrations to a dedicated release step.

- Security settings are incomplete for production:
  - `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, and CORS origins should be environment-driven.
  - Enable secure cookies, HSTS, and `SECURE_SSL_REDIRECT` when behind a trusted proxy.
  - `SECURE_PROXY_SSL_HEADER` should only be trusted behind the actual proxy.

- The docs are stale:
  - `README.md` and `docs/vast-deploy.md` still describe Dramatiq, `IS_ASYNC`, `USE_PYTORCH`, and `rundramatiq`.
  - The code uses the custom Redis queue and `runworker`.
  - `README.md` says async processing is optional; in practice Redis is always required for `generate`.
  - `pytorch_mt3/README.md` references files that no longer exist.

---

## 3. P1 — make Vast inference cheaper and faster

These changes directly affect serverless cost, cold-start time, and quality.

### 3.1 Load the model once per worker

`transcribe_audio()` in `transcribeapp/tasks.py:23-30` constructs a new `PyTorchInferenceModel` for every job:

```python
inference_model = PyTorchInferenceModel(checkpoint_path, "mt3")
```

That reloads the 175 MB checkpoint and reinitializes the model for every request.

Fix:

- Cache the model at module/process level.
- Load it once at worker startup.
- On Vast, load it before the worker reports ready.
- Add a warmup inference so the first user request does not pay initialization costs.

### 3.2 Add a KV cache to `generate()`

`pytorch_mt3/pytorch_model.py:530-546` recomputes the entire decoder prefix for every generated token:

```python
for _ in range(max_length - 1):
    seq_len = generated.size(1)
    causal_mask = self._create_causal_mask(seq_len, device)
    logits = self.decoder(generated, encoder_output, self_attn_mask=causal_mask)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
```

This is O(T²) decoding. The original MT3 attention layer has an autoregressive cache; the PyTorch implementation does not.

This is the highest-leverage inference improvement in the project. It should substantially reduce decode time and serverless cost. If only one inference optimization is made, make it this one.

### 3.3 Batch chunks and fix padding

`pytorch_mt3/standalone_inference.py:129-168` processes one 256-frame chunk at a time. Batching a few chunks through the encoder/decoder would improve GPU utilization.

Also, the last chunk is padded with:

```python
chunk_frames = F.pad(chunk_frames, (0, 0, 0, padding))
```

This pads with `0.0`. In log-mel space, silence is approximately `log(1e-5) ≈ -11.5`, so `0.0` looks much louder than silence. This can produce phantom notes at the end of a song.

Fix:

- Pad with the silence value, not `0.0`.
- Pass an encoder padding/attention mask through the encoder and cross-attention.
- Add a parity test against the original JAX preprocessing.

Additional preprocessing discrepancy: the PyTorch STFT uses `center=False` and pads only to the hop boundary, while MT3 uses `pad_end=True`. This drops roughly the last 15 frames (~0.12 s) of audio. Test and align if quality matters.

### 3.4 Use inference precision deliberately

Try BF16/FP16 autocast and `torch.inference_mode()` on the GPU worker. Compare MIDI output against FP32 on a handful of reference songs before adopting it. On a 45M-parameter model this can give a useful speedup with little quality risk, but it needs a quality gate.

### 3.5 Bake the checkpoint and load it strictly

`pytorch_mt3/download_checkpoint.py` downloads the 175 MB checkpoint on first use. That:

- adds cold-start time,
- depends on GitHub Releases remaining available,
- results in every new worker downloading the same file.

Fix:

- Bake the checkpoint into the worker image.
- Verify a checksum at startup.
- Consider storing it in R2/S3 as an alternative source.

`pytorch_mt3/standalone_inference.py:57` uses:

```python
missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
```

It prints the missing count but does not fail on unexpected keys. Add assertions so a broken conversion cannot silently run with random weights.

### 3.6 Pin GPU compatibility and reduce image size

The current stack is `torch 2.6.0+cu124`. That does not support Blackwell GPUs (RTX 5090, B200). Vast may schedule onto one because it is cheap.

Fix:

- Pin the Vast workergroup to Ampere/Ada GPUs: RTX 3090/4090, A10, L4, L40S, A100.
- Or upgrade to a newer CUDA build and test deliberately.

`torchcodec 0.10` is pinned but the code uses `torchaudio.load`, not torchcodec. Torchcodec 0.10 pairs with a much newer torch than 2.6. Remove it unless it is actually needed.

The worker image is also large and slow to pull. Use a lean CUDA/PyTorch base, a multi-stage build, and avoid keeping `build-essential` and `git` in the final image. Push the worker image to GHCR and pin by digest instead of scp'ing code to `/app`.

---

## 4. Vast serverless integration options

### Option A — least rewrite (recommended first)

Keep Django, Redis, and `runworker`. Turn `runworker` into a CPU dispatcher:

1. Web uploads audio to R2 and creates the job.
2. Dispatcher sends an R2 presigned audio URL to a Vast inference endpoint.
3. Vast worker downloads the audio, transcribes, and returns MIDI bytes or uploads the MIDI to R2.
4. Dispatcher saves the result through the existing flow.

Only `transcribe_audio()` changes, plus media storage and a small Vast inference server. Status and progress remain intact. This is the fastest path to a working deployment.

### Option B — cleaner long-term

Remove Redis and the subscriber. Django starts a Vast session, stores the session credentials, and the Vast worker writes MIDI to R2 and calls back or is polled. Fewer moving parts, but more new code and session-state management.

### Option C — fastest experiment

Use the newer Vast **Deployments** beta:

- `@context` loads the model once per worker.
- `@remote` exposes a transcription function.

This is the least infrastructure code, but it is beta, serializes arguments as base64, and is less proven. Good for validating the model on Vast; not the first production dependency.

### Vast-specific gotchas

- Vast's standard PyWorker template clones a public repo and runs `pip install -r requirements.txt` on worker startup. For a Torch environment this is a terrible cold start.
  - Build a dedicated worker image with Torch, ffmpeg, the model, and the checkpoint already present.
  - Alternatively, use a custom template that starts from a prebuilt image.

- The worker cannot reach `http://web:8008/media/...`; that hostname exists only inside the compose network.
  - Use R2/S3 presigned URLs for audio input and MIDI output.
  - Do not rely on base64 audio unless the files are small.

- Vast's SDK has a roughly 600-second worker request timeout.
  - If inference is ~0.5× real-time, a 10-minute song takes ~5 minutes, which fits.
  - Longer audio needs Vast sessions or client-side splitting.
  - Set a max duration for the first version.

- Autoscaling and cost:
  - `cold_workers=0` avoids idle GPU charges but can mean a multi-minute first request after idle.
  - `cold_workers=1` gives a much better UX at the cost of an always-warm worker.
  - Loading time is billed; keep image pull and model load fast.
  - Set `max_workers=1` unless one GPU can genuinely serve concurrent requests.

---

## 5. Suggested rollout order

1. **Fix P0 and deploy the web tier only.**
   - Fix media root, auth, queue attribution, and storage.
   - Deploy Django + Postgres + R2 on Linode.
   - Verify upload/status/media flows with inference mocked.
   - This gets the public API stable without GPU spend.

2. **Build the Vast worker image.**
   - Add a tiny inference HTTP server.
   - Load the model once at startup.
   - Bake the checkpoint.
   - Pin GPU requirements.
   - Add the worker image to CI and push to GHCR.

3. **Wire the dispatcher using Option A.**
   - Run one real song end-to-end.
   - Record cold-start time, inference time, and cost.

4. **Optimize inference.**
   - Add model caching and KV cache.
   - Re-measure.
   - Tune max audio duration and autoscaling from real numbers.

5. **Add operations.**
   - Health/readiness checks that include Redis, DB, worker heartbeat, and model state.
   - Sentry or structured logging.
   - Retention cleanup for old audio/MIDI.
   - A real `transcribeapp` test suite.
   - CI that builds and pushes both web and worker images.

---

## 6. Decisions needed

1. **Public demo or account-gated?**
   - Determines whether we add API keys + ownership, or just throttling + Turnstile.

2. **Keep the Linode web tier?**
   - Recommended. Vast serverless cannot host Django/Postgres/Redis well.

3. **Option A or B for Vast?**
   - A is faster to ship; B is cleaner long-term.

4. **Longest audio to support?**
   - Decides whether a single synchronous Vast request is enough or sessions are required.

---

## 7. File references

- `musictranscription/settings.py`
  - `MEDIA_ROOT = BASE_DIR` (line 242)
  - `TASK_PROCESSING_TIMEOUT` unused (line 129)
  - `SITE_ID = 1` without `django.contrib.sites` (line 174)
- `musictranscription/urls.py`
  - Wildcard media serving (line 59)
- `transcribeapp/views.py`
  - Unauthenticated `_result` (lines 113-152)
  - Status overflow source (lines 143-144)
  - Audio URL construction (lines 84-85)
- `transcribeapp/queue.py`
  - Claim logic (lines 34-56)
  - No TTL on completed tasks (lines 79-94)
- `transcribeapp/management/commands/runworker.py`
  - Message/claim mismatch (lines 59-81)
  - YouTube status length (line 119)
- `transcribeapp/models.py`
  - `status` field (line 10)
- `transcribeapp/tasks.py`
  - Model loaded per request (lines 23-30)
- `pytorch_mt3/pytorch_model.py`
  - No KV cache (lines 530-546)
- `pytorch_mt3/standalone_inference.py`
  - `strict=False` (line 57)
  - Padding with `0.0` (lines 137-141)
- `docker-compose.yml`
  - No media volume for web (lines 30-47)
- `.github/workflows/ci.yml`
  - Only web image built/pushed (lines 90-100)
- `README.md`
  - Stale Dramatiq / `IS_ASYNC` references (lines 58, 95)
- `docs/vast-deploy.md`
  - Stale `rundramatiq` instructions (lines 64-73, 130-134)
