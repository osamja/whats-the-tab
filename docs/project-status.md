# Project Status & Roadmap

_Last updated: 2026-11-24_

## Live Production Architecture

- **Frontend:** [pyaar.ai](https://pyaar.ai) (`pyaar-ai-website` @ `abbafbe`, automatic retry UI).
- **Web & Queue Tier (Linode):** Docker Compose stack (`whatsthetab-linode`):
  - `web` (Django + gunicorn + Redis subscriber, port `127.0.0.1:18009`)
  - `dispatcher` & `dispatcher_secondary` (`manage.py runworker`, both targeting `deployment-whats-the-tab-linode-retry`)
  - `postgres:15-alpine` & `redis:7-alpine` (AOF enabled)
  - Private Linode Object Storage (`SERVE_MEDIA=False`, presigned URLs signed per job claim)
- **GPU Inference Tier (Vast.ai Serverless):**
  - Active deployment: `6147` (`whats-the-tab`, `tag=linode-retry`, endpoint `45874` / `deployment-whats-the-tab-linode-retry`)
  - Worker image: `ghcr.io/osamja/whats-the-tab-worker@sha256:afbb981d488f3cf165cf641274424e2dd0cfb1d8f13f295c72dd451c4eaeeaea`
  - Scaling: `max_workers=4`, `min_load=500`, `cold_workers=2`, `cold_mult=3.0` (2 warm RTX 3060s at ~$0.0644/hr each + 2 cold standby RTX 3060s)

## Recent Milestones

- [x] Queue-level retry handling (`retrying` status, fresh presigned URL per attempt, stale-task reaping)
- [x] Unwrapped Vast serialized dict payloads (`_extract_invalid_audio`) for permanent input rejections
- [x] Production cutover to `deployment-whats-the-tab-linode-retry` with live Playwright smoke check
- [x] Stopped & destroyed old `linode-pool` Vast instances (`61861753`, `61862652`)
- [x] Added `VAST_COLD_WORKERS` support and configured 2 warm + 2 cold standby workers on endpoint `45874`
- [x] Reject over-duration (`> MAX_AUDIO_DURATION_SECONDS`) uploads immediately at `/transcribe/upload/` (HTTP 422) before S3 upload or GPU dispatch
- [x] Disabled `/transcribe/upload_from_yt/` by default (`ENABLE_YOUTUBE_UPLOAD=False`, HTTP 410) and sanitized worker YouTube download errors

## Production Observations (First 22h on Retry Release)

- **Reliability:** 9/9 valid audio transcriptions completed with 0 Vast or infrastructure errors and 0 retries needed.
- **Duration rejections:** 5 real user uploads exceeded the 600s limit (`629.5s`, `1038.1s` ×2, `769.1s` ×2). Rejecting these at `/upload/` prevents redundant S3 writes, queue waits, and Vast round-trips.
- **YouTube endpoint:** 1 direct API call to `/transcribe/upload_from_yt/` failed with YouTube's bot-check error, motivating disabling the endpoint by default.

## Next / Open Items

1. **Ops & Cost Hygiene**
   - Delete 8 unused old Vast deployments (`5788`–`6142`) so only the active production deployment remains.
   - Remove orphan Docker container `whatsthetab-linode-dispatcher-1080` and clean up old `/tmp/whatsthetab-*` / `/tmp/pyaar-retry-release-*` rollout artifacts once rollback is no longer needed.
   - Monitor cold worker availability (one stopped cold worker went `unavail` when its host GPU was rented) and evaluate cost vs. latency (currently ~$93/mo for 2 always-on RTX 3060s at ~15 jobs/day).
2. **Frontend Repo Hygiene**
   - Sync `/home/sammy/workspace/pyaar-ai-website` with `origin/main` (currently 7 commits behind with local edits in `public/llms.txt` and `src/App.js`) and surface the 10-minute audio duration limit directly in the upload UI.
3. **Future Product Ideas**
   - Decide whether to support >10 minute songs (e.g., audio chunking or raising `MAX_AUDIO_DURATION_SECONDS`) if over-limit upload rejections remain common.
