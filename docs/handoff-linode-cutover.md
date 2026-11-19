# Handoff for the Linode cutover

Deploy the tested RTX 3060-backed transcription stack on Linode. Prove the
audio-to-MIDI flow on a Linode-only origin before changing public nginx
routing. The repo is open source. Keep docs, commits, workflow logs, and issue
reports suitable for public visibility. Store real credentials and presigned
URLs outside Git and redact them from test evidence.

```mermaid
flowchart LR
    Browser["Browser"] --> Nginx["Linode nginx"]
    subgraph Linode
        Nginx --> Web["Django + subscriber"]
        Web --> DB[("PostgreSQL")]
        Web -->|"Enqueue"| Redis[("Redis")]
        Redis -->|"Claim"| Dispatcher["CPU dispatcher"]
        Dispatcher -->|"Authenticated MIDI callback"| Web
    end
    Web --> Bucket[("Private Linode Object Storage")]
    Dispatcher -->|"Fresh signed audio URL"| Vast["Vast RTX 3060"]
    Vast -->|"Fetch audio"| Bucket
    Vast -->|"Return MIDI"| Dispatcher
```

## Starting point

| Item | Last verified state |
| --- | --- |
| Code | Vast dispatch and the GPU comparison are on `origin/main` through `147c33f` |
| Local Compose | 27 tests passed. Local RTX 4090 completed three recordings and recovered after a corrupt upload |
| Vast staging | Isolated desktop staging used the real private Linode bucket, Vast, and the authenticated Django result callback |
| RTX 3060 | Deployment `whats-the-tab-3060`, ID `5789`, endpoint ID `45265`, endpoint name `deployment-whats-the-tab-3060-default` |
| RTX 4090 comparison | Deployment `whats-the-tab`, ID `5788`, endpoint ID `45264` |
| GPU image | Public pull by digest was verified for `ghcr.io/osamja/whats-the-tab-worker@sha256:890de0cb5e7c477ccb7397afd8556af86a7914c4f19c63ff76dac68e823ce49a` |
| Vast worker state | Both deployments were stopped after the tests and both rented instances exited |
| Linode Object Storage | A temporary object returned HTTP 403 unsigned and HTTP 200 with a signed GET. The probe object was removed |
| Public routing | Last inspection found `pyaar.ai` transcription locations pointing to `vast-ai-backend`. The cutover had not started |

The two GPUs returned byte-identical MIDI for both files. The 3060 produced
four notes for the melody and 730 notes for the 71-second piano recording.
Its end-to-end cold melody took 392.68 seconds and its warm piano took
37.38 seconds. The 4090 took 438.28 seconds and 17.24 seconds respectively.
The selected 3060 host was about $0.070/hr versus $0.486/hr for the 4090.
The estimated warm job cost was 69% lower on the 3060, excluding idle time.
See the [Vast runbook](vast-serverless.md) for the full comparison.

## 1. Inventory and prepare Linode

- Use the existing SSH connection helper on the owner's desktop
  (`~/Work/linode.sh`) or the SSH session already on Linode. Last inspection saw
  the proxy Compose project under `~/workspace/nginx_sammyjaved_proxy`, with
  a `my-site-proxy` nginx service and a `prod_network`. Recheck names, ports,
  mounted config, and the active upstream using `docker compose ls`,
  `docker ps`, `ss -lnt`, and `nginx -T` inside the proxy container. Filter
  or redact config output before sharing it.
- Use a real Git checkout at a recorded revision. The directory named
  `~/workspace/whats-the-tab` existed during the prior inspection and did
  not have Git metadata. Inspect its contents before reusing it and choose
  a new directory if needed.
- Determine where the legacy service stores uploads, MIDI, job records, and
  queued work. If existing links or jobs must survive, agree on a migration
  or a way to route legacy job IDs before switching traffic. A fresh database
  can reuse old numeric IDs. Keep database and queue credentials independent
  of other apps on the host. Back up any production state being changed.
- Confirm the production browser's actual URL and payload contract, including
  upload size, CORS, auth, download URLs, and whether it uses YouTube or
  legacy `/transcribe/download/` and `/transcribe/download_wav/` routes.
  The current backend offers `/transcribe/download_midi/<id>/` and
  `/transcribe/midi/<id>/`. Resolve client compatibility before cutover.
- The verified Vast key lives in the owner's desktop `.env`. Arrange a
  secure transfer or provision a dispatcher key on Linode. Restrict secret
  file permissions to the service account. Avoid printing `.env`,
  `docker compose config` with secrets expanded, or a signed URL.
- Agree on the acceptable first-request latency. Zero standby workers kept
  the staging configuration small, but both observed cold starts exceeded
  six minutes. Keep the 600-second dispatcher deadline and the observed cold
  timing in view when choosing a production standby and timeout policy.

## 2. Bring up a private Linode origin

Use Compose with a Linode-specific override. The checked-in Compose file is
the development baseline. Before starting it on Linode, configure the following.

- Run a dedicated PostgreSQL, Redis, and Django stack with persistent volumes.
  Remove the development host publishes for PostgreSQL and Redis. A
  loopback Redis listener already occupied port 6379 during the prior
  inspection. Use a non-default database password and a matching
  `DATABASE_URL`. Keep Django private to the Compose networks or bind it to
  a free loopback port for staging. Attach nginx to a shared network and use
  a unique DNS alias when ready for cutover. Use the nginx container's
  view of the network to verify reachability.
- Run `web` with a strong `SECRET_KEY`, a separate `WORKER_SHARED_SECRET`,
  `DEBUG=False`, and `SERVE_MEDIA=False`. Wire the production host and
  origin settings into the container explicitly. nginx supplies the public
  HTTPS redirect and forwards the original `Host` and
  `X-Forwarded-Proto`. If enabling Django's HTTPS redirect, account for it
  in both the internal HTTP callback and the Compose readiness probe before
  deploying. Enable secure cookies for the public HTTPS origin as applicable.
  Confirm that any `TRANSCRIBE_API_KEY` policy agrees with the clients.
- Configure the private bucket using [object-storage.md](object-storage.md).
  Give web and dispatcher the same bucket, region, endpoint, and addressing
  style. For Linode retain the `when_required` checksum settings.
  Check the exact signed GET hostname against the worker's `VAST_AUDIO_HOST`.
  Ensure the signed URL lifetime exceeds the dispatcher request deadline.
- Add a supervised CPU dispatcher service running `python manage.py runworker`
  with the pinned optional SDK in [requirements.txt](../deploy/vast/requirements.txt).
  Build it from the web runtime or equivalent Python 3.11 environment with
  the same checkout. Include ffmpeg if supporting YouTube downloads. Start
  it after Django is healthy. Compose web already starts the subscriber.
  The local `gpu` profile is for executing MT3 on the Docker host.
- The dispatcher shares `DATABASE_URL`, `REDIS_URL`, storage credentials,
  and `WORKER_SHARED_SECRET` with web. Set `WEB_URL=http://web:8008` when both
  services use the same Compose network. Set the verified Vast endpoint
  name as `VAST_ENDPOINT_NAME`. Pass `VAST_API_KEY` only to the dispatcher
  and the provisioning environment. The Vast GPU receives a presigned URL
  and carries no storage, PostgreSQL, Redis, or Django credentials.
- Check Compose syntax with `docker compose config --quiet`. Start the
  services with the actual override enabled and use `docker compose ps` and
  `/transcribe/ready/` to confirm PostgreSQL and Redis are reachable and
  the dispatcher is heartbeating. Use a disposable Redis for the Django
  test suite because queue tests clear the configured test Redis database.

## 3. Activate the 3060 and validate the Linode path

1. Inspect deployment `5789`. Verify the image digest, GPU, one-worker
   cap, and signed audio host. Start the existing deployment in Vast. If
   recreating it, follow the [Vast runbook](vast-serverless.md) and explicitly
   set `VAST_DEPLOYMENT_NAME=whats-the-tab-3060` and `VAST_GPU_NAME="RTX 3060"`.
   Read back the resulting endpoint name before configuring the dispatcher.
2. Test the private Linode origin from the host or via an SSH tunnel using a
   disposable, cleared-for-testing audio file. Use an upload of a four-second
   melody when available, then a representative song on the same worker.

   ```bash
   uv run python scripts/smoke_compose.py \
     --url http://127.0.0.1:18009/transcribe --timeout 650 \
     /path/to/test-melody.wav /tmp/linode-melody.midi
   ```

   Substitute the actual private origin. The smoke script assumes the
   public-facing test endpoints accept requests without a client API key.
   Where a client key is required, exercise the same requests with approved
   client credentials or use a private staging instance. The callback still
   needs to reject requests without the worker token.
3. Confirm the job moves from queued to processing to completed, with a
   downloadable, parseable MIDI containing notes. Verify data lands in the
   intended database and private bucket, unsigned retrieval fails, and the
   dispatcher succeeds via its internal callback. Record cold and warm
   enqueue-to-download times, any limit hit, and the selected hourly rate.
4. Run a malformed audio test and then a valid request to confirm recovery.
   If required for the client, test YouTube upload and the subsequent
   transcription separately. Clean up disposable test jobs and media using
   their recorded IDs. Keep the old nginx route active through this phase.

## 4. Cut over and verify rollback

1. Capture the exact active nginx route config and upstream membership.
   In the source Compose/proxy checkout, prepare the minimal transcription
   route change to the verified Linode service alias. Preserve the
   frontend's paths, trailing slashes, request limits, headers, TLS, and
   routing for other applications. Keep the internal result callback on
   the private network where practical.
2. Verify service name resolution and HTTP readiness from inside nginx.
   Validate the candidate configuration with `nginx -t`. Apply it through
   the project's normal reload/redeploy process after the timing and
   legacy-job handling are agreed.
3. Run a public HTTPS smoke test and test the actual frontend. Confirm upload,
   status, MIDI download, CORS/auth, and error behavior. Watch nginx, Django,
   dispatcher, and Vast logs for route errors, repeated timeouts, or failures.
   Spot-check adjacent services on the same proxy.
4. If the public path fails or latency exceeds the agreed threshold, restore
   the prior nginx config, validate, reload, and verify the old route. Keep the
   new database, bucket objects, and queued jobs available for reconciliation.
   Avoid treating new job IDs as legacy IDs. If stopping the test backend,
   stop the Vast **deployment** and confirm its worker exits. Stopping just
   the rented instance caused the autoscaler to restart it during staging.

Done means the public client completes a real HTTPS upload-to-MIDI request
through Linode and Vast, the result stays in private storage, the measured
latency is acceptable, and the operator can restore the old route. Record
the deployed commit, image digest, endpoint, Compose and nginx config
locations, smoke test outcomes, and rollback command in the handoff notes.

## Linode Compose configuration

Run these commands from the checkout on Linode. Use Compose v2.24.4 or
newer for the port overrides. Keep the environment file outside the
checkout and readable only by the deployment account.

Set `SECRET_KEY`, `WORKER_SHARED_SECRET`, and `LINODE_DB_PASSWORD` to
independent random values. Use a URL-safe database password, such as
`openssl rand -hex 32`. Supply `VAST_API_KEY` and the five required
`AWS_*` variables from the private bucket configuration. The override
defaults to the verified RTX 3060 endpoint and stages Django on
`127.0.0.1:18009`. Set `LINODE_WEB_PORT` if that port is in use.
The frontend checkout sends requests without a client API key. Leave
`TRANSCRIBE_API_KEY` empty unless the frontend is updated to supply one.

```bash
compose=(docker compose -p whatsthetab-linode --env-file "$LINODE_ENV_FILE"
  -f docker-compose.yml -f docker-compose.linode.yml)
"${compose[@]}" config --quiet
"${compose[@]}" up --build -d --wait
"${compose[@]}" ps
curl --fail --silent http://127.0.0.1:18009/transcribe/ready/
```

Wait for `workers_online` to reach 1 before submitting the smoke job.
PostgreSQL and Redis stay on the project network. For nginx verification,
attach only web to the proxy network using a unique alias after the private
smoke test. Keep the attachment in the deployed Compose configuration so
it survives container replacement.
