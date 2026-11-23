# Vast Serverless deployment

Django, PostgreSQL, Redis, the Redis subscriber, and the dispatcher run on
Linode. The dispatcher reads the audio record and signs a fresh object storage
URL on each claim. Vast runs MT3 and returns MIDI to the dispatcher, which
uploads it through Django's authenticated result callback.

## Provision the GPU endpoint

1. Build the root `Dockerfile` and publish it with the Worker Image workflow
   or Docker. Use an immutable image tag or digest accessible to Vast.
   The image includes the converted MT3 checkpoint and its CUDA dependencies.
2. In the provisioning shell set `VAST_API_KEY`, `VAST_WORKER_IMAGE`, and
   `VAST_AUDIO_HOST`. The audio host is the exact hostname from a signed GET
   URL, for example `whats-the-tab-media.us-lax-4.linodeobjects.com`. The
   worker accepts HTTPS on that host and follows no redirects. Match
   `MAX_AUDIO_DURATION_SECONDS` to the web and dispatcher settings.
   To run a hardware comparison alongside the default deployment, set
   `VAST_DEPLOYMENT_NAME=whats-the-tab-3060` and `VAST_GPU_NAME="RTX 3060"`.
   To admit multiple GPU models in one deployment, use a comma-separated
   list such as `VAST_GPU_NAME="RTX 3060,GTX 1080 Ti"`. Validate a private
   audio-to-MIDI job on the GTX 1080 Ti before relying on it for capacity.
   `VAST_DEPLOYMENT_TAG` selects the deployment tag. Set
   `VAST_MAX_WORKERS` to cap recruitment, `VAST_MIN_LOAD` to set the active
   performance target, and `VAST_COLD_WORKERS` to keep stopped standby workers.
   Verify the resulting worker count after the endpoint has been idle.
3. From the repository root, register the deployment. The pinned SDK uses a
   Python 3.12 API during provisioning. The Django dispatcher and worker
   image use Python 3.11.

   ```bash
   uv run --no-project --python 3.12 --with-requirements deploy/vast/requirements.txt \
     python -m deploy.vast.deployment
   ```

4. Find the deployment's endpoint in the Vast console. Use its displayed
   endpoint name for `VAST_ENDPOINT_NAME`. Each deployment caps scaling at
   one GPU and requests zero cold workers. The default GPU is an RTX 4090.
   Its autoscaling benchmark runs a generated four-second melody.

See the [Vast Deployments configuration documentation](https://docs.vast.ai/guides/serverless/deployments/configuration).

## Start the Linode services

Configure Django, gunicorn, PostgreSQL, Redis, and `manage.py subscriber`.
Use the [object storage configuration](object-storage.md) with a private bucket
and `SERVE_MEDIA=False`. For Compose, the `web` entry point runs migrations
and starts the subscriber. For a host installation, run migrations and
`collectstatic`, and supervise the subscriber and gunicorn separately.

Run the dispatcher on Linode with the same database, Redis, bucket credentials,
and `WORKER_SHARED_SECRET` as Django. Set `WEB_URL` to the Django origin
reachable from the dispatcher. If Django runs in Compose and the dispatcher
runs directly on the same host, use the host-published addresses and port
8008 for `WEB_URL`.

In the dispatcher's environment, set `VAST_API_KEY` and the endpoint name.
Install the SDK into the project's environment after syncing and start the
dispatcher with that environment's Python.

```bash
uv sync --frozen --no-dev
uv pip install -r deploy/vast/requirements.txt
.venv/bin/python manage.py runworker
```

Keep the Vast key in the provisioning and dispatcher environments. The GPU
worker receives the deployment name, audio host, and duration setting in its
deployment configuration. YouTube download jobs still run on the dispatcher, where
`ffmpeg` is needed if the source requires postprocessing.

To switch back, clear `VAST_ENDPOINT_NAME` and run `runworker` on a host
with the local MT3 dependencies and checkpoint. Stop the Vast dispatcher
before switching if all new jobs should use local inference.

## Validate the first deployment

1. Upload a four-second melody and enqueue transcription. Confirm the job
   reaches `completed` and the downloaded MIDI contains notes.
2. Repeat with a real song. Record cold and warm latency, GPU cost, and
   whether queueing or processing reached the dispatcher's 600-second deadline.
   Use those measurements to tune the duration limit and standby capacity.
3. Fetch a media object anonymously and expect access to be denied. Fetch
   the same object with a fresh presigned URL and expect success.
4. Verify the local fallback and confirm that callback authentication and
   stale Redis task recovery still work.

Keep the signed URL lifetime longer than the maximum dispatch wait. The
dispatcher sends start and finish progress for Vast jobs. Local inference
continues to report per-chunk progress.

### Transcription retries

The Linode queue owns the retry budget. A transient Vast failure or a lost
dispatcher lease can schedule the same job ID twice, roughly 30 seconds
and two minutes later with jitter. Each attempt signs a new audio URL and
calls the same endpoint, allowing Vast to route it again. The status API
reports `retrying`, and the frontend keeps polling the original job.

An unreadable file or an audio limit violation ends the job immediately.
Deploy the matching worker image and Vast module to return these input
failures explicitly. If a result callback times out, the dispatcher checks
whether Django saved the MIDI before resending the file. Django keeps the
first successfully saved MIDI.

In staging, a connection loss injected after a real Vast request started
put the job into `retrying`. A replacement dispatcher claimed the same ID
after the delay and produced a private, downloadable four-note MIDI.

In a separate drill, we destroyed an isolated Vast GPU while it was
transcribing a 70-second piano fixture. The SDK routed the request to the
other worker within the same application attempt. The original job
completed with a 338-note MIDI. The two drills verified recovery both
inside the SDK and after a failure reaches the Linode queue.

### RTX 4090 and RTX 3060 staging comparison

Both endpoints ran the same worker image digest and processed the same inputs
through separate staging queues. Both returned byte-identical MIDI for the
four-second melody (4 notes) and the 71-second piano recording (730 notes).

| GPU | Selected host rate | Cold melody | Warm piano | Estimated cost of warm run |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 | $0.486/hr | 438.28 s | 17.24 s | $0.00233 |
| RTX 3060 | $0.070/hr | 392.68 s | 37.38 s | $0.00073 |

Times are measured from enqueue to MIDI download. The first request included
image pull and worker initialization on each host. The estimated cost is the
host's hourly rate multiplied by elapsed time and excludes idle time. In this
test the RTX 3060 took about 2.2 times as long on warm piano and cost about
69% less. Cold startup was several minutes on both hosts.

After a staging run, stop the deployment through Vast to release its workers.
The autoscaler can restart an instance stopped individually while the
deployment remains active.
