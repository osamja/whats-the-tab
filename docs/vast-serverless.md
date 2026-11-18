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
3. From the repository root, register the deployment. The pinned SDK uses a
   Python 3.12 API during provisioning. The Django dispatcher and worker
   image use Python 3.11.

   ```bash
   uv run --no-project --python 3.12 --with-requirements deploy/vast/requirements.txt \
     python -m deploy.vast.deployment
   ```

4. Find the endpoint associated with `whats-the-tab` in the Vast console.
   Use its displayed endpoint name for `VAST_ENDPOINT_NAME`. The deployment
   initially caps scaling at one RTX 4090 and requests zero cold workers.
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
worker receives only the audio host and duration setting in its deployment
configuration. YouTube download jobs still run on the dispatcher, where
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
