# Object Storage

The app supports any S3-compatible object store through `django-storages`. When
`AWS_STORAGE_BUCKET_NAME` is set, uploaded audio and generated MIDI are stored in
the bucket instead of the local `media/` directory.

This is required when the web tier and GPU workers run on different hosts, such
as Django on Linode and inference on Vast.ai.

## Why object storage

- Vast workers cannot access the web container's filesystem.
- Audio can be fetched with a time-limited presigned URL.
- Generated MIDI can be downloaded without making the bucket public.
- Media survives web container redeploys and scaling.

Keep the bucket private. The backend generates presigned GET/PUT URLs; no public
bucket or CORS configuration is needed for the current server-to-server flow.

## Linode Object Storage

### 1. Create the bucket

In Linode Cloud Manager:

1. Open **Object Storage**.
2. Create a bucket, for example `whats-the-tab-media`.
3. Choose the region closest to the web host.
4. Leave versioning off unless you specifically need it.

### 2. Create an access key

1. Open **Object Storage → Access Keys**.
2. Create a key with **Read/Write** permissions.
3. Scope it to the bucket if using limited permissions.
4. Copy the access key and secret key immediately; the secret is shown once.

### 3. Configure the app

Add these values to `.env`:

```bash
AWS_STORAGE_BUCKET_NAME=whats-the-tab-media
AWS_S3_ENDPOINT_URL=https://<region-or-cluster>.linodeobjects.com
AWS_S3_REGION_NAME=us-east-1
AWS_ACCESS_KEY_ID=<access-key>
AWS_SECRET_ACCESS_KEY=<secret-key>
AWS_S3_ADDRESSING_STYLE=virtual
AWS_QUERYSTRING_EXPIRE=3600
AWS_REQUEST_CHECKSUM_CALCULATION=when_required
AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
SERVE_MEDIA=False
```

Use the endpoint shown for your bucket. For example, a Los Angeles bucket may use:

```bash
AWS_S3_ENDPOINT_URL=https://us-lax-4.linodeobjects.com
AWS_S3_REGION_NAME=us-east-1
```

Linode's S3 API uses the cluster name in the endpoint hostname. In our setup,
`us-east-1` is the region value used for request signing.

The two checksum settings are important with boto3 1.36 and later. Without
them, Linode uploads can fail with `SignatureDoesNotMatch`,
`MissingContentLength`, or `NotImplemented`.

### 4. Verify

Run a small save/read/delete cycle:

```bash
uv run python - <<'PY'
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "musictranscription.settings")
import django
django.setup()

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

name = default_storage.save("audios/storage-test.txt", ContentFile(b"ok"))
print("saved:", name)
print("exists:", default_storage.exists(name))
print("read:", default_storage.open(name, "rb").read())
print("url:", default_storage.url(name))
default_storage.delete(name)
print("deleted:", not default_storage.exists(name))
PY
```

A successful run uploads, reads, generates a presigned URL, and deletes the
object. Fetching the presigned URL with `curl` should return `200`.

## Cloudflare R2 and AWS S3

The same settings work for other S3-compatible providers. For R2, use:

```bash
AWS_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
AWS_S3_REGION_NAME=auto
AWS_S3_ADDRESSING_STYLE=virtual
```

For AWS S3, use the bucket's regional endpoint and region, and normal IAM
credentials.

## Docker Compose

The current Compose file does not pass `AWS_*` variables into the `web`
container. Running directly with `uv run` loads `.env`; running through Docker
Compose currently keeps using local `media/`.

To use object storage with Compose, add the variables to the `web` service's
`environment` block or an `env_file` entry.

## Retention

`python manage.py prune_jobs --days 7` deletes old job records. `django-cleanup`
then deletes the associated audio and MIDI objects from the bucket. You can also
add a bucket lifecycle policy as a backstop.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `SignatureDoesNotMatch` on upload | Missing the boto3 checksum settings above |
| `MissingContentLength` on upload | Missing the boto3 checksum settings above |
| `403` on a presigned URL | Expired URL, wrong bucket/key, or key permissions |
| `/media/...` returns `404` | Expected when `SERVE_MEDIA=False` and S3 is enabled |
| Worker cannot download audio | Presigned URL expired or the endpoint is unreachable |
| Docker still writes to `media/` | `AWS_*` variables are not passed to the container |
| `NoSuchBucket` | Bucket name or endpoint region/cluster does not match |

## Security notes

- `.env` is gitignored. Never commit access keys or secret keys.
- Keep the bucket private; presigned URLs are bearer tokens until they expire.
- Use short `AWS_QUERYSTRING_EXPIRE` values for sensitive files.
- Rotate access keys periodically.
