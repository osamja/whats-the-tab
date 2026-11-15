"""Download and convert the public MT3 JAX checkpoint.

The converted PyTorch checkpoint used to be published as a GitHub release. If
that download is unavailable, this module fetches the original checkpoint from
the public ``gs://mt3`` bucket and converts it using
``convert_jax_to_pytorch``.
"""

import json
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

BUCKET = "mt3"
PREFIX = "checkpoints/mt3/"
DEFAULT_OUTPUT = Path(__file__).parent / "mt3_pytorch_checkpoint.pt"


def _list_checkpoint_objects():
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
    token = None
    objects = []

    while True:
        query = f"{url}?prefix={PREFIX}&maxResults=1000&fields=items(name,size),nextPageToken"
        if token:
            query += f"&pageToken={token}"
        with urllib.request.urlopen(query, timeout=60) as response:
            data = json.load(response)
        objects.extend(data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            break

    return objects


def _download_object(item, destination):
    name = item["name"]
    if name.endswith("/"):
        return

    relative = name[len(PREFIX):]
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    media_url = f"https://storage.googleapis.com/{BUCKET}/{quote(name, safe='/')}"
    with urllib.request.urlopen(media_url, timeout=300) as response:
        target.write_bytes(response.read())


def download_jax_checkpoint(destination):
    """Download the original T5X checkpoint into ``destination``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    objects = _list_checkpoint_objects()
    print(f"Downloading {len(objects)} checkpoint objects...")

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(_download_object, item, destination)
            for item in objects
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % 50 == 0:
                print(f"  downloaded {index}/{len(objects)}")

    print("Download complete")


def prepare_checkpoint(output_path=DEFAULT_OUTPUT):
    """Ensure a converted PyTorch checkpoint exists at ``output_path``."""
    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    jax_dir = Path(tempfile.mkdtemp(prefix="mt3-jax-"))
    try:
        download_jax_checkpoint(jax_dir)
        from .convert_jax_to_pytorch import convert_checkpoint

        convert_checkpoint(str(jax_dir), str(output_path), verbose=True)
    finally:
        shutil.rmtree(jax_dir, ignore_errors=True)

    if not output_path.exists():
        raise RuntimeError("Checkpoint conversion did not produce an output file")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    print(f"Checkpoint ready at: {prepare_checkpoint(args.output)}")
