#!/usr/bin/env python3
"""Download PyTorch MT3 checkpoint from GitHub Releases if not present."""

import urllib.request
from pathlib import Path

CHECKPOINT_FILENAME = "mt3_pytorch_checkpoint.pt"
RELEASE_URL = "https://github.com/osamja/whats-the-tab/releases/download/v1.0-pytorch-checkpoint/mt3_pytorch_checkpoint.pt"


def ensure_checkpoint(checkpoint_path=None):
    """Ensure checkpoint exists, downloading if necessary.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses default location.

    Returns:
        Path to the checkpoint file.
    """
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent / CHECKPOINT_FILENAME
    else:
        checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        return checkpoint_path

    # Download the pre-converted checkpoint when the release asset exists.
    print("Downloading checkpoint from GitHub Releases (~175 MB)...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(RELEASE_URL, checkpoint_path)
        print(f"✓ Downloaded to {checkpoint_path}")
    except Exception as release_error:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        print("Release download failed; rebuilding from the public MT3 checkpoint...")
        try:
            from .prepare_checkpoint import prepare_checkpoint

            prepare_checkpoint(checkpoint_path)
            print(f"✓ Converted checkpoint saved to {checkpoint_path}")
        except Exception as conversion_error:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise FileNotFoundError(
                "Failed to download or convert the checkpoint.\n"
                f"Release URL: {RELEASE_URL}\n"
                f"Destination: {checkpoint_path}\n"
                f"Release error: {release_error}\n"
                f"Conversion error: {conversion_error}"
            ) from conversion_error

    return checkpoint_path


if __name__ == "__main__":
    print(f"Checkpoint ready at: {ensure_checkpoint()}")
