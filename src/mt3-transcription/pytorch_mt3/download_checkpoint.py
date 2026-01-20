#!/usr/bin/env python3
"""
Download PyTorch MT3 checkpoint from GitHub Releases if not present.

This script automatically downloads the model checkpoint file if it doesn't
exist locally. It's designed to be called automatically by the inference code.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional


CHECKPOINT_FILENAME = "mt3_pytorch_checkpoint.pt"
GITHUB_RELEASE_URL = "https://github.com/osamja/whats-the-tab/releases/download/v1.0-pytorch-checkpoint/mt3_pytorch_checkpoint.pt"
EXPECTED_SIZE_MB = 175  # Approximate size in MB
EXPECTED_SIZE_BYTES = EXPECTED_SIZE_MB * 1024 * 1024


def get_checkpoint_path() -> Path:
    """Get the path to the checkpoint file."""
    return Path(__file__).parent / CHECKPOINT_FILENAME


def download_with_progress(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading checkpoint from GitHub Releases...")
    print(f"URL: {url}")
    print(f"Destination: {output_path}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100.0)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)"
            )
            sys.stdout.flush()

    try:
        # Create temp file first
        temp_path = output_path.with_suffix('.pt.tmp')
        urllib.request.urlretrieve(url, temp_path, reporthook)
        print()  # New line after progress

        # Rename to final name
        temp_path.rename(output_path)
        print(f"✓ Download complete: {output_path}")

    except Exception as e:
        print(f"\n✗ Download failed: {e}", file=sys.stderr)
        if temp_path.exists():
            temp_path.unlink()
        raise


def verify_checkpoint(checkpoint_path: Path) -> bool:
    """Verify that the checkpoint file is valid."""
    if not checkpoint_path.exists():
        return False

    file_size = checkpoint_path.stat().st_size

    # Check if size is reasonable (within 10% of expected)
    size_diff = abs(file_size - EXPECTED_SIZE_BYTES) / EXPECTED_SIZE_BYTES
    if size_diff > 0.1:
        print(
            f"Warning: Checkpoint size ({file_size / (1024*1024):.1f} MB) "
            f"differs significantly from expected ({EXPECTED_SIZE_MB} MB)",
            file=sys.stderr
        )
        return False

    return True


def ensure_checkpoint(checkpoint_path: Optional[Path] = None) -> Path:
    """
    Ensure the checkpoint file exists, downloading it if necessary.

    Args:
        checkpoint_path: Optional custom path to checkpoint. If None, uses default.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If checkpoint cannot be downloaded.
    """
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path()
    else:
        checkpoint_path = Path(checkpoint_path)

    # Check if checkpoint already exists and is valid
    if checkpoint_path.exists():
        if verify_checkpoint(checkpoint_path):
            print(f"✓ Checkpoint already exists: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"⚠ Existing checkpoint appears corrupted, re-downloading...")
            checkpoint_path.unlink()

    # Download checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_with_progress(GITHUB_RELEASE_URL, checkpoint_path)
    except Exception as e:
        error_msg = (
            f"Failed to download checkpoint from {GITHUB_RELEASE_URL}\n"
            f"Error: {e}\n\n"
            f"Please download manually:\n"
            f"1. Visit: https://github.com/osamja/whats-the-tab/releases\n"
            f"2. Download '{CHECKPOINT_FILENAME}'\n"
            f"3. Place it at: {checkpoint_path}"
        )
        raise FileNotFoundError(error_msg) from e

    # Verify downloaded file
    if not verify_checkpoint(checkpoint_path):
        checkpoint_path.unlink()
        raise FileNotFoundError(
            f"Downloaded checkpoint appears corrupted. Please try again or download manually."
        )

    return checkpoint_path


def main():
    """Command-line interface for downloading the checkpoint."""
    checkpoint_path = ensure_checkpoint()
    print(f"\n✓ Checkpoint ready at: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
