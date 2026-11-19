"""Vast Deployments entry point. Run only from the provisioning environment."""

import asyncio
import base64
import logging
import math
import os
import struct
import tempfile
import wave
from pathlib import Path
from urllib.parse import urlsplit

import requests
from vastai import Deployment
from vastai.data.query import RTX_4090, gpu_name, num_gpus

deployment_name = os.getenv("VAST_DEPLOYMENT_NAME", "whats-the-tab")
deployment_gpu = os.getenv("VAST_GPU_NAME", RTX_4090)
app = Deployment(name=deployment_name)
image = app.image(
    os.getenv("VAST_WORKER_IMAGE", "ghcr.io/osamja/whats-the-tab-worker:latest"),
    storage=30,
)
image.venv("/app/.venv").require(gpu_name == deployment_gpu, num_gpus == 1)
image.env(
    VAST_DEPLOYMENT_NAME=deployment_name,
    VAST_AUDIO_HOST=os.getenv("VAST_AUDIO_HOST", ""),
    MAX_AUDIO_DURATION_SECONDS=os.getenv("MAX_AUDIO_DURATION_SECONDS", "600"),
)
app.configure_autoscaling(max_workers=1, cold_workers=0, max_queue_time=300.0)

# YouTube jobs can produce larger files than the browser upload limit.
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
logger = logging.getLogger(__name__)


@app.context()
class MT3Context:
    async def __aenter__(self):
        import torch

        from transcribeapp.tasks import _get_inference_model

        # PyWorker enables DEBUG on the root logger. HTTP debug logs include
        # presigned query strings.
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        if not torch.cuda.is_available():
            raise RuntimeError("The Vast worker needs a CUDA GPU")
        _get_inference_model()
        return asyncio.Lock()

    async def __aexit__(self, exc_type, exc, tb):
        pass


def _download_audio(url, path):
    allowed_host = os.environ.get("VAST_AUDIO_HOST", "").lower()
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not allowed_host
        or parsed.netloc.lower() != allowed_host
        or not parsed.query
    ):
        raise ValueError("Expected a presigned HTTPS URL on the configured audio host")

    try:
        with requests.get(
            url, stream=True, timeout=(10, 30), allow_redirects=False
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Audio download returned HTTP {response.status_code}")
            size = 0
            with path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Audio exceeds the Vast worker download limit")
                    output.write(chunk)
    except requests.RequestException:
        raise RuntimeError("Audio download failed") from None


def _write_benchmark_audio(path):
    sample_rate = 16000
    samples = bytearray()
    for frequency in (261.63, 329.63, 392, 523.25):
        for sample in range(sample_rate):
            t = sample / sample_rate
            envelope = min(1, t * 50, (1 - t) * 20)
            value = int(12000 * envelope * math.sin(2 * math.pi * frequency * t))
            samples.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        audio.writeframes(samples)


def _transcribe(audio_url):
    from transcribeapp.tasks import transcribe_audio

    with tempfile.TemporaryDirectory() as directory:
        audio_path = Path(directory) / "audio.wav"
        if audio_url is None:
            _write_benchmark_audio(audio_path)
        else:
            _download_audio(audio_url, audio_path)
        midi_path = transcribe_audio(
            str(audio_path),
            directory,
            max_duration=int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "600")),
        )
        return base64.b64encode(Path(midi_path).read_bytes()).decode("ascii")


@app.remote(benchmark_dataset=[{"audio_url": None}], benchmark_runs=3)
async def transcribe(audio_url: str | None) -> str:
    async with app.get_context(MT3Context):
        try:
            return await asyncio.to_thread(_transcribe, audio_url)
        except Exception as exc:
            # A failed request can contain a presigned URL in its exception.
            logger.error("MT3 request failed (%s)", type(exc).__name__)
            raise


if __name__ == "__main__":
    if not os.getenv("VAST_AUDIO_HOST"):
        raise SystemExit("Set VAST_AUDIO_HOST to the hostname in a presigned audio URL")
    app.ensure_ready()
