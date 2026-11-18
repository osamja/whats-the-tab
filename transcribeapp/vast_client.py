"""Dispatch inference to an existing Vast Deployments endpoint."""

import asyncio
import base64
import binascii
import io
from pathlib import Path
from urllib.parse import urlsplit

from django.conf import settings


def check_vast_configuration():
    if not settings.VAST_API_KEY:
        raise RuntimeError("VAST_API_KEY is required when VAST_ENDPOINT_NAME is set")
    if not settings.AWS_STORAGE_BUCKET_NAME:
        raise RuntimeError("Vast inference requires shared object storage")
    try:
        from vastai import Serverless
    except ImportError:
        Serverless = None
    if Serverless is None:
        raise RuntimeError("Install deploy/vast/requirements.txt for Vast inference") from None


async def _request_transcription(audio_url):
    from vastai import Serverless

    try:
        async with asyncio.timeout(600):
            async with Serverless(api_key=settings.VAST_API_KEY) as serverless:
                endpoint = await serverless.get_endpoint(settings.VAST_ENDPOINT_NAME)
                response = await endpoint.request(
                    "/remote/transcribe",
                    {"kwargs": {"audio_url": audio_url}},
                    retry=False,
                )
    except TimeoutError:
        raise RuntimeError("Vast transcription timed out") from None
    except Exception:
        raise RuntimeError("Vast request failed; check the endpoint and worker logs") from None

    try:
        encoded_midi = response["response"]["result"]["ok"]
        if response["status"] != 200 or not isinstance(encoded_midi, str):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Vast transcription failed; check the worker logs") from None
    return encoded_midi


def transcribe_via_vast(task_id, output_dir):
    from mido import MidiFile

    from .models import AudioMIDI

    # Sign on every claim so the URL gets its full lifetime after queueing or reaping.
    audio_file = AudioMIDI.objects.get(pk=task_id).audio_file
    if not audio_file:
        raise RuntimeError("No audio file is available for transcription")
    audio_url = audio_file.url
    parsed_url = urlsplit(audio_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc or not parsed_url.query:
        raise RuntimeError("Vast inference requires a presigned HTTPS audio URL")

    encoded_midi = asyncio.run(_request_transcription(audio_url))
    try:
        midi = base64.b64decode(encoded_midi, validate=True)
        parsed_midi = MidiFile(file=io.BytesIO(midi))
        if not parsed_midi.tracks:
            raise ValueError
    except (binascii.Error, EOFError, OSError, ValueError, KeyError, IndexError):
        raise RuntimeError("Vast returned invalid MIDI") from None

    output_path = Path(output_dir) / "transcription.midi"
    output_path.write_bytes(midi)
    return output_path
