"""Dispatch inference to an existing Vast Deployments endpoint."""

import asyncio
import base64
import binascii
import io
from pathlib import Path
from urllib.parse import urlsplit

from django.conf import settings

from .tasks import InvalidAudioError


class RetryableTranscriptionError(RuntimeError):
    """The same job may succeed on a later attempt."""


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
        raise RetryableTranscriptionError("Vast transcription timed out.") from None
    except Exception:
        raise RetryableTranscriptionError("Could not reach Vast for transcription.") from None

    status = response.get("status")
    if status in (400, 401, 403, 413, 422):
        raise RuntimeError("Vast rejected the transcription request.")
    result = response.get("response") or {}
    outcome = result.get("result") if isinstance(result, dict) else None
    value = outcome.get("ok") if isinstance(outcome, dict) else None
    if status == 200 and isinstance(value, dict) and "invalid_audio" in value:
        raise InvalidAudioError(value["invalid_audio"])
    if status != 200 or not isinstance(value, str):
        raise RetryableTranscriptionError("Vast could not complete the transcription.")
    return value


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
        raise RetryableTranscriptionError("Vast returned invalid MIDI") from None

    output_path = Path(output_dir) / "transcription.midi"
    output_path.write_bytes(midi)
    return output_path
