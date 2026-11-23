import hmac
import os
import shutil
import subprocess
import tempfile
import uuid

from django.conf import settings
from django.db import transaction
from redis.exceptions import RedisError
from django.http import FileResponse, Http404, HttpResponseRedirect, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import AudioMIDI
from .queue import enqueue_task, get_redis, get_task_state
from .ratelimit import check_rate_limit
from .tasks import get_audio_filename

UPLOAD_FORMATS = (
    "aac,ac3,aiff,amr,ape,asf,avi,caf,eac3,flac,flv,matroska,webm,"
    "mov,mp4,m4a,3gp,3g2,mj2,mp3,mpeg,mpegts,ogg,tta,wav,wv"
)


def _upload_has_decodable_audio(audio_file):
    # Restrict probing to local media containers and decode only the first frame.
    with tempfile.NamedTemporaryFile() as source:
        try:
            shutil.copyfileobj(audio_file, source)
        finally:
            audio_file.seek(0)
        source.flush()
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist",
                    "file", "-format_whitelist", UPLOAD_FORMATS, "-i",
                    source.name, "-map", "0:a:0", "-frames:a", "1", "-ac", "1",
                    "-ar", "8000", "-f", "s16le", "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0 and bool(result.stdout)


def health(request):
    return JsonResponse({"status": "ok"})


def readiness(request):
    checks = {}
    ready = True

    try:
        from django.db import connection

        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception:
        ready = False
        checks["database"] = "error"

    try:
        r = get_redis()
        r.ping()
        workers = [
            key.decode() if isinstance(key, bytes) else key
            for key in r.scan_iter("worker:*")
        ]
        checks["redis"] = "ok"
        checks["workers_online"] = len(workers)
    except Exception:
        ready = False
        checks["redis"] = "error"

    return JsonResponse(
        {"status": "ok" if ready else "degraded", "checks": checks},
        status=200 if ready else 503,
    )


def _api_request_is_authorized(request):
    expected = settings.TRANSCRIBE_API_KEY
    if not expected:
        return True

    provided = request.headers.get("X-API-Key", "")
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    return bool(provided) and hmac.compare_digest(provided, expected)


def _api_unauthorized_response():
    response = JsonResponse({"error": "Unauthorized"}, status=401)
    response["WWW-Authenticate"] = 'Bearer realm="transcribe"'
    return response


def _rate_limit_response(request, action, limit):
    try:
        allowed, _ = check_rate_limit(request, action, limit=limit)
    except Exception as e:
        return JsonResponse(
            {"error": "Rate limiting is unavailable", "details": str(e)},
            status=503,
        )

    if not allowed:
        response = JsonResponse(
            {"error": "Rate limit exceeded. Try again later."},
            status=429,
        )
        response["Retry-After"] = str(settings.RATE_LIMIT_WINDOW_SECONDS)
        return response

    return None


@csrf_exempt
def upload_audio(request):
    if not _api_request_is_authorized(request):
        return _api_unauthorized_response()

    if request.method == "POST" and request.FILES.get("audio", False):
        limited = _rate_limit_response(
            request,
            "upload",
            settings.RATE_LIMIT_UPLOADS,
        )
        if limited:
            return limited

        audio_file = request.FILES["audio"]

        if audio_file.size > settings.MAX_UPLOAD_BYTES:
            return JsonResponse(
                {
                    "error": (
                        f"File too large. Maximum upload size is "
                        f"{settings.MAX_UPLOAD_BYTES} bytes."
                    )
                },
                status=413,
            )

        content_type = (audio_file.content_type or "").lower()
        if content_type and not (
            content_type.startswith("audio/")
            or content_type.startswith("video/")
            or content_type == "application/octet-stream"
        ):
            return JsonResponse(
                {"error": f"Unsupported audio content type: {content_type}"},
                status=415,
            )

        try:
            readable = _upload_has_decodable_audio(audio_file)
        except OSError:
            message = "Audio validation is temporarily unavailable. Please try again."
            return JsonResponse({"error": message, "message": message}, status=503)
        if not readable:
            message = "This file does not contain readable audio. Try exporting a fresh MP3."
            return JsonResponse({"error": message, "message": message}, status=422)

        audio_filename = get_audio_filename()

        audio_midi = AudioMIDI.objects.create(
            audio_file=audio_file,
            audio_filename=audio_filename,
        )

        return JsonResponse(
            {
                "message": "File uploaded successfully!",
                "audio_filename": audio_filename,
                "audio_midi_id": audio_midi.id,
            }
        )

    return JsonResponse({"error": "Failed to upload file"}, status=400)


@csrf_exempt
def upload_from_youtube(request):
    if not _api_request_is_authorized(request):
        return _api_unauthorized_response()

    if request.method == "POST":
        limited = _rate_limit_response(
            request,
            "youtube_download",
            settings.RATE_LIMIT_YOUTUBE_DOWNLOADS,
        )
        if limited:
            return limited

        youtube_url = request.POST.get("youtube_url")
        if not youtube_url:
            return JsonResponse({"error": "No YouTube URL provided"}, status=400)

        audio_midi = None
        try:
            audio_midi = AudioMIDI.objects.create(
                youtube_url=youtube_url,
                audio_filename=f"{uuid.uuid4().hex}.mp4",
                status=AudioMIDI.Status.QUEUED,
            )

            enqueue_task(
                {
                    "type": "youtube_download",
                    "audio_midi_id": str(audio_midi.id),
                    "youtube_url": youtube_url,
                }
            )

            return JsonResponse(
                {
                    "message": "YouTube audio download queued.",
                    "audio_midi_id": audio_midi.id,
                }
            )
        except Exception as e:
            if audio_midi is not None:
                audio_midi.status = AudioMIDI.Status.FAILED
                audio_midi.error_message = f"Failed to enqueue download: {e}"
                audio_midi.save(update_fields=["status", "error_message"])
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def transcribe(request):
    if not _api_request_is_authorized(request):
        return _api_unauthorized_response()

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    limited = _rate_limit_response(
        request,
        "transcribe",
        settings.RATE_LIMIT_TRANSCRIPTIONS,
    )
    if limited:
        return limited

    try:
        audio_midi_id = request.POST["audio_midi_id"]
        with transaction.atomic():
            audio_midi = AudioMIDI.objects.select_for_update().get(id=audio_midi_id)

            if not audio_midi.audio_file:
                return JsonResponse(
                    {
                        "error": (
                            "Audio is not ready yet. Wait for the YouTube download "
                            "to finish before requesting transcription."
                        )
                    },
                    status=409,
                )

            if audio_midi.status in (
                AudioMIDI.Status.QUEUED, AudioMIDI.Status.PROCESSING, AudioMIDI.Status.COMPLETED,
            ):
                return JsonResponse({"audio_midi_id": audio_midi.id, "status": audio_midi.status})

            audio_midi.status = AudioMIDI.Status.QUEUED
            audio_midi.error_message = ""
            audio_midi.save(update_fields=["status", "error_message"])

            audio_url = audio_midi.audio_file.url
            if not audio_url.startswith(("http://", "https://")):
                audio_url = f"{settings.WEB_URL.rstrip('/')}{audio_url}"

            try:
                enqueue_task(
                    {
                        "type": "transcribe",
                        "audio_midi_id": str(audio_midi_id),
                        "audio_url": audio_url,
                    }
                )
            except Exception:
                audio_midi.status = AudioMIDI.Status.FAILED
                audio_midi.error_message = "Could not queue transcription."
                audio_midi.save(update_fields=["status", "error_message"])
                return JsonResponse({"error": audio_midi.error_message}, status=503)

            return JsonResponse(
                {
                    "message": "Transcription queued. Check status endpoint for updates.",
                    "audio_midi_id": audio_midi_id,
                }
            )

    except KeyError as e:
        return JsonResponse({"error": f"Missing key in request data: {str(e)}"}, status=400)
    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI not found"}, status=404)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JsonResponse(
            {"error": "Internal server error", "details": str(e)}, status=500
        )


def _worker_request_is_authorized(request):
    """Validate the shared worker secret on internal callbacks."""
    expected = settings.WORKER_SHARED_SECRET
    provided = request.headers.get("X-Worker-Token", "")
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided, expected)


@csrf_exempt
def result(request):
    if not _worker_request_is_authorized(request):
        return JsonResponse({"error": "Forbidden"}, status=403)

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    audio_midi_id = request.POST.get("audio_midi_id")
    task_type = request.POST.get("task_type")
    result_file = request.FILES.get("result_file")

    if not audio_midi_id or not result_file:
        return JsonResponse({"error": "Missing audio_midi_id or result_file"}, status=400)

    valid_result_types = {
        AudioMIDI.Status.COMPLETED,
        AudioMIDI.Status.YOUTUBE_AUDIO_DOWNLOADED,
    }
    if task_type not in valid_result_types:
        return JsonResponse({"error": "Invalid task_type"}, status=400)

    try:
        with transaction.atomic():
            audio_midi = AudioMIDI.objects.select_for_update().get(id=audio_midi_id)
            return _save_result(audio_midi, audio_midi_id, task_type, result_file)
    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI not found"}, status=404)

def _save_result(audio_midi, audio_midi_id, task_type, result_file):
    is_midi = task_type == AudioMIDI.Status.COMPLETED
    if (is_midi and audio_midi.midi_file) or (not is_midi and audio_midi.audio_file):
        return JsonResponse({
            "message": "Already processed", "audio_midi_id": audio_midi_id,
            "status": audio_midi.status,
        })

    if task_type == AudioMIDI.Status.YOUTUBE_AUDIO_DOWNLOADED:
        audio_midi.audio_file.save(audio_midi.audio_filename, result_file, save=False)
    else:
        midi_filename = f"{audio_midi.id}.midi"
        audio_midi.midi_file.save(midi_filename, result_file, save=False)

    audio_midi.status = task_type
    audio_midi.error_message = ""
    if (
        task_type == AudioMIDI.Status.COMPLETED
        and audio_midi.total_chunks
    ):
        # The final progress pub/sub event can be lost during a subscriber
        # restart. Completion always means every chunk is done.
        audio_midi.current_chunk = audio_midi.total_chunks
    audio_midi.save()

    return JsonResponse(
        {
            "message": "Result received",
            "audio_midi_id": audio_midi_id,
            "status": audio_midi.status,
        }
    )


def metrics(request):
    if not _api_request_is_authorized(request):
        return _api_unauthorized_response()

    from .queue import get_queue_stats

    try:
        stats = get_queue_stats()
        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def audio_status(request, audio_midi_id):
    if not _api_request_is_authorized(request):
        return _api_unauthorized_response()

    try:
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
        response_data = {
            "audio_midi_id": audio_midi.id,
            "audio_filename": audio_midi.audio_filename,
            "created_at": audio_midi.created_at,
            "updated_at": audio_midi.updated_at,
            "status": audio_midi.status,
            "has_midi": bool(audio_midi.midi_file),
            "current_chunk": audio_midi.current_chunk,
            "total_chunks": audio_midi.total_chunks,
            "error_message": audio_midi.error_message,
        }
        if audio_midi.status in (AudioMIDI.Status.QUEUED, AudioMIDI.Status.PROCESSING):
            try:
                state = get_task_state(audio_midi_id) or {}
            except RedisError:
                state = {}
            if state.get("status") == "retrying":
                response_data.update(
                    status="retrying", retry_at=state["retry_at"],
                    message="Retrying automatically", current_chunk=0, total_chunks=0,
                )
            elif state.get("status") == "failed":
                response_data.update(status="failed", error_message=state.get("error", ""))
        return JsonResponse(response_data, status=200)

    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI object not found"}, status=404)

    except Exception as e:
        return JsonResponse(
            {"error": "Internal server error", "details": str(e)}, status=500
        )


@csrf_exempt
def get_midi(request, audio_midi_id):
    if not _api_request_is_authorized(request):
        return _api_unauthorized_response()

    try:
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

        if not audio_midi.midi_file:
            raise Http404("No MIDI file found for this transcription.")

        # The frontend reads the preview with fetch, so keep it on this origin.
        return FileResponse(
            audio_midi.midi_file.open(),
            content_type="audio/midi",
            filename=os.path.basename(audio_midi.midi_file.name),
        )

    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI object not found"}, status=404)
    except Exception as e:
        return JsonResponse(
            {"error": "Internal server error", "details": str(e)}, status=500
        )


@csrf_exempt
def download_midi(request, audio_midi_id):
    if not _api_request_is_authorized(request):
        return _api_unauthorized_response()

    try:
        audio_midi = AudioMIDI.objects.get(pk=audio_midi_id)

        if not audio_midi.midi_file:
            raise Http404("No MIDI file found for this transcription.")

        midi_url = audio_midi.midi_file.url
        if midi_url.startswith(("http://", "https://")):
            return HttpResponseRedirect(midi_url)

        return FileResponse(
            audio_midi.midi_file.open(),
            as_attachment=True,
            filename=os.path.basename(audio_midi.midi_file.name),
        )

    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI object not found"}, status=404)
    except Exception as e:
        return JsonResponse(
            {"error": "Internal server error", "details": str(e)}, status=500
        )
