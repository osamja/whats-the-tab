import hmac
import os
import uuid
from urllib.parse import quote

from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import AudioMIDI
from .queue import enqueue_task, get_redis
from .ratelimit import check_rate_limit
from .tasks import get_audio_filename


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
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

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

        audio_midi.status = AudioMIDI.Status.QUEUED
        audio_midi.error_message = ""
        audio_midi.save(update_fields=["status", "error_message"])

        web_url = settings.WEB_URL.rstrip("/")
        audio_url = (
            f"{web_url}/media/{quote(audio_midi.audio_file.name, safe='/')}"
        )

        try:
            enqueue_task(
                {
                    "type": "transcribe",
                    "audio_midi_id": str(audio_midi_id),
                    "audio_url": audio_url,
                }
            )
        except Exception as e:
            audio_midi.status = AudioMIDI.Status.FAILED
            audio_midi.error_message = f"Failed to enqueue task: {e}"
            audio_midi.save(update_fields=["status", "error_message"])
            raise

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
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI not found"}, status=404)

    if audio_midi.status == task_type:
        if task_type == AudioMIDI.Status.COMPLETED and audio_midi.midi_file:
            return JsonResponse({"message": "Already processed", "audio_midi_id": audio_midi_id, "status": audio_midi.status})
        if task_type == AudioMIDI.Status.YOUTUBE_AUDIO_DOWNLOADED and audio_midi.audio_file:
            return JsonResponse({"message": "Already processed", "audio_midi_id": audio_midi_id, "status": audio_midi.status})

    if task_type == AudioMIDI.Status.YOUTUBE_AUDIO_DOWNLOADED:
        audio_midi.audio_file.save(audio_midi.audio_filename, result_file)
    else:
        midi_filename = f"{audio_midi.id}.midi"
        audio_midi.midi_file.save(midi_filename, result_file)

    audio_midi.status = task_type
    audio_midi.error_message = ""
    audio_midi.save()

    return JsonResponse(
        {
            "message": "Result received",
            "audio_midi_id": audio_midi_id,
            "status": audio_midi.status,
        }
    )


def metrics(request):
    from .queue import get_queue_stats

    try:
        stats = get_queue_stats()
        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def audio_status(request, audio_midi_id):
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
        return JsonResponse(response_data, status=200)

    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI object not found"}, status=404)

    except Exception as e:
        return JsonResponse(
            {"error": "Internal server error", "details": str(e)}, status=500
        )


@csrf_exempt
def get_midi(request, audio_midi_id):
    try:
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

        if not audio_midi.midi_file:
            raise Http404("No MIDI file found for this transcription.")

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
    try:
        audio_midi = AudioMIDI.objects.get(pk=audio_midi_id)

        if not audio_midi.midi_file:
            raise Http404("No MIDI file found for this transcription.")

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
