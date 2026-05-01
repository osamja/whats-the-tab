import os
import uuid

from django.conf import settings
from django.core.files import File as DjangoFile
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import AudioMIDI
from .queue import enqueue_task
from .tasks import get_audio_filename


def health(request):
    return JsonResponse({"status": "ok"})


@csrf_exempt
def upload_audio(request):
    if request.method == "POST" and request.FILES.get("audio", False):
        audio_file = request.FILES["audio"]
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
        youtube_url = request.POST.get("youtube_url")
        if not youtube_url:
            return JsonResponse({"error": "No YouTube URL provided"}, status=400)

        try:
            audio_midi = AudioMIDI.objects.create(
                youtube_url=youtube_url,
                audio_filename=f"{uuid.uuid4().hex}.mp4",
                status="queued",
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
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def transcribe(request):
    try:
        if request.method == "POST":
            audio_midi_id = request.POST["audio_midi_id"]
            audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

            audio_midi.status = "queued"
            audio_midi.save()

            web_url = settings.WEB_URL.rstrip("/")
            audio_url = f"{web_url}/media/{audio_midi.audio_file.name}"

            enqueue_task(
                {
                    "type": "transcribe",
                    "audio_midi_id": str(audio_midi_id),
                    "audio_url": audio_url,
                }
            )

            return JsonResponse(
                {
                    "message": "Transcription queued. Check status endpoint for updates.",
                    "audio_midi_id": audio_midi_id,
                }
            )

    except KeyError as e:
        return JsonResponse({"error": f"Missing key in request data: {str(e)}"}, status=400)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JsonResponse(
            {"error": "Internal server error", "details": str(e)}, status=500
        )


@csrf_exempt
def result(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    audio_midi_id = request.POST.get("audio_midi_id")
    task_type = request.POST.get("task_type")
    result_file = request.FILES.get("result_file")

    if not audio_midi_id or not result_file:
        return JsonResponse({"error": "Missing audio_midi_id or result_file"}, status=400)

    try:
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
    except AudioMIDI.DoesNotExist:
        return JsonResponse({"error": "AudioMIDI not found"}, status=404)

    already_completed = task_type in ("completed", "youtube_audio_downloaded")
    if already_completed and audio_midi.status == task_type:
        if task_type == "completed" and audio_midi.midi_file:
            return JsonResponse({"message": "Already processed", "audio_midi_id": audio_midi_id, "status": audio_midi.status})
        if task_type == "youtube_audio_downloaded" and audio_midi.audio_file:
            return JsonResponse({"message": "Already processed", "audio_midi_id": audio_midi_id, "status": audio_midi.status})

    if task_type == "youtube_audio_downloaded":
        audio_midi.audio_file.save(audio_midi.audio_filename, result_file)
    else:
        midi_filename = f"{audio_midi.id}.midi"
        audio_midi.midi_file.save(midi_filename, result_file)

    audio_midi.status = task_type
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
