from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
import uuid
from .tasks import get_audio_filename, generate_midi_from_audio, download_youtube_audio_and_save
from .models import AudioMIDI
import os

from dotenv import load_dotenv

load_dotenv()

IS_ASYNC = os.getenv('IS_ASYNC', 'False').lower() in ('true', '1', 't')

@csrf_exempt  # @todo remove for prod
def upload_audio(request):
  if request.method == 'POST' and request.FILES.get('audio', False):
    audio_file = request.FILES['audio']
    audio_filename = get_audio_filename()

    audio_midi = AudioMIDI.objects.create(
      audio_file=audio_file,
      audio_filename=audio_filename
    )

    id = audio_midi.id

    return JsonResponse({
      'message': 'File uploaded successfully!',
      'audio_filename': audio_filename,
      'audio_midi_id': id,
    })

  return JsonResponse({'error': 'Failed to upload file'}, status=400)

@csrf_exempt
def upload_from_youtube(request):
    if request.method == 'POST':
        youtube_url = request.POST.get('youtube_url')
        if not youtube_url:
            return JsonResponse({'error': 'No YouTube URL provided'}, status=400)
        try:
            audio_midi = AudioMIDI.objects.create(
                youtube_url=youtube_url,
                audio_filename=f"{uuid.uuid4().hex}.mp4",  # Temporary filename placeholder
                status='initiated'
            )

            if IS_ASYNC:
                download_youtube_audio_and_save.send(audio_midi.id)
            else:
                download_youtube_audio_and_save(audio_midi.id)

            return JsonResponse({
                'message': 'YouTube audio download initiated.',
                'audio_midi_id': audio_midi.id
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt  # @todo remove for prod
def transcribe(request):
  try:
    if request.method == 'POST':
      audio_midi_id = request.POST['audio_midi_id']
      audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

      audio_midi.status = 'processing'
      audio_midi.save()

      if IS_ASYNC:
        generate_midi_from_audio.send(audio_midi_id)
        return JsonResponse({
            'message':'Queued MIDI generation task. Check status endpoint for updates.',
            'audio_midi_id': audio_midi_id,
        })
      else:
        generate_midi_from_audio(audio_midi_id)
        return JsonResponse({
            'message':'Completed MIDI generation.',
            'audio_midi_id': audio_midi_id,
        })

  except KeyError as e:
        error_message = f"Missing key in request data: {str(e)}"
        return JsonResponse({'error': error_message}, status=400)
  except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': 'Internal server error', 'details': str(e)}, status=500)

@csrf_exempt  # @todo remove for prod
def audio_status(request, audio_midi_id):
    try:
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
        response_data = {
            'audio_midi_id': audio_midi.id,
            'audio_filename': audio_midi.audio_filename,
            'created_at': audio_midi.created_at,
            'updated_at': audio_midi.updated_at,
            'status': audio_midi.status,
            'has_midi': bool(audio_midi.midi_file),
        }
        return JsonResponse(response_data, status=200)

    except AudioMIDI.DoesNotExist:
        return JsonResponse({'error': 'AudioMIDI object not found'}, status=404)

    except Exception as e:
        return JsonResponse({'error': 'Internal server error', 'details': str(e)}, status=500)

@csrf_exempt
def get_midi(request, audio_midi_id):
    try:
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

        midi_info = {
            'audio_midi_id': audio_midi_id,
            'audio_filename': audio_midi.audio_filename,
            'midi_file_name': audio_midi.midi_file.name if audio_midi.midi_file else None,
        }

        return JsonResponse(midi_info, status=200)

    except AudioMIDI.DoesNotExist:
        return JsonResponse({'error': 'AudioMIDI object not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': 'Internal server error', 'details': str(e)}, status=500)

@csrf_exempt
def download_midi(request, audio_midi_id):
    try:
        audio_midi = AudioMIDI.objects.get(pk=audio_midi_id)

        if not audio_midi.midi_file:
            raise Http404("No MIDI file found for this transcription.")

        return FileResponse(audio_midi.midi_file.open(), as_attachment=True, filename=os.path.basename(audio_midi.midi_file.name))

    except AudioMIDI.DoesNotExist:
        return JsonResponse({'error': 'AudioMIDI object not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': 'Internal server error', 'details': str(e)}, status=500)
