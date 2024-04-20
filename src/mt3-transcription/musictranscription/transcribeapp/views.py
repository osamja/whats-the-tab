from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import uuid
from .tasks import get_audio_filename, getAudioDirectory, generate_midi_from_audio
from .models import AudioMIDI
import pdb
import os
import io
from django.core.files import File

import dramatiq

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

    # audio_midi = AudioMIDI.objects.create(audio_file=audio_file)
    return JsonResponse({
      'message': 'File uploaded successfully!',
      'audio_filename': audio_filename,
      'id': id,
    })
  
  return JsonResponse({'error': 'Failed to upload file'}, status=400)

@csrf_exempt  # @todo remove for prod
def transcribe(request):
  try:   

    if request.method == 'POST':
      audio_midi_id = request.POST['audio_id']
      
      num_transcription_segments = request.POST.get('num_transcription_segments', 10)
      audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

      # set number of transcription segments
      audio_midi.num_transcription_segments = num_transcription_segments
      audio_midi.status = 'processing'
      audio_midi.save()

      # import pdb; pdb.set_trace()
      generate_midi_from_audio.send(audio_midi_id)

      return JsonResponse({
        'message':'Created MIDI generation task. Check status endpoint for updates.',
        'audio_midi_id ': audio_midi_id,
      })
  except KeyError as e:
        # Handle the case where 'audio_id' is not provided
        error_message = f"Missing key in request data: {str(e)}"
        audio_midi.status = 'failed: ' + str(e)
        audio_midi.save()
        return JsonResponse({'error': error_message}, status=400) 
  except Exception as e:
        audio_midi.status = 'failed: ' + str(e)
        audio_midi.save()
        # General exception handler for any other unanticipated exceptions
        return JsonResponse({'error': 'Internal server error'}, status=500)

@csrf_exempt  # @todo remove for prod
def download_midi(request, audio_id):
    # import pdb; pdb.set_trace()
    try:
        audio_midi = AudioMIDI.objects.get(pk=audio_id)
        # Assuming midi_file is the field name in your model where the file path is stored
        response = FileResponse(audio_midi.midi_file.open(), as_attachment=True, filename=audio_midi.midi_file.name)
        return response
    except AudioMIDI.DoesNotExist:
        raise Http404("No MIDI file found for the provided ID.")
    except Exception as e:
        # General exception handler for any other unanticipated exceptions
        return JsonResponse({'error': 'Internal server error'}, status=500)

@csrf_exempt  # @todo remove for prod
def audio_status(request, audio_id):
    try:
        audio_midi = AudioMIDI.objects.get(id=audio_id)
        response_data = {
            'audio_id': audio_midi.id,
            'audio_filename': audio_midi.audio_filename,
            'midi_filename': audio_midi.midi_file.name if audio_midi.midi_file else None,
            'created_at': audio_midi.created_at,
            'updated_at': audio_midi.updated_at,
            'status': audio_midi.status,
            'current_segment': audio_midi.current_segment,
            'num_transcription_segments': audio_midi.num_transcription_segments,
        }
        return JsonResponse(response_data, status=200)

    except AudioMIDI.DoesNotExist:
        return JsonResponse({'error': 'AudioMIDI object not found'}, status=404)

    except Exception as e:
        return JsonResponse({'error': 'Internal server error', 'details': str(e)}, status=500)
