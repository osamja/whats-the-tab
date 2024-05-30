from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import uuid
from .tasks import get_audio_filename, getAudioDirectory, generate_midi_from_audio, download_youtube_audio_and_save
from .models import AudioMIDI, MIDIChunk
import pdb
import os
import io
from django.core.files import File
import base64


import dramatiq
from django.core.files.base import ContentFile

import subprocess

IS_ASYNC = False

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
#   import pdb; pdb.set_trace()
  try:
    if request.method == 'POST':
      audio_midi_id = request.POST['audio_midi_id']
      
      num_transcription_segments = request.POST.get('num_transcription_segments', 1)
      audio_chunk_length = request.POST.get('audio_chunk_length', 10)
      audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

      # set audio midi object fields
      audio_midi.num_transcription_segments = int(num_transcription_segments)
      audio_midi.audio_chunk_length = int(audio_chunk_length)
      audio_midi.status = 'processing'
      audio_midi.save()

      if IS_ASYNC:
        generate_midi_from_audio.send(audio_midi_id)
      else:
        generate_midi_from_audio(audio_midi_id)

      return JsonResponse({
        'message':'Created MIDI generation task. Check status endpoint for updates.',
        'audio_midi_id ': audio_midi_id,
      })
  except KeyError as e:
        # Handle the case where 'audio_id' is not provided
        error_message = f"Missing key in request data: {str(e)}"
        # audio_midi.status = 'failed: ' + str(e)
        # audio_midi.save()
        return JsonResponse({'error': error_message}, status=400) 
  except Exception as e:
        # audio_midi.status = 'failed: ' + str(e)
        # audio_midi.save()
        # General exception handler for any other unanticipated exceptions
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt  # @todo remove for prod
def audio_status(request, audio_midi_id):
    try:
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
        response_data = {
            'audio_midi_id': audio_midi.id,
            'audio_filename': audio_midi.audio_filename,
            # 'audio_file': audio_midi.audio_file.name,
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

@csrf_exempt
def list_midi_chunks(request, audio_midi_id):
    try:
        # Fetch the AudioMIDI instance by ID
        audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

        # Fetch all MIDI chunks related to this AudioMIDI instance
        midi_chunks = MIDIChunk.objects.filter(audio_midi=audio_midi).order_by('segment_index')
        
        # Prepare data to return
        chunks_info = [{
            'midi_file_name': chunk.midi_file.name if chunk.midi_file else None,
            'segment_index': chunk.segment_index
        } for chunk in midi_chunks]

        return JsonResponse({
            'audio_midi_id': audio_midi_id,
            'audio_filename': audio_midi.audio_filename,
            'midi_chunks': chunks_info
        }, status=200)

    except AudioMIDI.DoesNotExist:
        return JsonResponse({'error': 'AudioMIDI object not found'}, status=404)
    except Exception as e:
        # Handle unexpected errors
        return JsonResponse({'error': 'Internal server error', 'details': str(e)}, status=500)

@csrf_exempt
def download_midi_chunk(request, audio_midi_id, segment_index):
    try:
        # Fetch the AudioMIDI instance by ID
        audio_midi = AudioMIDI.objects.get(pk=audio_midi_id)

        # Fetch the MIDI chunk based on the segment index
        midi_chunk = MIDIChunk.objects.get(audio_midi=audio_midi, segment_index=segment_index)

        # Assuming midi_file is the field name in the MIDIChunk model where the file path is stored
        if not midi_chunk.midi_file:
            raise Http404("No MIDI file found for the provided segment index.")

        # Return the MIDI file as a response
        return FileResponse(midi_chunk.midi_file.open(), as_attachment=True, filename=midi_chunk.midi_file.name)

    except AudioMIDI.DoesNotExist:
        return JsonResponse({'error': 'AudioMIDI object not found'}, status=404)
    except MIDIChunk.DoesNotExist:
        return JsonResponse({'error': 'MIDI chunk not found'}, status=404)
    except Exception as e:
        # General exception handler for any other unanticipated exceptions
        return JsonResponse({'error': 'Internal server error', 'details': str(e)}, status=500)
