from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import uuid
from .tasks import get_audio_filename, getAudioDirectory
from .models import AudioMIDI
from .ml import sayHi, generate_midi_from_audio
import pdb
import os
import io
from django.core.files import File

@csrf_exempt  # @todo remove for prod
def upload_audio(request):
  if request.method == 'POST' and request.FILES['audio']:
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
  if request.method == 'POST':
    audio_id = request.POST['audio_id']
    num_transcription_segments = request.POST['num_transcription_segments']
    audio_midi = AudioMIDI.objects.get(id=audio_id)
    # load the audio file
    audio_file = audio_midi.audio_file
    midi_file, midi_filename = generate_midi_from_audio(audio_midi.id, audio_file, num_transcription_segments)

    # Assuming `output_midi` is your MidiFile object
    # Save the MidiFile data to a BytesIO object
    midi_buffer = io.BytesIO()
    midi_file.save(file=midi_buffer)

    # It's important to seek back to the beginning of the BytesIO object after writing to it
    midi_buffer.seek(0)

    # Create a Django File object wrapping the BytesIO buffer
    wrapped_file = File(midi_buffer, name=midi_filename)
    audio_midi.midi_file = wrapped_file
    
    audio_midi.save()

    return JsonResponse({
      'message':'created midi successfully',
      'audio id ': audio_id,
      'audio filename': audio_midi.audio_filename,
      'midi filename': audio_midi.midi_file.name
    })

  return JsonResponse({'message': 'Transcribe view'})

@csrf_exempt  # @todo remove for prod
def download_midi(request, audio_id):
    try:
        audio_midi = AudioMIDI.objects.get(pk=audio_id)
        # Assuming midi_file is the field name in your model where the file path is stored
        response = FileResponse(audio_midi.midi_file.open(), as_attachment=True, filename=audio_midi.midi_file.name)
        return response
    except AudioMIDI.DoesNotExist:
        raise Http404("No MIDI file found for the provided ID.")

