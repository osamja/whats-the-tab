from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import uuid
from .tasks import convert_audio_to_midi, get_audio_filename, getAudioDirectory
import pdb
import os

@csrf_exempt  # @todo remove for prod
def upload_audio(request):
  if request.method == 'POST' and request.FILES['audio']:
    audio_file = request.FILES['audio']
    audio_filename = get_audio_filename()
    audio_path = getAudioDirectory() + audio_filename

    FileSystemStorage().save(audio_path, audio_file)

    # audio_midi = AudioMIDI.objects.create(audio_file=audio_file)
    # convert_audio_to_midi.delay(audio_midi.id)  # Asynchronously process the audio
    return JsonResponse({
      'message': 'File uploaded successfully!',
      'audio_filename': audio_filename
    })
  return JsonResponse({'error': 'Failed to upload file'}, status=400)

@csrf_exempt  # @todo remove for prod
def transcribe(request):
  if request.method == 'POST':
    audio_filename = request.POST['audio_filename']
    midi = convert_audio_to_midi(audio_filename)
    return JsonResponse({'message': audio_filename})
  return JsonResponse({'message': 'Transcribe view'})
  # return render(request, 'transcribe.html')
