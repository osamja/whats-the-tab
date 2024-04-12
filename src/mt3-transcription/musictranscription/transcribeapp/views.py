from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
# from .tasks import convert_audio_to_midi  # Assume you have an asynchronous task

def upload_audio(request):
    if request.method == 'POST' and request.FILES['audio']:
        audio_file = request.FILES['audio']
        # audio_midi = AudioMIDI.objects.create(audio_file=audio_file)
        # convert_audio_to_midi.delay(audio_midi.id)  # Asynchronously process the audio
        return JsonResponse({'message': 'File uploaded successfully!'})
    return JsonResponse({'error': 'Failed to upload file'}, status=400)
