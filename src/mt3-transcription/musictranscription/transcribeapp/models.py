from django.db import models

class AudioMIDI(models.Model):
    audio_filename = models.CharField(max_length=255)
    audio_file = models.FileField(upload_to='audios/')
    midi_file = models.FileField(upload_to='midis/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, default='pending')  # Status of the transcription
    current_segment = models.IntegerField(default=0)  # Current segment being transcribed
    num_transcription_segments = models.IntegerField(default=10)  # Number

