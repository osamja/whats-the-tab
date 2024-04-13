from django.db import models

class AudioMIDI(models.Model):
    audio_filename = models.CharField(max_length=255)
    audio_file = models.FileField(upload_to='audios/')
    midi_file = models.FileField(upload_to='midis/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
