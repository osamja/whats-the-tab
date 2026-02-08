from django.db import models

class AudioMIDI(models.Model):
    audio_filename = models.CharField(max_length=255)
    audio_file = models.FileField(upload_to='audios/')
    midi_file = models.FileField(upload_to='midi/', blank=True, null=True)
    youtube_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, default='pending')
