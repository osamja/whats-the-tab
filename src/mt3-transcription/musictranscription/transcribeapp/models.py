from django.db import models

class AudioMIDI(models.Model):
    audio_file = models.FileField(upload_to='audios/')
    midi_file = models.FileField(upload_to='midis/', blank=True, null=True)
