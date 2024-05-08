from django.db import models

class AudioMIDI(models.Model):
    audio_filename = models.CharField(max_length=255)
    audio_file = models.FileField(upload_to='audios/')
    midi_file = models.FileField(upload_to='midis/', blank=True, null=True)
    youtube_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, default='pending')  # Status of the transcription
    current_segment = models.IntegerField(default=0)  # Current segment being transcribed
    num_transcription_segments = models.IntegerField(default=10)  # Number
    is_midi2wav = models.BooleanField(default=False)  # Whether the audio is a MIDI wav file
    midi_wav_file = models.FileField(upload_to='midi_2_wav/', blank=True, null=True)  # MIDI as wav file

class AudioChunk(models.Model):
    audio_midi = models.ForeignKey(AudioMIDI, on_delete=models.CASCADE, related_name='audio_chunks')
    chunk_file = models.FileField(upload_to='audios/')
    segment_index = models.IntegerField(default=0)  # Order of the chunk

    def __str__(self):
        return f"Chunk {self.segment_index} for {self.audio_midi.audio_filename}"
