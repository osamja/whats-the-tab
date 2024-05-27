from django.db import models

class AudioMIDI(models.Model):
    audio_filename = models.CharField(max_length=255)
    audio_file = models.FileField(upload_to='audios/')
    youtube_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, default='pending')
    current_segment = models.IntegerField(default=0)
    num_transcription_segments = models.IntegerField(default=1)
    audio_chunk_length = models.IntegerField(default=10)

class AudioChunk(models.Model):
    audio_midi = models.ForeignKey(AudioMIDI, on_delete=models.CASCADE, related_name='audio_chunks')
    chunk_file = models.FileField(upload_to='audio_chunks/')
    segment_index = models.IntegerField(default=0)  # Order of the chunk

    def __str__(self):
        return f"Chunk {self.segment_index} for {self.audio_midi.audio_filename}"

class MIDIChunk(models.Model):
    audio_midi = models.ForeignKey(AudioMIDI, on_delete=models.CASCADE, related_name='midi_chunks')
    midi_file = models.FileField(upload_to='midi_chunks/')
    segment_index = models.IntegerField(default=0)  # Order of the MIDI segment

    def __str__(self):
        return f"MIDI Chunk {self.segment_index} for {self.audio_midi.audio_filename}"
