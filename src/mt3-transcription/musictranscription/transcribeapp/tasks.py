# tasks.py

def convert_audio_to_midi(audio_midi_id):
    from .models import AudioMIDI
    audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
    # Your conversion code here, saving the MIDI file to audio_midi.midi_file
    audio_midi.midi_file.save('output.midi', ContentFile(midi_content))
    audio_midi.save()
