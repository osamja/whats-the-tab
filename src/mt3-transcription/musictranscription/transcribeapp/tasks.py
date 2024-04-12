# tasks.py
import datetime, uuid

def convert_audio_to_midi(audio_filename):
    
    from .models import AudioMIDI
    audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
    # Your conversion code here, saving the MIDI file to audio_midi.midi_file
    audio_midi.midi_file.save('output.midi', ContentFile(midi_content))
    audio_midi.save()

def get_audio_filename():
   fileHash = uuid.uuid4()
   date = getDate()
   audio_filename = date + '-' + fileHash.hex + '.wav'
   return audio_filename

def getAudioDirectory():
  return 'content/audio/'
    
def getDate():
  date = str(datetime.datetime.now())
  date = date.replace(" ", "-")
  date = date.replace(":", "-")
  date = date.replace(".", "-")
  return date
