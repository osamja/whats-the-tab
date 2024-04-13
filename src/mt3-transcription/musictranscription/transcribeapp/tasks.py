# tasks.py
import datetime, uuid
from .models import AudioMIDI
from .ml import sayHi, generate_midi_from_audio

def convert_audio_to_midi(audio_id):
    audio_midi = AudioMIDI.objects.get(id=audio_id)
    # load the audio file
    audio_file = audio_midi.audio_file
    midi_file = generate_midi_from_audio(audio_midi.id, audio_file)
    return midi_file
    # open the audio file
    # audio_file.open(mode='rb')
    # # read the audio file
    # audio_content = audio_file.read()
    # audio_file.close()
    # # Your conversion code here, saving the MIDI file to audio_midi.midi_file
    # audio_midi.midi_file.save('output.midi', ContentFile(midi_content))
    # audio_midi.save()

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




