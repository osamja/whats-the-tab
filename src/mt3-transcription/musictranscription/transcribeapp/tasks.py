# tasks.py
import datetime, uuid
from transcribeapp.models import AudioMIDI
import io
from django.core.files import File
import dramatiq

from .ml import split_mp3, transcribe_and_download, copy_acoustic_guitar_events, plot_note_on_times, InferenceModel

MODEL = "mt3" #@param["ismir2021", "mt3"]
mt3_path = 'checkpoints'

checkpoint_path = f'{mt3_path}/{MODEL}/'

print(checkpoint_path)

@dramatiq.actor
def generate_midi_from_audio(audio_midi_id):
  audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
  audio = audio_midi.audio_file
  num_transcription_segments = audio_midi.num_transcription_segments

  inference_model = InferenceModel(checkpoint_path, MODEL)

  # mp3 is split into N segments of audio chunk length.
  # To transcribe entire mp3, num_transcription_segments = len(audio) / audio_chunk_length
  # To transcribe the first 2 seconds of an mp3, set NUM_TRANSCRIPTION_SEGMENTS to 1 assuming length is 2 seconds
  NUM_TRANSCRIPTION_SEGMENTS = int(num_transcription_segments)
  AUDIO_CHUNK_LENGTH = 2000
  split_audio, split_audio_filenames = split_mp3(audio, AUDIO_CHUNK_LENGTH, NUM_TRANSCRIPTION_SEGMENTS)

  midi_files = transcribe_and_download(audio_midi, split_audio, split_audio_filenames, inference_model)

  # Replace with the path to your MIDI file
  midi_file_path = midi_files[0]
  plot_note_on_times(midi_file_path)

  # Copy acoustic guitar events to a new MIDI file
  output_file = 'acoustic_guitar_only.midi'
  acoustic_guitar_midi = copy_acoustic_guitar_events(midi_files, output_file)

  print(f"Acoustic guitar events copied to '{output_file}'")

  midi_file, midi_filename = acoustic_guitar_midi, output_file
  # Assuming `output_midi` is your MidiFile object
  # Save the MidiFile data to a BytesIO object
  midi_buffer = io.BytesIO()
  midi_file.save(file=midi_buffer)

  # It's important to seek back to the beginning of the BytesIO object after writing to it
  midi_buffer.seek(0)

  # Create a Django File object wrapping the BytesIO buffer
  wrapped_file = File(midi_buffer, name=midi_filename)
  audio_midi.midi_file = wrapped_file
  
  audio_midi.save()

  audio_midi.status = 'completed'
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




