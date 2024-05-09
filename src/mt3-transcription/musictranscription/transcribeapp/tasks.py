# tasks.py
import datetime, uuid
from transcribeapp.models import AudioMIDI
import io
from django.core.files import File
from django.core.files.base import ContentFile
import dramatiq
import fluidsynth
import mido
from django.utils.timezone import now
from pytube import YouTube
import tempfile
import os


from .ml import split_audio_segments, transcribe_and_download, copy_acoustic_guitar_events, plot_note_on_times, InferenceModel, stitch_midi_files, midi_files_to_wav, combine_wavs

MODEL = "mt3" #@param["ismir2021", "mt3"]
mt3_path = 'checkpoints'

checkpoint_path = f'{mt3_path}/{MODEL}/'

print(checkpoint_path)

@dramatiq.actor(max_retries=3, min_backoff=1000, max_backoff=10000)
def download_youtube_audio_and_save(audio_midi_id):
    audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
    yt = YouTube(audio_midi.youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        audio_stream.download(filename=tmp_file.name)
        
        # Move temporary file to BytesIO for further processing or storage
        with open(tmp_file.name, 'rb') as file_data:
            audio_midi.audio_file.save(audio_midi.audio_filename, ContentFile(file_data.read()))
        
        audio_midi.status = 'youtube_audio_downloaded'
        audio_midi.save()

    # Clean up the temporary file
    os.remove(tmp_file.name)

@dramatiq.actor(max_retries=3, min_backoff=1000, max_backoff=10000)
def generate_midi_from_audio(audio_midi_id):
  audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
  num_transcription_segments = audio_midi.num_transcription_segments
  is_midi2wav = audio_midi.is_midi2wav

  inference_model = InferenceModel(checkpoint_path, MODEL)

  # mp3 is split into N segments of audio chunk length.
  # To transcribe entire mp3, num_transcription_segments = len(audio) / audio_chunk_length
  # To transcribe the first 2 seconds of an mp3, set NUM_TRANSCRIPTION_SEGMENTS to 1 assuming length is 2 seconds
  NUM_TRANSCRIPTION_SEGMENTS = int(num_transcription_segments)
  AUDIO_CHUNK_LENGTH = 5000
  # check if audio is an mp3 or wav file

  split_audio_segments(audio_midi, AUDIO_CHUNK_LENGTH, NUM_TRANSCRIPTION_SEGMENTS)

  split_filenames = getSplitFilenames(audio_midi)
  
  transcribe_and_download(audio_midi, split_filenames, inference_model)

  midi_files = getSplitMIDIFiles(audio_midi)

  if is_midi2wav:
    wav_files = midi_files_to_wav(midi_files, 'output.wav')
    # Combine all WAV files into one
    combined_output = combine_wavs(wav_files)
    unique_filename = f"combined_output_{now().strftime('%Y%m%d%H%M%S')}.wav"
    # Save combined WAV file using Django's FileField
    audio_midi.midi_wav_file.save(unique_filename, ContentFile(combined_output))
    print("Conversion and combination complete. Output saved as 'combined_output.wav'.")

  # Copy acoustic guitar events to a new MIDI file
  output_file = 'acoustic_guitar_only.midi'
  acoustic_guitar_midi = copy_acoustic_guitar_events(midi_files, output_file)
  midi_file, midi_filename = acoustic_guitar_midi, output_file

  print(f"Acoustic guitar events copied to '{output_file}'")

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

def getSplitMIDIFiles(audio_midi):
  split_midi_files = []
  for midi_chunk in audio_midi.midi_chunks.all():
    split_midi_files.append(midi_chunk.midi_file.path)
  return split_midi_files

def getSplitFilenames(audio_midi):
  split_filenames = []
  for audio_chunk in audio_midi.audio_chunks.all():
    split_filenames.append(audio_chunk.chunk_file.path)
  return split_filenames

def get_audio_filename(is_mp4=False):
  fileHash = uuid.uuid4()
  date = getDate()
  if is_mp4:
    audio_filename = date + '-' + fileHash.hex + '.mp4'
  else:
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




