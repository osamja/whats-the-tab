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

# Import both JAX and PyTorch implementations
from .ml import split_audio_segments, InferenceModel
from dotenv import load_dotenv

load_dotenv()

# Choose backend: 'pytorch' or 'jax'
USE_PYTORCH = os.getenv('USE_PYTORCH', 'True').lower() in ('true', '1', 't')

MODEL = "mt3" #@param["ismir2021", "mt3"]

if USE_PYTORCH:
    from .ml_pytorch import PyTorchInferenceModel, transcribe_and_download
    pytorch_checkpoint_path = 'pytorch_mt3/mt3_pytorch_checkpoint.pt'
    print(f"Using PyTorch backend with checkpoint: {pytorch_checkpoint_path}")
else:
    from .ml import transcribe_and_download
    mt3_path = 'checkpoints'
    checkpoint_path = f'{mt3_path}/{MODEL}/'
    print(f"Using JAX backend with checkpoint: {checkpoint_path}")

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

  # Initialize inference model based on backend
  if USE_PYTORCH:
    inference_model = PyTorchInferenceModel(pytorch_checkpoint_path, MODEL)
  else:
    inference_model = InferenceModel(checkpoint_path, MODEL)

  NUM_TRANSCRIPTION_SEGMENTS = int(audio_midi.num_transcription_segments)
  AUDIO_CHUNK_LENGTH = int(audio_midi.audio_chunk_length) * 1000 # in milliseconds

  split_audio_segments(audio_midi, AUDIO_CHUNK_LENGTH, NUM_TRANSCRIPTION_SEGMENTS)

  split_filenames = getSplitFilenames(audio_midi)

  transcribe_and_download(audio_midi, split_filenames, inference_model)

  audio_midi.status = 'completed'
  audio_midi.save()

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




