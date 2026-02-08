# tasks.py
import datetime, uuid
from transcribeapp.models import AudioMIDI
import io
from django.core.files import File
from django.core.files.base import ContentFile
import dramatiq
import note_seq
from django.utils.timezone import now
from pytube import YouTube
import tempfile
import os

from dotenv import load_dotenv

load_dotenv()

# Choose backend: 'pytorch' or 'jax'
USE_PYTORCH = os.getenv('USE_PYTORCH', 'True').lower() in ('true', '1', 't')

MODEL = "mt3" #@param["ismir2021", "mt3"]

if USE_PYTORCH:
    from .ml_pytorch import PyTorchInferenceModel, load_audio, SAMPLE_RATE
    pytorch_checkpoint_path = 'pytorch_mt3/mt3_pytorch_checkpoint.pt'
    print(f"Using PyTorch backend with checkpoint: {pytorch_checkpoint_path}")
else:
    from .ml import InferenceModel
    import librosa
    SAMPLE_RATE = 16000
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
        audio = load_audio(audio_midi.audio_file.path, sample_rate=SAMPLE_RATE, mono=True).cpu().numpy()
    else:
        inference_model = InferenceModel(checkpoint_path, MODEL)
        audio, _ = librosa.load(audio_midi.audio_file.path, sr=SAMPLE_RATE, mono=True)

    # Transcribe audio to note sequence
    est_ns = inference_model(audio)

    # Save MIDI to a temporary file, then attach to model
    midi_filename = f"{audio_midi.id}.midi"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.midi') as tmp:
        note_seq.sequence_proto_to_midi_file(est_ns, tmp.name)
        with open(tmp.name, 'rb') as midi_file:
            audio_midi.midi_file.save(midi_filename, File(midi_file))
    os.remove(tmp.name)

    audio_midi.status = 'completed'
    audio_midi.save()

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
