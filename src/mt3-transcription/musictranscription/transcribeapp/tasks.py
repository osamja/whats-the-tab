# tasks.py
import datetime, uuid
from transcribeapp.models import AudioMIDI
from django.core.files import File
from django.core.files.base import ContentFile
import dramatiq
import note_seq
from pytube import YouTube
import tempfile
import os

from .ml_pytorch import PyTorchInferenceModel, load_audio, SAMPLE_RATE

CHECKPOINT_PATH = 'pytorch_mt3/mt3_pytorch_checkpoint.pt'

@dramatiq.actor(max_retries=3, min_backoff=1000, max_backoff=10000)
def download_youtube_audio_and_save(audio_midi_id):
    audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
    yt = YouTube(audio_midi.youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        audio_stream.download(filename=tmp_file.name)

        with open(tmp_file.name, 'rb') as file_data:
            audio_midi.audio_file.save(audio_midi.audio_filename, ContentFile(file_data.read()))

        audio_midi.status = 'youtube_audio_downloaded'
        audio_midi.save()

    os.remove(tmp_file.name)

@dramatiq.actor(max_retries=3, min_backoff=1000, max_backoff=10000)
def generate_midi_from_audio(audio_midi_id):
    audio_midi = AudioMIDI.objects.get(id=audio_midi_id)

    inference_model = PyTorchInferenceModel(CHECKPOINT_PATH, "mt3")
    audio = load_audio(audio_midi.audio_file.path, sample_rate=SAMPLE_RATE, mono=True).cpu().numpy()

    est_ns = inference_model(audio)

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
