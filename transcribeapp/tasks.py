import datetime
import os
import threading
import uuid

CHECKPOINT_PATH = "pytorch_mt3/mt3_pytorch_checkpoint.pt"

_inference_model = None
_inference_model_lock = threading.Lock()


def _get_inference_model():
    """Load the MT3 model once per worker process.

    The checkpoint is large and loading it on every task wastes several seconds
    of GPU billing time. The worker processes one task at a time, but a lock
    keeps this safe if that ever changes.
    """
    global _inference_model
    if _inference_model is None:
        with _inference_model_lock:
            if _inference_model is None:
                from .ml_pytorch import PyTorchInferenceModel

                _inference_model = PyTorchInferenceModel(CHECKPOINT_PATH, "mt3")
    return _inference_model


def get_audio_filename(is_mp4=False):
    ext = ".mp4" if is_mp4 else ".wav"
    return f"{datetime.datetime.now().isoformat()}-{uuid.uuid4().hex}{ext}"


def download_youtube_audio(youtube_url, output_dir):
    import yt_dlp
    from django.conf import settings

    max_duration = settings.MAX_AUDIO_DURATION_SECONDS

    def reject_too_long(info, *, incomplete):
        duration = info.get("duration")
        if duration and duration > max_duration:
            return f"Video exceeds the {max_duration}s duration limit"
        return None

    output_template = os.path.join(output_dir, f"{uuid.uuid4().hex}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 100 * 1024 * 1024,
        "match_filter": reject_too_long,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        output_path = ydl.prepare_filename(info)

    if not os.path.exists(output_path):
        raise RuntimeError("YouTube download did not produce an audio file")

    return output_path


def transcribe_audio(
    audio_filepath, output_dir, progress_callback=None, *, max_duration=None
):
    import note_seq
    import torchaudio
    from pytorch_mt3.pytorch_spectrograms import load_audio
    from .ml_pytorch import SAMPLE_RATE

    if max_duration is None:
        from django.conf import settings

        max_duration = settings.MAX_AUDIO_DURATION_SECONDS

    # Check duration before decoding the whole file into memory.
    try:
        info = torchaudio.info(audio_filepath)
    except Exception:
        # Some containers cannot report metadata; fall back to the loaded
        # waveform length below.
        info = None

    if info and info.sample_rate and info.num_frames:
        duration_seconds = info.num_frames / info.sample_rate
        if duration_seconds > max_duration:
            raise ValueError(
                f"Audio is {duration_seconds:.1f}s long; maximum is "
                f"{max_duration}s"
            )

    audio = (
        load_audio(audio_filepath, sample_rate=SAMPLE_RATE, mono=True).cpu().numpy()
    )

    duration_seconds = len(audio) / SAMPLE_RATE
    if duration_seconds > max_duration:
        raise ValueError(
            f"Audio is {duration_seconds:.1f}s long; maximum is {max_duration}s"
        )

    inference_model = _get_inference_model()
    est_ns = inference_model(audio, progress_callback=progress_callback)

    midi_filename = f"{uuid.uuid4().hex}.midi"
    output_path = os.path.join(output_dir, midi_filename)
    note_seq.sequence_proto_to_midi_file(est_ns, output_path)

    return output_path
