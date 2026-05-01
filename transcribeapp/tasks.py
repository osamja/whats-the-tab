import uuid, datetime


def get_audio_filename(is_mp4=False):
    ext = ".mp4" if is_mp4 else ".wav"
    return f"{datetime.datetime.now().isoformat()}-{uuid.uuid4().hex}{ext}"


def download_youtube_audio(youtube_url, output_dir):
    from pytube import YouTube
    import os

    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()

    filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(output_dir, filename)
    audio_stream.download(output_path=output_dir, filename=filename)

    return output_path


def transcribe_audio(audio_filepath, output_dir, progress_callback=None):
    import note_seq
    from .ml_pytorch import PyTorchInferenceModel, load_audio, SAMPLE_RATE
    import os

    checkpoint_path = "pytorch_mt3/mt3_pytorch_checkpoint.pt"

    inference_model = PyTorchInferenceModel(checkpoint_path, "mt3")
    audio = (
        load_audio(audio_filepath, sample_rate=SAMPLE_RATE, mono=True).cpu().numpy()
    )

    est_ns = inference_model(audio, progress_callback=progress_callback)

    midi_filename = f"{uuid.uuid4().hex}.midi"
    output_path = os.path.join(output_dir, midi_filename)
    note_seq.sequence_proto_to_midi_file(est_ns, output_path)

    return output_path
