"""PyTorch-based MT3 inference adapter for Django.

This module intentionally delegates core inference/decoding to
`pytorch_mt3.standalone_inference` so Django and standalone paths stay aligned.
"""

import os
import sys
import note_seq

# Import PyTorch MT3 implementation from sibling directory.
pytorch_mt3_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "pytorch_mt3",
)
sys.path.insert(0, pytorch_mt3_path)

from pytorch_mt3.pytorch_spectrograms import load_audio
from pytorch_mt3.standalone_inference import StandaloneMT3

SAMPLE_RATE = 16000


class PyTorchInferenceModel:
    """PyTorch wrapper for MT3 music transcription.

    This class provides the same interface as the JAX InferenceModel
    but uses PyTorch for inference.
    """

    def __init__(self, checkpoint_path, model_type="mt3"):
        """Initialize PyTorch MT3 model wrapper.

        Args:
            checkpoint_path: Path to PyTorch checkpoint.
            model_type: MT3 variant name. Only `mt3` is supported in the
                standalone PyTorch implementation.
        """
        if model_type != "mt3":
            raise ValueError(
                f"Unsupported model_type for PyTorch standalone backend: {model_type}"
            )
        self.model_type = model_type
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.model = StandaloneMT3(checkpoint_path, device=None)

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str) -> str:
        """Resolve checkpoint path for Django runtime.

        `tasks.py` passes a relative path. Resolve it against the project root
        so StandaloneMT3 doesn't try to download into a cwd-dependent location.
        """
        if os.path.isabs(checkpoint_path):
            return checkpoint_path

        if os.path.exists(checkpoint_path):
            return checkpoint_path

        # Project root is src/mt3-transcription from this file location.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidate = os.path.join(project_root, checkpoint_path)
        if os.path.exists(candidate):
            return candidate

        # Fallback: expected location inside pytorch_mt3 directory.
        return os.path.join(pytorch_mt3_path, os.path.basename(checkpoint_path))

    def __call__(self, audio):
        """Infer note sequence from audio samples.

        Args:
            audio: 1-d numpy array of audio samples (16kHz) for a single example.

        Returns:
            A note_sequence of the transcribed audio.
        """
        print(f"  Processing audio: {len(audio)/SAMPLE_RATE:.2f}s")
        result = self.model.transcribe(audio)
        est_ns = result.get("note_sequence")
        if est_ns is None:
            raise RuntimeError("PyTorch transcription completed without note_sequence.")
        print(f"  ✓ Generated note sequence with {len(est_ns.notes)} notes")
        return est_ns


# Helper functions for Django integration

def transcribe_audio(audio, inference_model, play=False):
    """Transcribe audio using inference model.

    Args:
        audio: Audio samples (numpy array)
        inference_model: PyTorchInferenceModel or JAX InferenceModel
        play: Whether to play the result (not implemented for PyTorch)

    Returns:
        note_seq.NoteSequence
    """
    est_ns = inference_model(audio)

    if play:
        # Note: Playing is not implemented in PyTorch version yet
        print("Warning: Audio playback not implemented for PyTorch model")

    return est_ns


def download_midi(est_ns, download_path='transcription.midi'):
    """Save note sequence to MIDI file.

    Args:
        est_ns: note_seq.NoteSequence
        download_path: Output path for MIDI file
    """
    note_seq.sequence_proto_to_midi_file(est_ns, download_path)


def transcribe_and_download(audio_midi, split_filenames, inference_model):
    """Transcribe audio chunks and save MIDI files.

    This function is compatible with both JAX and PyTorch inference models.

    Args:
        audio_midi: AudioMIDI Django model instance
        split_filenames: List of audio chunk file paths
        inference_model: PyTorchInferenceModel or JAX InferenceModel
    """
    from .models import MIDIChunk
    from django.core.files import File

    # Delete existing MIDI chunks
    MIDIChunk.objects.filter(audio_midi=audio_midi).delete()

    for i, audio_filename in enumerate(split_filenames):
        print(f"Transcribing chunk {i+1}/{len(split_filenames)}: {audio_filename}")

        # Load audio via PyTorch path to keep preprocessing dependencies aligned.
        audio = load_audio(audio_filename, sample_rate=SAMPLE_RATE, mono=True).cpu().numpy()

        # Transcribe
        est_ns = transcribe_audio(audio, inference_model)

        # Create MIDI filename
        midi_filename = 'midi_chunks/' + audio_filename.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.midi'
        download_midi(est_ns, midi_filename)

        # Save to database
        with open(midi_filename, 'rb') as midi_file:
            midi_chunk = MIDIChunk(
                audio_midi=audio_midi,
                midi_file=File(midi_file, name=os.path.basename(midi_filename)),
                segment_index=i
            )
            midi_chunk.save()

        # Update progress
        audio_midi.current_segment = i + 1
        audio_midi.save()

        print(f"✓ Chunk {i+1} complete")
