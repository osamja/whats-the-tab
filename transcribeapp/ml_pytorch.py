"""PyTorch-based MT3 inference adapter for Django.

This module intentionally delegates core inference/decoding to
`pytorch_mt3.standalone_inference` so Django and standalone paths stay aligned.
"""

import os
import sys
import note_seq

# Import PyTorch MT3 implementation from sibling directory.
pytorch_mt3_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
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

        # Project root is the repository root from this file location.
        project_root = os.path.dirname(os.path.dirname(__file__))
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
        print(f"  Generated note sequence with {len(est_ns.notes)} notes")
        return est_ns
