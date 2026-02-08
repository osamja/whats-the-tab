"""PyTorch MT3 Implementation Package.

This package contains a PyTorch re-implementation of the MT3 model
for music transcription.
"""

from .pytorch_model import MT3Model, MT3Config
from .pytorch_spectrograms import (
    SpectrogramExtractor,
    SpectrogramConfig,
    load_audio,
    audio_to_frames,
)

__all__ = [
    'MT3Model',
    'MT3Config',
    'SpectrogramExtractor',
    'SpectrogramConfig',
    'load_audio',
    'audio_to_frames',
]
