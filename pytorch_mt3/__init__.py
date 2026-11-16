"""PyTorch MT3 Implementation Package.

This package contains a PyTorch re-implementation of the MT3 model
for music transcription.

Imports are intentionally lazy so that importing the package does not require
Torch (which is only installed in the GPU worker environment).
"""

from importlib import import_module

_EXPORTS = {
    "MT3Model": (".pytorch_model", "MT3Model"),
    "MT3Config": (".pytorch_model", "MT3Config"),
    "SpectrogramExtractor": (".pytorch_spectrograms", "SpectrogramExtractor"),
    "SpectrogramConfig": (".pytorch_spectrograms", "SpectrogramConfig"),
    "load_audio": (".pytorch_spectrograms", "load_audio"),
    "audio_to_frames": (".pytorch_spectrograms", "audio_to_frames"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
