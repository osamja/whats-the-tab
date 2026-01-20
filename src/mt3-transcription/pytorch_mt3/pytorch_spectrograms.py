"""PyTorch implementation of spectrogram computation for MT3.

Matches the functionality of mt3/spectrograms.py and mt3/spectral_ops.py
using PyTorch/torchaudio instead of TensorFlow.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional
import numpy as np


class SpectrogramConfig:
    """Configuration for spectrogram computation.

    Matches mt3/spectrograms.py:SpectrogramConfig
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_width: int = 128,
        num_mel_bins: int = 512,
        fft_size: int = 2048,
        mel_lo_hz: float = 20.0,
    ):
        self.sample_rate = sample_rate
        self.hop_width = hop_width
        self.num_mel_bins = num_mel_bins
        self.fft_size = fft_size
        self.mel_lo_hz = mel_lo_hz

    @property
    def frames_per_second(self) -> float:
        """Number of spectrogram frames per second of audio."""
        return self.sample_rate / self.hop_width


class SpectrogramExtractor(torch.nn.Module):
    """Extract mel spectrograms from audio using PyTorch.

    This class provides functionality equivalent to compute_spectrogram
    in the original TensorFlow implementation.
    """

    def __init__(self, config: Optional[SpectrogramConfig] = None):
        """Initialize spectrogram extractor.

        Args:
            config: Spectrogram configuration. Uses defaults if None.
        """
        super().__init__()
        if config is None:
            config = SpectrogramConfig()
        self.config = config

        # Create mel spectrogram transform as a submodule
        # Note: torchaudio expects hop_length and win_length
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.fft_size,
            hop_length=config.hop_width,
            win_length=config.fft_size,
            n_mels=config.num_mel_bins,
            f_min=config.mel_lo_hz,
            f_max=config.sample_rate / 2,
            power=2.0,  # Power spectrogram
            normalized=False,
        )

    def __call__(
        self,
        audio: torch.Tensor,
        pad_to_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute mel spectrogram from audio waveform.

        Args:
            audio: Audio waveform [batch, num_samples] or [num_samples]
            pad_to_length: If provided, pad/trim output to this many frames

        Returns:
            Log mel spectrogram [batch, time, freq] or [time, freq]
        """
        # Ensure audio is 2D (batch, samples)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute mel spectrogram
        # Output shape: [batch, n_mels, time]
        mel_spec = self.mel_spectrogram(audio)

        # Convert to log scale (add small epsilon to avoid log(0))
        log_mel_spec = torch.log(mel_spec + 1e-6)

        # Transpose to [batch, time, freq] to match MT3 convention
        log_mel_spec = log_mel_spec.transpose(1, 2)

        # Pad or trim to desired length if specified
        if pad_to_length is not None:
            current_length = log_mel_spec.size(1)
            if current_length < pad_to_length:
                # Pad with zeros
                padding = pad_to_length - current_length
                log_mel_spec = torch.nn.functional.pad(
                    log_mel_spec, (0, 0, 0, padding), value=0.0
                )
            elif current_length > pad_to_length:
                # Trim
                log_mel_spec = log_mel_spec[:, :pad_to_length, :]

        # Remove batch dimension if input was 1D
        if squeeze_output:
            log_mel_spec = log_mel_spec.squeeze(0)

        return log_mel_spec


def split_audio(
    samples: torch.Tensor,
    config: SpectrogramConfig,
) -> torch.Tensor:
    """Split audio into frames.

    Matches mt3/spectrograms.py:split_audio

    Args:
        samples: Audio samples [num_samples] or [batch, num_samples]
        config: Spectrogram configuration

    Returns:
        Framed audio [batch, num_frames, frame_length]
    """
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)

    # Pad audio to ensure it divides evenly into frames
    num_samples = samples.size(-1)
    frame_size = config.hop_width
    padding_needed = (frame_size - (num_samples % frame_size)) % frame_size

    if padding_needed > 0:
        samples = torch.nn.functional.pad(samples, (0, padding_needed))

    # Use unfold to create frames
    # This is equivalent to tf.signal.frame
    frames = samples.unfold(-1, frame_size, frame_size)

    return frames


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> torch.Tensor:
    """Load audio file and resample if necessary.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        mono: If True, convert to mono

    Returns:
        Audio tensor [num_samples] or [channels, num_samples]
    """
    # Load audio using torchaudio
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if requested
    if mono and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Return as 1D tensor if mono
    if mono:
        waveform = waveform.squeeze(0)

    return waveform


def audio_to_frames(
    audio: torch.Tensor,
    config: Optional[SpectrogramConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert audio to spectrogram frames with timestamps.

    Matches the functionality used in the JAX InferenceModel.

    Args:
        audio: Audio samples [num_samples]
        config: Spectrogram configuration

    Returns:
        frames: Log mel spectrogram frames [num_frames, num_mel_bins]
        frame_times: Timestamp for each frame [num_frames]
    """
    if config is None:
        config = SpectrogramConfig()

    # Store original device
    device = audio.device

    # Pad audio to frame boundary
    frame_size = config.hop_width
    num_samples = len(audio)
    padding = (frame_size - (num_samples % frame_size)) % frame_size

    if padding > 0:
        audio = torch.nn.functional.pad(audio, (0, padding))

    # Compute spectrogram (move extractor to same device as audio)
    extractor = SpectrogramExtractor(config).to(device)
    frames = extractor(audio)  # [num_frames, num_mel_bins]

    # Compute timestamps
    num_frames = frames.size(0)
    frame_times = torch.arange(num_frames, dtype=torch.float32, device=device) / config.frames_per_second

    return frames, frame_times


def compute_spectrogram_batch(
    audio_batch: torch.Tensor,
    config: Optional[SpectrogramConfig] = None,
) -> torch.Tensor:
    """Compute spectrograms for a batch of audio samples.

    Args:
        audio_batch: Audio samples [batch, num_samples]
        config: Spectrogram configuration

    Returns:
        Log mel spectrograms [batch, num_frames, num_mel_bins]
    """
    if config is None:
        config = SpectrogramConfig()

    extractor = SpectrogramExtractor(config)
    return extractor(audio_batch)


# Example usage
if __name__ == "__main__":
    # Create a simple test
    print("Testing PyTorch spectrogram extraction...")

    # Create synthetic audio (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    audio = torch.randn(num_samples)

    # Create config
    config = SpectrogramConfig()

    # Extract spectrogram
    extractor = SpectrogramExtractor(config)
    spectrogram = extractor(audio)

    print(f"Audio shape: {audio.shape}")
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Expected time frames: ~{int(duration * config.frames_per_second)}")
    print(f"Actual time frames: {spectrogram.shape[0]}")
    print(f"Mel bins: {spectrogram.shape[1]}")
    print(f"Frames per second: {config.frames_per_second}")

    # Test with batch
    audio_batch = torch.randn(4, num_samples)
    spec_batch = compute_spectrogram_batch(audio_batch, config)
    print(f"\nBatch spectrogram shape: {spec_batch.shape}")

    # Test audio_to_frames
    frames, times = audio_to_frames(audio, config)
    print(f"\nFrames shape: {frames.shape}")
    print(f"Frame times shape: {times.shape}")
    print(f"First 5 frame times: {times[:5]}")
    print(f"Last 5 frame times: {times[-5:]}")
