"""Step-by-step PyTorch vs regular MT3 pipeline comparison.

Default flow:
- Load the same MP3.
- Take a fixed 30s window.
- Compare only the first 2.048s chunk (256 frames at 16k/128 hop).
- Report intermediate stats and divergence hints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from pytorch_spectrograms import SpectrogramConfig, audio_to_frames, load_audio
from standalone_inference import StandaloneMT3
from mt3_decoding import vocabularies as pt_vocabularies


def _arr_stats(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x)
    if x.size == 0:
        return {"shape": list(x.shape), "dtype": str(x.dtype), "size": 0}
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "size": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(np.square(x.astype(np.float64))))),
        "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16],
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _event_counts(codec, decoded_tokens: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in decoded_tokens.tolist():
        if 0 <= t < codec.num_classes:
            ev = codec.decode_event_index(int(t))
            counts[ev.type] = counts.get(ev.type, 0) + 1
    return counts


def _note_summary(ns) -> Dict[str, Any]:
    notes = list(ns.notes)
    if not notes:
        return {"notes": 0}
    pitches = [n.pitch for n in notes]
    starts = [n.start_time for n in notes]
    ends = [n.end_time for n in notes]
    progs = [n.program for n in notes if not n.is_drum]
    return {
        "notes": len(notes),
        "unique_pitches": len(set(pitches)),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "start_min": float(min(starts)),
        "start_max": float(max(starts)),
        "end_max": float(max(ends)),
        "unique_programs": len(set(progs)),
    }


def run_comparison(
    audio_path: str,
    checkpoint_path: str,
    regular_checkpoint_path: str,
    window_start_sec: float,
    window_duration_sec: float,
    chunk_duration_sec: float,
) -> Dict[str, Any]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    sample_rate = 16000
    chunk_samples = int(round(chunk_duration_sec * sample_rate))
    window_start = int(round(window_start_sec * sample_rate))
    window_len = int(round(window_duration_sec * sample_rate))

    # ---------- PyTorch pipeline ----------
    pt_audio_full = load_audio(audio_path, sample_rate=sample_rate, mono=True).cpu().numpy()
    pt_window = pt_audio_full[window_start: window_start + window_len]
    pt_chunk = pt_window[:chunk_samples]

    pt_model = StandaloneMT3(checkpoint_path, device="cpu")
    pt_spec_cfg = SpectrogramConfig()
    pt_frames, pt_times = audio_to_frames(torch.from_numpy(pt_chunk).float(), pt_spec_cfg)
    pt_frames_np = pt_frames.cpu().numpy()
    pt_times_np = pt_times.cpu().numpy()

    chunk_size = pt_model.config.max_encoder_length
    pt_chunk_frames = pt_frames[:chunk_size]
    if pt_chunk_frames.shape[0] < chunk_size:
        pt_chunk_frames = torch.nn.functional.pad(
            pt_chunk_frames, (0, 0, 0, chunk_size - pt_chunk_frames.shape[0])
        )
    pt_chunk_frames_np = pt_chunk_frames.cpu().numpy()

    with torch.no_grad():
        pt_tokens_encoded_raw = pt_model.model.generate(
            pt_chunk_frames.unsqueeze(0),
            max_length=pt_model.config.max_decoder_length,
            start_token_id=0,
            eos_token_id=1,
            temperature=1.0,
        ).squeeze(0).cpu().numpy().astype(np.int32)
    pt_tokens_encoded = np.array(
        pt_model._trim_generated_prefix(pt_tokens_encoded_raw.tolist()), dtype=np.int32
    )

    pt_codec = pt_vocabularies.build_codec(
        vocab_config=pt_vocabularies.VocabularyConfig(num_velocity_bins=1)
    )
    pt_vocab = pt_vocabularies.vocabulary_from_codec(pt_codec)
    pt_tokens_decoded = np.array(pt_vocab.decode(pt_tokens_encoded.tolist()), dtype=np.int32)
    if pt_tokens_decoded.size:
        first_valid = np.argmax(pt_tokens_decoded >= 0)
        if pt_tokens_decoded[first_valid] >= 0:
            pt_tokens_decoded = pt_tokens_decoded[first_valid:]
        else:
            pt_tokens_decoded = np.array([], np.int32)
    pt_ns = pt_model._decode_to_note_sequence([pt_tokens_encoded.tolist()], [window_start_sec])

    # ---------- Regular/JAX pipeline ----------
    # Import after env vars are set.
    musictranscription_dir = str(Path(__file__).resolve().parents[1] / "musictranscription")
    if musictranscription_dir not in sys.path:
        sys.path.insert(0, musictranscription_dir)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "musictranscription.settings")
    import django  # pylint: disable=import-outside-toplevel
    django.setup()
    import librosa  # pylint: disable=import-outside-toplevel
    import jax  # pylint: disable=import-outside-toplevel
    from transcribeapp.ml import InferenceModel  # pylint: disable=import-outside-toplevel

    reg_model = InferenceModel(checkpoint_path=regular_checkpoint_path, model_type="mt3")
    reg_audio_full, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    reg_window = reg_audio_full[window_start: window_start + window_len]
    reg_chunk = reg_window[:chunk_samples]

    reg_frames_raw, reg_times_raw = reg_model._audio_to_frames(reg_chunk)
    reg_ds = reg_model.audio_to_dataset(reg_chunk)
    reg_ds_pp = reg_model.preprocess(reg_ds)
    reg_examples = list(reg_ds_pp.as_numpy_iterator())
    reg_example0 = reg_examples[0]

    reg_model_ds = reg_model.model.FEATURE_CONVERTER_CLS(pack=False)(
        reg_ds_pp, task_feature_lengths=reg_model.sequence_length
    ).batch(reg_model.batch_size)
    reg_batch = next(reg_model_ds.as_numpy_iterator())

    raw_pred, _ = reg_model._predict_fn(
        reg_model._train_state.params, reg_batch, jax.random.PRNGKey(0)
    )
    reg_tokens_encoded = np.asarray(raw_pred)[0].astype(np.int32)
    reg_tokens_decoded = reg_model.predict_tokens(reg_batch, seed=0)[0].astype(np.int32)
    reg_tokens_trimmed = reg_model._trim_eos(reg_tokens_decoded)
    reg_pred = reg_model.postprocess(reg_tokens_decoded, reg_example0)
    reg_ns = reg_model(reg_chunk)

    # ---------- Comparisons ----------
    reg_spec = np.asarray(reg_example0["inputs"])
    pt_spec = np.asarray(pt_chunk_frames_np)
    spec_n = min(pt_spec.shape[0], reg_spec.shape[0])
    spec_m = min(pt_spec.shape[1], reg_spec.shape[1])
    pt_spec_cmp = pt_spec[:spec_n, :spec_m]
    reg_spec_cmp = reg_spec[:spec_n, :spec_m]

    min_wave = min(pt_chunk.size, reg_chunk.size)
    pt_wave_cmp = pt_chunk[:min_wave]
    reg_wave_cmp = reg_chunk[:min_wave]

    report = {
        "config": {
            "audio_path": audio_path,
            "window_start_sec": window_start_sec,
            "window_duration_sec": window_duration_sec,
            "chunk_duration_sec": chunk_duration_sec,
            "sample_rate": sample_rate,
            "chunk_samples": chunk_samples,
        },
        "pytorch": {
            "audio_chunk_stats": _arr_stats(pt_chunk),
            "frames_stats": _arr_stats(pt_frames_np),
            "frame_times_stats": _arr_stats(pt_times_np),
            "model_input_chunk_stats": _arr_stats(pt_chunk_frames_np),
            "tokens_encoded_stats": _arr_stats(pt_tokens_encoded),
            "tokens_decoded_stats": _arr_stats(pt_tokens_decoded),
            "tokens_decoded_head": pt_tokens_decoded[:64].tolist(),
            "event_counts": _event_counts(pt_codec, pt_tokens_decoded),
            "notes": _note_summary(pt_ns),
        },
        "regular": {
            "audio_chunk_stats": _arr_stats(reg_chunk),
            "frames_raw_stats": _arr_stats(np.asarray(reg_frames_raw)),
            "frame_times_raw_stats": _arr_stats(np.asarray(reg_times_raw)),
            "preprocessed_inputs_stats": _arr_stats(reg_spec),
            "tokens_encoded_stats": _arr_stats(reg_tokens_encoded),
            "tokens_decoded_stats": _arr_stats(reg_tokens_decoded),
            "tokens_trimmed_stats": _arr_stats(reg_tokens_trimmed),
            "tokens_decoded_head": reg_tokens_decoded[:64].tolist(),
            "postprocess_start_time": float(reg_pred["start_time"]),
            "event_counts": _event_counts(reg_model.codec, reg_tokens_trimmed),
            "notes": _note_summary(reg_ns),
        },
        "diff": {
            "audio_mae": float(np.mean(np.abs(pt_wave_cmp - reg_wave_cmp))),
            "audio_corr": _corr(pt_wave_cmp, reg_wave_cmp),
            "spectrogram_mae": float(np.mean(np.abs(pt_spec_cmp - reg_spec_cmp))),
            "spectrogram_corr": _corr(pt_spec_cmp, reg_spec_cmp),
            "encoded_token_equal_prefix_len": int(
                np.argmax(np.concatenate(([False], pt_tokens_encoded != reg_tokens_encoded[: len(pt_tokens_encoded)])))
                if len(pt_tokens_encoded) <= len(reg_tokens_encoded) and np.any(pt_tokens_encoded != reg_tokens_encoded[: len(pt_tokens_encoded)])
                else min(len(pt_tokens_encoded), len(reg_tokens_encoded))
            ),
            "decoded_token_equal_prefix_len": int(
                np.argmax(np.concatenate(([False], pt_tokens_decoded != reg_tokens_trimmed[: len(pt_tokens_decoded)])))
                if len(pt_tokens_decoded) <= len(reg_tokens_trimmed) and np.any(pt_tokens_decoded != reg_tokens_trimmed[: len(pt_tokens_decoded)])
                else min(len(pt_tokens_decoded), len(reg_tokens_trimmed))
            ),
            "pytorch_note_count": int(len(pt_ns.notes)),
            "regular_note_count": int(len(reg_ns.notes)),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PyTorch and regular MT3 pipelines step-by-step")
    parser.add_argument("--audio", required=True, help="Path to input audio")
    parser.add_argument(
        "--pytorch-checkpoint",
        default="mt3_pytorch_checkpoint.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--regular-checkpoint",
        default=str(Path(__file__).resolve().parents[1] / "musictranscription" / "checkpoints" / "mt3"),
        help="Path to regular MT3 checkpoint directory",
    )
    parser.add_argument("--window-start", type=float, default=30.0)
    parser.add_argument("--window-duration", type=float, default=30.0)
    parser.add_argument("--chunk-duration", type=float, default=2.048)
    parser.add_argument("--out", default="/tmp/mt3_debug_report.json")
    args = parser.parse_args()

    report = run_comparison(
        audio_path=args.audio,
        checkpoint_path=args.pytorch_checkpoint,
        regular_checkpoint_path=args.regular_checkpoint,
        window_start_sec=args.window_start,
        window_duration_sec=args.window_duration,
        chunk_duration_sec=args.chunk_duration,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Saved report: {out_path}")
    print(
        "Quick diff: "
        f"audio_mae={report['diff']['audio_mae']:.6f}, "
        f"spec_mae={report['diff']['spectrogram_mae']:.6f}, "
        f"pt_notes={report['diff']['pytorch_note_count']}, "
        f"reg_notes={report['diff']['regular_note_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
