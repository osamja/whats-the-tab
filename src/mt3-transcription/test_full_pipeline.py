"""Test complete Audio → MIDI pipeline with PyTorch MT3.

This script tests the full pipeline:
1. Load audio file
2. Run PyTorch inference
3. Decode tokens to note sequence
4. Save as MIDI file
5. Verify MIDI output
"""

import os
import sys
import time
import importlib.util

# Load modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Add MT3 to path for imports
sys.path.insert(0, 'mt3')

print("Loading modules...")
ml_pytorch = load_module('ml_pytorch', 'musictranscription/transcribeapp/ml_pytorch.py')

PyTorchInferenceModel = ml_pytorch.PyTorchInferenceModel
transcribe_audio = ml_pytorch.transcribe_audio
download_midi = ml_pytorch.download_midi


def test_full_pipeline():
    """Test complete Audio → MIDI pipeline."""
    print("=" * 80)
    print("Testing Full Audio → MIDI Pipeline with PyTorch MT3")
    print("=" * 80)

    # Find test audio
    test_files = [
        'john_mayer_neon_10sec.mp3',
        'test_10sec.mp3',
        '/home/samus/whats-the-tab/dataset/in-the-morning-jcole.mp3',
    ]

    test_audio = None
    for f in test_files:
        if os.path.exists(f):
            test_audio = f
            break

    if not test_audio:
        print("\n❌ No test audio found!")
        print("Please create test audio with:")
        print("  python pytorch_mt3/example_pytorch_inference.py --create-test-audio")
        return 1

    print(f"\n📁 Test audio: {test_audio}")

    # Initialize PyTorch inference model
    print("\n🔧 Initializing PyTorch MT3 model...")
    checkpoint_path = 'pytorch_mt3/mt3_pytorch_checkpoint.pt'

    try:
        model = PyTorchInferenceModel(
            checkpoint_path=checkpoint_path,
            model_type='mt3'
        )
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load audio
    print(f"\n🎵 Loading audio...")
    import librosa
    audio, sr = librosa.load(test_audio, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Samples: {len(audio):,}")

    # Limit to 30 seconds for testing
    max_duration = 30
    if duration > max_duration:
        print(f"  Trimming to {max_duration}s for faster testing...")
        audio = audio[:int(sr * max_duration)]
        duration = max_duration

    # Transcribe
    print(f"\n🎹 Transcribing audio to MIDI...")
    start_time = time.time()

    try:
        est_ns = transcribe_audio(audio, model, play=False)
        elapsed = time.time() - start_time

        print(f"\n✓ Transcription complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Real-time factor: {elapsed/duration:.2f}x")

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Analyze note sequence
    print(f"\n📊 Note Sequence Analysis:")
    print(f"  Total notes: {len(est_ns.notes)}")

    if len(est_ns.notes) > 0:
        pitches = [note.pitch for note in est_ns.notes]
        velocities = [note.velocity for note in est_ns.notes]
        durations = [note.end_time - note.start_time for note in est_ns.notes]

        print(f"  Pitch range: {min(pitches)} - {max(pitches)} (MIDI)")
        print(f"  Velocity range: {min(velocities)} - {max(velocities)}")
        print(f"  Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
        print(f"  Total duration: {est_ns.total_time:.2f}s")

        print(f"\n  First 10 notes:")
        for i, note in enumerate(est_ns.notes[:10]):
            note_name = librosa.midi_to_note(note.pitch)
            print(f"    {i+1}. {note_name:4s} (MIDI {note.pitch:3d}) "
                  f"@ {note.start_time:6.2f}s - {note.end_time:6.2f}s "
                  f"vel={note.velocity:3d}")

        # Check for tempo
        if hasattr(est_ns, 'tempos') and len(est_ns.tempos) > 0:
            print(f"\n  Tempo: {est_ns.tempos[0].qpm:.1f} BPM")

    else:
        print("  ⚠️  Warning: No notes detected!")
        print("  This might indicate:")
        print("    - Silent audio input")
        print("    - Model not properly trained")
        print("    - Vocabulary decoding issue")

    # Save MIDI
    output_path = 'test_output.midi'
    print(f"\n💾 Saving MIDI to: {output_path}")

    try:
        download_midi(est_ns, output_path)
        file_size = os.path.getsize(output_path)
        print(f"  ✓ MIDI file saved ({file_size:,} bytes)")

        # Verify MIDI file
        print(f"\n🔍 Verifying MIDI file...")
        import mido
        mid = mido.MidiFile(output_path)

        print(f"  Type: {mid.type}")
        print(f"  Ticks per beat: {mid.ticks_per_beat}")
        print(f"  Tracks: {len(mid.tracks)}")

        # Count note events
        note_ons = sum(1 for track in mid.tracks for msg in track if msg.type == 'note_on')
        note_offs = sum(1 for track in mid.tracks for msg in track if msg.type == 'note_off')

        print(f"  Note events: {note_ons} note_on, {note_offs} note_off")

        if note_ons > 0:
            print(f"\n  ✓ MIDI file contains notes!")
        else:
            print(f"\n  ⚠️  MIDI file has no notes")

    except Exception as e:
        print(f"  ❌ Error saving/verifying MIDI: {e}")
        import traceback
        traceback.print_exc()

    # Success summary
    print("\n" + "=" * 80)
    if len(est_ns.notes) > 0:
        print("✅ SUCCESS! Full Audio → MIDI Pipeline Working!")
        print("=" * 80)
        print(f"\n✓ Audio loaded: {duration:.2f}s")
        print(f"✓ Inference completed: {elapsed:.2f}s ({elapsed/duration:.2f}x real-time)")
        print(f"✓ Notes generated: {len(est_ns.notes)}")
        print(f"✓ MIDI saved: {output_path}")
        print("\nYou can now:")
        print(f"  1. Play the MIDI: timidity {output_path}")
        print(f"  2. View in DAW: open {output_path}")
        print(f"  3. Convert to audio: timidity {output_path} -Ow -o output.wav")
    else:
        print("⚠️  Pipeline completed but no notes detected")
        print("=" * 80)
        print("\nThis might be normal for:")
        print("  - Very quiet or silent audio")
        print("  - Non-musical audio (speech, noise)")
        print("\nTry with different audio that has clear musical content.")

    return 0 if len(est_ns.notes) > 0 else 1


if __name__ == "__main__":
    sys.exit(test_full_pipeline())
