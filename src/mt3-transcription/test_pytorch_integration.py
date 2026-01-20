"""Simple test of PyTorch integration without Django.

This script tests the PyTorch ML module directly.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'musictranscription'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch_mt3'))

def test_pytorch_ml_module():
    """Test PyTorch ML module."""
    print("=" * 80)
    print("Testing PyTorch ML Module")
    print("=" * 80)

    try:
        # Import PyTorch ML module directly
        print("\nImporting PyTorch ML module...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ml_pytorch",
            "musictranscription/transcribeapp/ml_pytorch.py"
        )
        ml_pytorch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_pytorch)

        # Create inference model
        print("Creating PyTorch inference model...")
        checkpoint_path = 'pytorch_mt3/mt3_pytorch_checkpoint.pt'

        model = ml_pytorch.PyTorchInferenceModel(
            checkpoint_path=checkpoint_path,
            model_type='mt3'
        )

        print(f"✓ Model created successfully")
        print(f"  Device: {model.device}")
        print(f"  Vocabulary size: {model.vocabulary.vocab_size}")
        print(f"  Input length: {model.inputs_length}")

        # Test with audio
        test_audio_file = 'john_mayer_neon_10sec.mp3'
        if not os.path.exists(test_audio_file):
            test_audio_file = 'test_10sec.mp3'

        if os.path.exists(test_audio_file):
            print(f"\nTesting transcription on: {test_audio_file}")

            import librosa
            audio, sr = librosa.load(test_audio_file, sr=16000, mono=True)
            print(f"  Audio loaded: {len(audio)/sr:.2f}s")

            print("  Running inference...")
            import time
            start = time.time()

            est_ns = model(audio)

            elapsed = time.time() - start

            print(f"  ✓ Inference complete in {elapsed:.2f}s")
            print(f"  Note sequence has {len(est_ns.notes)} notes")

            # Print first few notes
            if len(est_ns.notes) > 0:
                print(f"\n  First 5 notes:")
                for i, note in enumerate(est_ns.notes[:5]):
                    print(f"    {i+1}. Pitch: {note.pitch}, Start: {note.start_time:.2f}s, End: {note.end_time:.2f}s")
            else:
                print("  Warning: No notes detected")

            # Save MIDI
            midi_path = 'test_pytorch_output.midi'
            ml_pytorch.download_midi(est_ns, midi_path)
            print(f"\n  ✓ MIDI saved to: {midi_path}")

            print("\n" + "=" * 80)
            print("SUCCESS! PyTorch ML module working")
            print("=" * 80)

            return 0
        else:
            print(f"\nWarning: Test audio file not found")
            print("Skipping audio test")
            return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_pytorch_ml_module())
