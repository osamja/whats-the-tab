"""Test script for Django API with PyTorch backend.

This script tests the PyTorch inference through the Django transcription API.
"""

import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'musictranscription.settings')
django.setup()

from transcribeapp.models import AudioMIDI, AudioChunk, MIDIChunk
from transcribeapp.tasks import generate_midi_from_audio
from django.core.files import File
import librosa


def test_pytorch_django_transcription():
    """Test end-to-end transcription with PyTorch backend."""
    print("=" * 80)
    print("Testing Django API with PyTorch Backend")
    print("=" * 80)

    # Use existing test audio
    test_audio_path = 'john_mayer_neon_10sec.mp3'
    if not os.path.exists(test_audio_path):
        test_audio_path = 'test_10sec.mp3'

    if not os.path.exists(test_audio_path):
        print(f"Error: Test audio not found. Please create one first.")
        return 1

    print(f"\nTest audio: {test_audio_path}")

    # Create AudioMIDI object
    print("\nCreating AudioMIDI object...")
    with open(test_audio_path, 'rb') as f:
        audio_midi = AudioMIDI.objects.create(
            audio_file=File(f, name=os.path.basename(test_audio_path)),
            audio_filename=os.path.basename(test_audio_path),
            num_transcription_segments=2,  # Process 2 chunks for testing
            audio_chunk_length=5,  # 5 seconds per chunk
            status='processing'
        )

    print(f"✓ Created AudioMIDI object with ID: {audio_midi.id}")

    # Run transcription
    print("\nRunning transcription with PyTorch backend...")
    try:
        generate_midi_from_audio(audio_midi.id)

        # Reload to get updated status
        audio_midi.refresh_from_db()

        print(f"\n✓ Transcription complete!")
        print(f"  Status: {audio_midi.status}")
        print(f"  Segments processed: {audio_midi.current_segment}/{audio_midi.num_transcription_segments}")

        # Check MIDI chunks
        midi_chunks = MIDIChunk.objects.filter(audio_midi=audio_midi)
        print(f"\n  MIDI chunks created: {midi_chunks.count()}")

        for chunk in midi_chunks:
            print(f"    - Segment {chunk.segment_index}: {chunk.midi_file.name}")

        print("\n" + "=" * 80)
        print("SUCCESS! PyTorch transcription working through Django API")
        print("=" * 80)

        # Cleanup option
        cleanup = input("\nDelete test data? (y/n): ").lower().strip()
        if cleanup == 'y':
            # Delete MIDI files
            for chunk in midi_chunks:
                if chunk.midi_file and os.path.exists(chunk.midi_file.path):
                    os.remove(chunk.midi_file.path)
                chunk.delete()

            # Delete audio chunks
            for audio_chunk in audio_midi.audio_chunks.all():
                if audio_chunk.chunk_file and os.path.exists(audio_chunk.chunk_file.path):
                    os.remove(audio_chunk.chunk_file.path)
                audio_chunk.delete()

            # Delete audio file
            if audio_midi.audio_file and os.path.exists(audio_midi.audio_file.path):
                os.remove(audio_midi.audio_file.path)

            audio_midi.delete()
            print("✓ Test data cleaned up")

        return 0

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        audio_midi.status = 'failed'
        audio_midi.save()

        return 1


if __name__ == "__main__":
    sys.exit(test_pytorch_django_transcription())
