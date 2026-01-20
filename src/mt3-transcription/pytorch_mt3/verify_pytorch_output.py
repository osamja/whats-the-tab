"""
Verify PyTorch MT3 implementation produces reasonable outputs.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_mt3.standalone_inference import StandaloneMT3


def analyze_tokens(tokens, audio_duration):
    """Analyze token distribution and characteristics."""
    print("\n" + "="*80)
    print("Token Analysis")
    print("="*80)

    # Basic statistics
    tokens_array = np.array(tokens)
    print(f"\nBasic Statistics:")
    print(f"  Total tokens: {len(tokens_array):,}")
    print(f"  Tokens per second: {len(tokens_array) / audio_duration:.1f}")
    print(f"  Token range: [{tokens_array.min()}, {tokens_array.max()}]")
    print(f"  Mean: {tokens_array.mean():.2f}")
    print(f"  Std: {tokens_array.std():.2f}")

    # Vocabulary usage
    unique_tokens = np.unique(tokens_array)
    print(f"\nVocabulary Usage:")
    print(f"  Unique tokens: {len(unique_tokens)}")
    print(f"  Vocabulary coverage: {len(unique_tokens)/1536*100:.1f}%")

    # Token distribution
    token_counts = np.bincount(tokens_array, minlength=1536)
    most_common_idx = np.argsort(token_counts)[-10:][::-1]
    print(f"\nTop 10 most common tokens:")
    for idx in most_common_idx:
        if token_counts[idx] > 0:
            print(f"  Token {idx:4d}: {token_counts[idx]:5d} times ({token_counts[idx]/len(tokens_array)*100:5.1f}%)")

    # Pattern analysis
    print(f"\nPattern Analysis:")
    # Check for repeated tokens
    repeats = 0
    for i in range(1, len(tokens_array)):
        if tokens_array[i] == tokens_array[i-1]:
            repeats += 1
    print(f"  Consecutive repeats: {repeats} ({repeats/len(tokens_array)*100:.1f}%)")

    # Check for runs of the same token
    max_run = 1
    current_run = 1
    for i in range(1, len(tokens_array)):
        if tokens_array[i] == tokens_array[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    print(f"  Maximum run length: {max_run}")

    # Distribution entropy (measure of diversity)
    probs = token_counts[token_counts > 0] / len(tokens_array)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(unique_tokens))
    print(f"  Token entropy: {entropy:.2f} / {max_entropy:.2f} (max)")
    print(f"  Normalized entropy: {entropy/max_entropy:.2%}")

    # Validation checks
    print(f"\n" + "="*80)
    print("Validation Checks")
    print("="*80)

    checks_passed = 0
    total_checks = 0

    # Check 1: Tokens in valid range
    total_checks += 1
    if tokens_array.min() >= 0 and tokens_array.max() < 1536:
        print(f"  ✓ All tokens within valid vocabulary range [0, 1536)")
        checks_passed += 1
    else:
        print(f"  ✗ Some tokens outside valid range")

    # Check 2: Token diversity
    total_checks += 1
    if len(unique_tokens) > 10:
        print(f"  ✓ Good token diversity ({len(unique_tokens)} unique tokens)")
        checks_passed += 1
    else:
        print(f"  ✗ Low token diversity ({len(unique_tokens)} unique tokens)")

    # Check 3: Not all same token
    total_checks += 1
    if len(unique_tokens) > 1:
        print(f"  ✓ Tokens show variation (not all identical)")
        checks_passed += 1
    else:
        print(f"  ✗ All tokens are identical!")

    # Check 4: Reasonable token density
    total_checks += 1
    tokens_per_sec = len(tokens_array) / audio_duration
    if 50 < tokens_per_sec < 2000:  # Reasonable range for music transcription
        print(f"  ✓ Token density looks reasonable ({tokens_per_sec:.1f} tokens/sec)")
        checks_passed += 1
    else:
        print(f"  ⚠ Unusual token density ({tokens_per_sec:.1f} tokens/sec)")

    # Check 5: No excessive repetition
    total_checks += 1
    if max_run < 100:  # No token repeats more than 100 times in a row
        print(f"  ✓ No excessive repetition (max run: {max_run})")
        checks_passed += 1
    else:
        print(f"  ✗ Excessive repetition detected (max run: {max_run})")

    # Check 6: Entropy check
    total_checks += 1
    if entropy/max_entropy > 0.5:  # Using more than half the available entropy
        print(f"  ✓ Good token distribution diversity ({entropy/max_entropy:.1%})")
        checks_passed += 1
    else:
        print(f"  ⚠ Low distribution diversity ({entropy/max_entropy:.1%})")

    print(f"\n" + "="*80)
    print(f"Validation Result: {checks_passed}/{total_checks} checks passed")
    print("="*80)

    if checks_passed == total_checks:
        print("\n✓ PyTorch implementation appears to be working correctly!")
        return True
    elif checks_passed >= total_checks * 0.7:
        print("\n⚠ PyTorch implementation is mostly working, some minor issues")
        return True
    else:
        print("\n✗ PyTorch implementation may have issues")
        return False


def quick_verify(audio_path, max_duration=10.0):
    """Quick verification of PyTorch implementation."""
    print("="*80)
    print("PyTorch MT3 Implementation Verification")
    print("="*80)
    print(f"\nAudio file: {audio_path}")
    print(f"Max duration: {max_duration}s")

    checkpoint_path = "/home/samus/programming-projects/whats-the-tab/src/mt3-transcription/pytorch_mt3/mt3_pytorch_checkpoint.pt"

    if not os.path.exists(checkpoint_path):
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        return False

    # Load model
    print(f"\nLoading PyTorch MT3...")
    model = StandaloneMT3(checkpoint_path)

    # Load and process audio
    print(f"\nLoading audio...")
    from pytorch_mt3.pytorch_spectrograms import load_audio
    audio = load_audio(audio_path, sample_rate=16000, mono=True)

    # Trim to max duration
    max_samples = int(max_duration * 16000)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    actual_duration = len(audio) / 16000
    print(f"  Audio duration: {actual_duration:.2f}s")

    # Run inference
    print(f"\nRunning inference...")
    result = model.transcribe(audio)

    # Flatten tokens
    all_tokens = []
    for chunk_tokens in result['tokens']:
        all_tokens.extend(chunk_tokens)

    print(f"\n✓ Inference complete!")
    print(f"  Chunks processed: {result['num_chunks']}")
    print(f"  Total tokens: {len(all_tokens):,}")

    # Analyze tokens
    success = analyze_tokens(all_tokens, actual_duration)

    return success


if __name__ == "__main__":
    audio_path = "/home/samus/programming-projects/whats-the-tab/dataset/John Mayer Neon Live In LA 1080p.mp3"

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    success = quick_verify(audio_path, max_duration=10.0)
    sys.exit(0 if success else 1)
