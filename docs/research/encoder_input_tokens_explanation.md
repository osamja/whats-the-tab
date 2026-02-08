# MT3 Encoder Input Tokens - Detailed Explanation

## What are Encoder Input Tokens?

**Shape:** `[batch_size, sequence_length, depth]` (3D tensor)

Unlike typical transformers that use discrete token IDs, **MT3's encoder inputs are continuous-valued spectral features** from audio, not integer tokens!

## The Complete Pipeline: Audio → Encoder Input Tokens

```
Raw Audio (16kHz samples)
    ↓
Spectrogram Computation (ml.py:283, preprocessors.compute_spectrograms)
    ↓
Mel Spectrogram Features [batch, time_frames, frequency_bins]
    ↓
These ARE the encoder_input_tokens!
```

## Input Tensor Dimensions

The input is a **mel-spectrogram** with:
- **X-axis (sequence_length):** Time frames - each frame represents a small window of audio
- **Y-axis (depth):** Frequency bins - mel-scale frequency bands from low to high frequencies
- **Z-axis (batch):** Multiple audio samples processed together

So **yes**, you're correct:
- **Horizontal (X):** Time progression (audio frames)
- **Vertical (Y):** Frequency content (mel-frequency bins)

### Key Characteristics (from network.py:281, 168)

1. **3D Continuous Tensor:** Unlike text transformers that have `[batch, length]` with integer token IDs, these are **already feature vectors**:
   - Dimension 0: Batch size
   - Dimension 1: Time sequence (number of spectrogram frames)
   - Dimension 2: **Depth** (frequency bin features from mel-spectrogram)

2. **Direct Spectrogram Features:** The "tokens" are actually **mel-spectrogram coefficients** computed from audio (ml.py:267, preprocessors.compute_spectrograms)

3. **Each time frame is treated as one "token":** Instead of a single integer ID, each position has a vector of frequency features

## What Happens to Encoder Input Tokens in the Encoder

From `/home/samus/programming-projects/whats-the-tab/src/mt3-transcription/mt3/network.py:174-193`:

```python
# Line 174-179: Project continuous inputs to embedding dimension
x = DenseGeneral(emb_dim)(encoder_input_tokens)
# [batch, length, depth] → [batch, length, 512]

# Line 180: Add positional encoding
x = x + FixedEmbed()(positions)

# Line 181-183: Dropout
x = Dropout()(x)

# Lines 186-190: Process through encoder layers (default: 6 layers)
for each layer:
    x = EncoderLayer()(x, encoder_mask)

# Line 192: Final layer normalization
x = LayerNorm()(x)
```

### Processing Steps

1. **Projection** (line 174): Maps the frequency vector at each time frame to a 512-dimensional embedding
2. **Positional Encoding** (line 180): Adds time position information
3. **Dropout** (line 181): Regularization
4. **Transformer Layers** (line 186-190): Self-attention and feed-forward networks process the sequence
5. **Layer Norm** (line 192): Final normalization

## Key Difference from Text Transformers

| Text Transformer | MT3 Audio Transformer |
|---|---|
| `encoder_input_tokens`: Integer IDs `[batch, seq_len]` | Continuous features `[batch, seq_len, depth]` |
| Needs embedding lookup table | Uses dense projection (network.py:174) |
| Each position = 1 discrete token | Each position = vector of spectrogram features |
| Embeddings learned from scratch | Features come from audio signal processing |

## Debugging Tips

When inspecting `encoder_input_tokens` in your debugger:
- **Shape check:** Should be `(batch, length, depth)` where depth is the number of mel-spectrogram bins
- **Values:** Continuous floats (typically normalized spectrogram amplitudes)
- **After projection (line 174):** They become embeddings of size `emb_dim=512`
- **Visualization:** Think of it as an image where:
  - Horizontal axis = time
  - Vertical axis = frequency
  - Pixel intensity = energy at that time-frequency point

## Best Breakpoints for Layer-by-Layer Debugging

1. **Encoder entry:** `/home/samus/programming-projects/whats-the-tab/src/mt3-transcription/mt3/network.py:395` - Start of encoding
2. **Projection:** `network.py:174` - Continuous input projection to embeddings
3. **Encoder loop:** `network.py:186-191` - Iterate through encoder layers
4. **Per-layer:** Set breakpoint at line **188** to step into each `EncoderLayer`
5. **Attention:** Inside `EncoderLayer` at line **58** for multi-head self-attention
6. **Attention internals:** `/home/samus/programming-projects/whats-the-tab/src/mt3-transcription/mt3/layers.py:238-244` for Q, K, V projections
7. **Core attention math:** `layers.py:134` for the einsum dot-product operation

## Summary

The model treats audio spectrograms as if they're "continuous tokens":
- Each **time frame** in the spectrogram is like one **token position**
- Instead of a single integer ID, it's a **vector of frequency features**
- The transformer processes these continuous features to understand musical structure and transcribe notes
