# PyTorch Checkpoint Setup

The PyTorch MT3 model checkpoint (`mt3_pytorch_checkpoint.pt`, 175 MB) is hosted on GitHub Releases to keep the repository size manageable.

## Automatic Download

The checkpoint will be **automatically downloaded** when you run inference for the first time. No manual setup required!

```python
from pytorch_mt3.standalone_inference import StandaloneMT3

# This will auto-download the checkpoint if not present
model = StandaloneMT3('pytorch_mt3/mt3_pytorch_checkpoint.pt')
```

## Manual Download

If you prefer to download manually:

```bash
# Option 1: Use the download script
python src/mt3-transcription/pytorch_mt3/download_checkpoint.py

# Option 2: Download directly
wget https://github.com/osamja/whats-the-tab/releases/download/v1.0-pytorch-checkpoint/mt3_pytorch_checkpoint.pt \
     -O src/mt3-transcription/pytorch_mt3/mt3_pytorch_checkpoint.pt
```

## For Contributors

If you need to upload a new checkpoint version:

1. Run the upload instructions script:
   ```bash
   ./scripts/upload_checkpoint_to_release.sh
   ```

2. Follow the instructions to create a new GitHub Release

3. Upload the checkpoint file to the release

## Docker Setup

The Dockerfile automatically downloads the checkpoint during build:

```dockerfile
# Download checkpoint in a cached layer
RUN python pytorch_mt3/download_checkpoint.py

# Copy application code
COPY . /app
```

This approach ensures:
- Fast Docker builds (checkpoint layer is cached)
- No need for Git LFS
- Clean repository (no large binary files)
- Easy checkpoint updates (just create new release)

## Checkpoint Details

- **File**: `mt3_pytorch_checkpoint.pt`
- **Size**: ~175 MB
- **Format**: PyTorch state_dict
- **Download URL**: https://github.com/osamja/whats-the-tab/releases/download/v1.0-pytorch-checkpoint/mt3_pytorch_checkpoint.pt

## Troubleshooting

### Download fails

If automatic download fails, you'll see detailed error instructions. Download manually:

1. Visit https://github.com/osamja/whats-the-tab/releases
2. Download `mt3_pytorch_checkpoint.pt`
3. Place it at `src/mt3-transcription/pytorch_mt3/mt3_pytorch_checkpoint.pt`

### Checkpoint appears corrupted

Delete and re-download:

```bash
rm src/mt3-transcription/pytorch_mt3/mt3_pytorch_checkpoint.pt
python src/mt3-transcription/pytorch_mt3/download_checkpoint.py
```
