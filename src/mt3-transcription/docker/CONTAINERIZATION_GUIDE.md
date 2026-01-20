# PyTorch MT3 Containerization Guide

Complete guide for containerizing the PyTorch MT3 implementation with different dependency levels.

---

## 📦 Three Container Options

### Option 1: Minimal (Tokens Only) - **RECOMMENDED FOR INFERENCE**

**Size:** ~2 GB
**Dependencies:** torch, torchaudio, numpy
**Use Case:** Pure inference, tokens only, no MIDI output

```bash
cd docker
docker build -f Dockerfile.minimal -t pytorch-mt3:minimal ..
docker run --gpus all pytorch-mt3:minimal audio.mp3
```

**What works:**
- ✅ Audio → Tokens
- ✅ GPU acceleration
- ✅ Fast inference
- ❌ No MIDI files

---

### Option 2: MIDI Support - **RECOMMENDED FOR STANDALONE**

**Size:** ~5 GB
**Dependencies:** + librosa, note-seq, tensorflow, MT3 vocab
**Use Case:** Complete Audio → MIDI pipeline, standalone tool

```bash
cd docker
docker build -f Dockerfile.midi -t pytorch-mt3:midi ..
docker run --gpus all pytorch-mt3:midi audio.mp3 output.midi
```

**What works:**
- ✅ Audio → Tokens
- ✅ Tokens → MIDI files
- ✅ GPU acceleration
- ✅ Complete pipeline
- ❌ No API server

---

### Option 3: Full Django API - **RECOMMENDED FOR PRODUCTION**

**Size:** ~6 GB
**Dependencies:** + Django, dramatiq, PostgreSQL, Redis
**Use Case:** Full REST API with async processing

```bash
cd docker
docker-compose up
```

**What works:**
- ✅ Everything from Option 2
- ✅ REST API endpoints
- ✅ Async task processing
- ✅ Database storage
- ✅ YouTube downloads
- ✅ Multi-user support

---

## 🚀 Quick Start by Use Case

### Use Case 1: Simple Inference Service

```dockerfile
# Use minimal container
FROM pytorch-mt3:minimal

# Your custom inference code
COPY inference.py /app/
CMD ["python", "inference.py"]
```

**Dependencies:**
```txt
torch==2.3.0
torchaudio==2.3.0
numpy
```

---

### Use Case 2: MIDI Transcription Service

```dockerfile
# Use MIDI container
FROM pytorch-mt3:midi

# Expose inference endpoint
COPY api.py /app/
CMD ["python", "api.py"]
```

**Dependencies:**
```txt
# See requirements-midi.txt
torch, torchaudio, numpy
librosa, note-seq, mido
tensorflow, seqio, t5
```

---

### Use Case 3: Production API

```yaml
# Use docker-compose.yml
services:
  api:
    build: Dockerfile.django
    ports: ["8000:8000"]
  worker:
    build: Dockerfile.django
    command: dramatiq tasks
  db:
    image: postgres:15
  redis:
    image: redis:7
```

**Dependencies:**
```txt
# See requirements-django.txt
All from requirements-midi.txt
+ Django, dramatiq, gunicorn
+ PostgreSQL, Redis
```

---

## 📋 Dependency Breakdown

### Level 1: Core Inference (Minimal)

```python
# requirements-minimal.txt
torch==2.3.0              # Deep learning framework
torchaudio==2.3.0         # Audio processing
numpy>=1.24.0,<2.0.0     # Numerical operations
```

**Why these versions:**
- PyTorch 2.3.0: Latest stable with CUDA 12.1 support
- NumPy <2.0: Compatibility with other packages
- TorchAudio: Built-in mel spectrogram support

**Container size:** ~2 GB

---

### Level 2: MIDI Support (+3 GB)

```python
# Additional for requirements-midi.txt
librosa>=0.10.0           # Audio loading & analysis
note-seq>=0.0.5          # MIDI file generation
mido>=1.3.0              # MIDI file I/O
tensorflow>=2.12.0       # Required by note-seq
protobuf<4.0.0           # TensorFlow dependency
seqio>=0.0.20            # MT3 vocabulary
t5>=0.9.4                # T5 model base
gin-config>=0.5.0        # Configuration
```

**Why these:**
- `note-seq`: Google's MIDI library, industry standard
- `tensorflow`: Required by note-seq (CPU-only is fine)
- `seqio/t5`: For MT3 vocabulary decoder
- `librosa`: More flexible audio loading than torchaudio

**Container size:** ~5 GB

---

### Level 3: Django API (+1 GB)

```python
# Additional for requirements-django.txt
Django>=4.2.0            # Web framework
djangorestframework      # REST API
dramatiq>=1.15.0         # Task queue
psycopg2-binary          # PostgreSQL
gunicorn>=21.0.0         # WSGI server
pydub>=0.25.0            # Audio manipulation
pytube>=15.0.0           # YouTube downloads
python-dotenv>=1.0.0     # Environment config
```

**Why these:**
- `Django 4.2`: Latest LTS version
- `dramatiq`: Simpler than Celery, Redis-based
- `gunicorn`: Production WSGI server
- `pytube`: YouTube audio extraction

**Container size:** ~6 GB

---

## 🏗️ Build Instructions

### Minimal Container

```bash
# Build
docker build -f docker/Dockerfile.minimal -t pytorch-mt3:minimal .

# Test
docker run --gpus all pytorch-mt3:minimal python -c "
from pytorch_mt3 import MT3Model
print('✓ Model loaded')
"

# Run inference
docker run --gpus all -v $(pwd):/data pytorch-mt3:minimal \
    python pytorch_mt3/standalone_inference.py /data/audio.mp3
```

---

### MIDI Container

```bash
# Build
docker build -f docker/Dockerfile.midi -t pytorch-mt3:midi .

# Test
docker run --gpus all pytorch-mt3:midi python -c "
from pytorch_mt3.standalone_inference import StandaloneMT3
import note_seq
print('✓ MIDI support ready')
"

# Run inference with MIDI output
docker run --gpus all -v $(pwd):/data pytorch-mt3:midi \
    /data/audio.mp3 /data/output.midi
```

---

### Django API (with docker-compose)

```bash
# Build and start all services
cd docker
docker-compose up -d

# Check logs
docker-compose logs -f api

# Run migrations
docker-compose exec api python manage.py migrate

# Create superuser
docker-compose exec api python manage.py createsuperuser

# Test API
curl http://localhost:8000/api/health
```

---

## 🔧 Configuration Options

### Environment Variables

```bash
# Backend selection
USE_PYTORCH=True          # Use PyTorch instead of JAX

# Processing mode
IS_ASYNC=False            # Sync vs async processing

# Database (for Django)
POSTGRES_HOST=db
POSTGRES_DB=transcription
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis (for task queue)
REDIS_URL=redis://redis:6379/0

# Django
DEBUG=False
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=*
```

---

## 📊 Performance Comparison

### Inference Speed (10s audio on RTX 4090)

| Container | Size | Build Time | Inference Time | Memory |
|-----------|------|------------|----------------|--------|
| Minimal | 2 GB | 5 min | 4.9s | 2 GB |
| MIDI | 5 GB | 15 min | 4.9s | 2.5 GB |
| Django | 6 GB | 20 min | 4.9s | 3 GB |

**Note:** Inference time is the same; extra dependencies don't affect speed.

---

## 🎯 Deployment Recommendations

### Development
Use **Minimal** container:
- Fast builds
- Quick iterations
- Tokens only (add MIDI later)

### Staging
Use **MIDI** container:
- Complete pipeline
- Test MIDI output
- No database overhead

### Production
Use **Django** container:
- Full API
- Async processing
- Database persistence
- Scalable

---

## 🐳 Docker Best Practices

### Multi-Stage Builds

```dockerfile
# Stage 1: Build dependencies
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS builder
RUN pip install --user torch torchaudio

# Stage 2: Runtime
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
COPY --from=builder /root/.local /root/.local
COPY pytorch_mt3/ /app/
```

### Volume Mounts

```bash
# Mount data directory
docker run -v $(pwd)/data:/data pytorch-mt3:minimal

# Mount checkpoint separately (reuse across containers)
docker run -v $(pwd)/checkpoints:/checkpoints pytorch-mt3:minimal
```

### GPU Access

```bash
# Single GPU
docker run --gpus all ...

# Specific GPU
docker run --gpus '"device=0"' ...

# Multiple GPUs
docker run --gpus '"device=0,1"' ...
```

---

## 🔍 Debugging

### Check GPU Access

```bash
docker run --gpus all pytorch-mt3:minimal python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Test Model Loading

```bash
docker run --gpus all pytorch-mt3:minimal python -c "
from pytorch_mt3 import MT3Model, MT3Config
import torch
config = MT3Config()
model = MT3Model(config)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
"
```

### Check Dependencies

```bash
# List installed packages
docker run pytorch-mt3:minimal pip list

# Check specific package
docker run pytorch-mt3:minimal python -c "import torch; print(torch.__version__)"
```

---

## 📦 Registry & Distribution

### Push to Registry

```bash
# Tag
docker tag pytorch-mt3:minimal your-registry.com/pytorch-mt3:minimal

# Push
docker push your-registry.com/pytorch-mt3:minimal

# Pull on another machine
docker pull your-registry.com/pytorch-mt3:minimal
```

### Save/Load Images

```bash
# Save
docker save pytorch-mt3:minimal | gzip > pytorch-mt3-minimal.tar.gz

# Transfer to another machine
scp pytorch-mt3-minimal.tar.gz remote:/tmp/

# Load
docker load < pytorch-mt3-minimal.tar.gz
```

---

## 🚨 Common Issues

### Issue 1: CUDA not available
```bash
# Check nvidia-docker
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Issue 2: Out of memory
```bash
# Limit memory
docker run --memory="8g" --gpus all pytorch-mt3:minimal
```

### Issue 3: Slow builds
```bash
# Use BuildKit
DOCKER_BUILDKIT=1 docker build ...

# Cache dependencies
docker build --build-arg BUILDKIT_INLINE_CACHE=1 ...
```

---

## 📝 Summary & Recommendations

### For Your Use Case:

1. **Just need inference?** → Use **Minimal** (2 GB)
   ```bash
   docker build -f docker/Dockerfile.minimal -t mt3:minimal .
   ```

2. **Need MIDI files?** → Use **MIDI** (5 GB)
   ```bash
   docker build -f docker/Dockerfile.midi -t mt3:midi .
   ```

3. **Full production API?** → Use **Django** (6 GB)
   ```bash
   docker-compose up
   ```

### Recommended Stack:

```yaml
# Production deployment
services:
  inference:
    image: mt3:midi          # MIDI container
    deploy:
      replicas: 3            # Scale horizontally
      resources:
        limits:
          nvidia.com/gpu: 1
```

---

## 📚 Additional Resources

- **Dockerfiles:** `docker/Dockerfile.*`
- **Requirements:** `docker/requirements-*.txt`
- **Compose:** `docker/docker-compose.yml`
- **Tests:** Run `docker/test-containers.sh`

---

**Questions?** Check the main `FINAL_PYTORCH_SUMMARY.md` for complete documentation.
