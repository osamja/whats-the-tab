FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

COPY pyproject.toml ./
COPY uv.lock ./

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv sync --frozen --no-dev --no-install-project --extra gpu

COPY . .

# Bake the checkpoint into the image so worker cold starts do not depend on a
# 175 MB download from GitHub Releases.
RUN uv run python -c "from pytorch_mt3.download_checkpoint import ensure_checkpoint; ensure_checkpoint('pytorch_mt3/mt3_pytorch_checkpoint.pt')"

EXPOSE 8008

CMD ["uv", "run", "python", "manage.py", "runworker"]
