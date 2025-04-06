FROM python:3.10-slim

# Build argument to determine if we should use GPU or CPU
ARG USE_GPU=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    software-properties-common \
    libsndfile1 \
    fluidsynth \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the project
COPY . /app/

# Copy requirements and install dependencies
COPY requirements-cpu.txt requirements-gpu.txt /app/

# Install Python dependencies directly (no virtual env)
RUN pip install --upgrade pip

# Install either CPU or GPU dependencies based on build arg
RUN if [ "$USE_GPU" = "true" ] ; then \
        pip install -r /app/requirements-gpu.txt ; \
    else \
        pip install -r /app/requirements-cpu.txt ; \
    fi

# Install additional required Django packages explicitly
RUN pip install djangorestframework dj-rest-auth django-allauth

# Set up MT3
WORKDIR /app/src/mt3-transcription
RUN pip install -e .

# Set environment variables
ENV DJANGO_SETTINGS_MODULE=musictranscription.settings
ENV PYTHONPATH=/app

# Expose port for Django
EXPOSE 8008

# Create directory for redis socket
RUN mkdir -p /var/run/redis

# Copy entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set the entrypoint
# To debug using bash, override the command when running docker-compose:
# docker-compose run --rm django bash
ENTRYPOINT ["/app/docker-entrypoint.sh"] 