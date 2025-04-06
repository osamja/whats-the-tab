# Docker Setup for MT3 Transcription Application

This document provides instructions for running the MT3 transcription application using Docker.

## Prerequisites
- Docker
- Docker Compose

## Directory Structure
Before starting, ensure you create the following directories for persistent data:
```mkdir -p data/media data/static
```

## CPU vs GPU Configuration
The application can be configured to use either CPU or GPU for transcription:

- **CPU Mode**: Default setting, works on all machines but is slower for transcription
- **GPU Mode**: Requires CUDA-compatible GPU, significantly faster for transcription

### Using the Override File (Recommended)
Copy the example override file and customize it for your environment:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

Edit `docker-compose.override.yml` to:
1. Set `USE_GPU=true` if you have a GPU
2. Uncomment the GPU device settings if you're using the NVIDIA Container Runtime

### Directly Editing docker-compose.yml
Alternatively, you can directly edit the `docker-compose.yml` file and set the `USE_GPU` build argument to `true`:
```yaml
build:
  context: .
  dockerfile: Dockerfile
  args:
    - USE_GPU=true  # Changed from false to true
```

## Running the Application

### Build and Start the Containers
```bash
docker-compose up -d --build
```

This will start:
- A Redis service for message queuing
- The Django web application
- A worker for processing transcription tasks

### View Logs
To view the logs:
```bash
docker-compose logs -f
```

### Access the Application
The application will be available at:
```
http://localhost:8008
```

## Stopping the Application
```bash
docker-compose down
```

## Troubleshooting

### Missing Dependencies
If you encounter missing library errors, update the Dockerfile to install additional system packages.

### Model Not Loading
Ensure the model checkpoints are in the correct location. The default location is `/app/src/mt3-transcription/musictranscription/checkpoints`.

### GPU Issues
If you encounter GPU-related errors:
1. Make sure the host machine has the required CUDA libraries installed
2. Verify your GPU is CUDA-compatible
3. Check that the NVIDIA Container Runtime is properly configured
4. If issues persist, revert to CPU mode by setting `USE_GPU=false` in your configuration

### Redis Connection Issues
Check if Redis is running correctly:
```bash
docker-compose exec redis redis-cli ping
```
Should return "PONG"

## Additional Configuration

### Environment Variables
You can modify the services in the `docker-compose.yml` file to change:
- `DJANGO_ENV`: Set to "production" for production environments
- `START_SERVER`: Whether to start the Django server
- `START_REDIS`: Whether to start Redis inside the Django container (not needed if using the Redis service)
- `START_DRAMATIQ`: Whether to start Dramatiq workers inside the Django container

### Production Deployment
For production, it's recommended to use:
- Gunicorn instead of the Django development server
- Separate containers for different services
- External Redis service
- Proper secret management
- HTTPS configuration 