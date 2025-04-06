#!/bin/bash
set -e

# If the first argument is 'bash' or 'shell', just run bash
if [ "$1" = "bash" ] || [ "$1" = "shell" ]; then
    exec /bin/bash
    exit 0
fi

cd /app/src/mt3-transcription/musictranscription

# Apply database migrations
python manage.py migrate

# Start Redis in the background (if needed locally)
if [ "$START_REDIS" = "true" ]; then
    redis-server --daemonize yes
fi

# Start the dramatiq workers if needed
if [ "$START_DRAMATIQ" = "true" ]; then
    python run_dramatiq.py transcribeapp.tasks &
fi

# Start Django
if [ "$START_SERVER" = "true" ]; then
    # Use gunicorn in production
    if [ "$DJANGO_ENV" = "production" ]; then
        gunicorn --bind 0.0.0.0:8008 musictranscription.wsgi
    else
        python manage.py runserver 0.0.0.0:8008
    fi
fi

# If command is passed, execute it
exec "$@" 