#!/bin/sh
set -e

python manage.py migrate --noinput

(
  while true; do
    python manage.py subscriber
    echo "Subscriber stopped, restarting in 2 seconds..."
    sleep 2
  done
) &

exec gunicorn --bind 0.0.0.0:8008 --workers 2 --threads 2 --worker-class gthread --timeout 300 musictranscription.wsgi
