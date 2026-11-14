#!/bin/sh
set -e

# Run migrations here for simple/single-instance deployments. Set
# RUN_MIGRATIONS=False when migrations are handled by a release job.
if [ "${RUN_MIGRATIONS:-True}" = "True" ]; then
  python manage.py migrate --noinput
fi

(
  while true; do
    python manage.py subscriber
    echo "Subscriber stopped, restarting in 2 seconds..."
    sleep 2
  done
) &

exec gunicorn --bind 0.0.0.0:8008 --workers 2 --threads 2 --worker-class gthread --timeout 300 musictranscription.wsgi
