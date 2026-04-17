#!/bin/sh
# Runs at container start. Applies any pending schema migrations and
# refreshes collected static files, then hands control to gunicorn.
#
# Safe to re-run: both `migrate` and `collectstatic` are idempotent.

set -eu

python manage.py migrate --noinput
python manage.py collectstatic --noinput

exec "$@"
