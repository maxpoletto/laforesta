FROM python:3.14-slim

# --- OS deps ----------------------------------------------------------------
# Only runtime libs needed: sqlite3 ships as a Python stdlib C extension in
# python:3.14-slim, so no apt packages are strictly required.  Keeping the
# apt-get stanza minimal leaves a clean slate for future additions.

WORKDIR /app

# --- Python deps (cached layer) ---------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App source -------------------------------------------------------------
COPY . .

# --- Non-root user ----------------------------------------------------------
# Fixed UID so bind-mounted host dirs can be chowned to match.  Host must
# ensure /var/lib/abies/{data,static} are writable by UID 10001.
RUN groupadd --gid 10001 abies \
 && useradd  --uid 10001 --gid 10001 --home-dir /app --no-create-home abies \
 && chown -R abies:abies /app

USER abies

ENV DJANGO_SETTINGS_MODULE=config.settings \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Bind-mounted from host at runtime.
VOLUME ["/app/data", "/app/staticfiles"]

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
