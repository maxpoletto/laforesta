FROM python:3.14.10-slim-trixie@sha256:fa99b65193ea9dbde3fc2367f43d3a2907571ce2a15213a86f0053e1d49d8456

# --- OS deps ----------------------------------------------------------------
# sqlite3 ships as a Python stdlib C extension in python:3.14-slim.
# rasterio's wheel needs libexpat at import time.  Terser is present so
# production deploys can minify collected static JS inside the same container
# that runs collectstatic.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 terser \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (cached layer) ---------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
        --require-hashes --only-binary=:all: -r requirements.txt \
 && python -m pip check

# Record the runtime packages, then remove installers and their bundled libraries.
# Base/Python packages are pinned; apt security updates resolve at build time.
RUN mkdir -p /usr/local/share/abies \
 && python -m pip inspect --local \
    | python -c 'import json,sys; report=json.load(sys.stdin); report["installed"]=[item for item in report["installed"] if item["metadata"]["name"].lower() != "pip"]; json.dump(report,sys.stdout,indent=2)' \
    > /usr/local/share/abies/python-dependencies.json \
 && dpkg-query -W > /usr/local/share/abies/system-packages.txt \
 && python -m pip uninstall --yes --root-user-action=ignore pip \
 && python -c 'import ensurepip,pathlib,shutil; shutil.rmtree(pathlib.Path(ensurepip.__file__).parent)'

# --- App source -------------------------------------------------------------
COPY . .

# --- Active language -- materialize template symlinks ---------------------
# Parallels the install-time language selection in config/strings.py and
# apps/base/static/base/js/strings.js.  Kept consistent with the Makefile
# (LANGUAGE_SUFFIX); duplication is intentional because python:3.14-slim
# does not include `make`.
RUN python3 bin/link-templates.py _it

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
VOLUME ["/app/data", "/app/staticfiles", "/app/backup"]

EXPOSE 8000

# No ENTRYPOINT. Exec-form CMD runs directly when no args given (regular
# `compose up` -> gunicorn); user-supplied args (`compose run abies python
# manage.py migrate`) override CMD and run directly. Migrations and
# collectstatic are orchestrated by bin/deploy, not on container start.
# Keep one worker so the in-memory RateLimitMiddleware has process-wide
# semantics. Raising this needs shared rate-limit storage.
CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
