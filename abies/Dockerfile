FROM python:3.14-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apache2 libapache2-mod-wsgi-py3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Collect static files for production serving.
ENV DJANGO_SETTINGS_MODULE=config.settings
ENV DJANGO_DEBUG=0
ENV DJANGO_SECRET_KEY=placeholder-for-collectstatic
RUN python manage.py collectstatic --noinput

# Reset placeholder — real key comes from environment at runtime.
ENV DJANGO_SECRET_KEY=

# Data directory is mounted from the host.
VOLUME /app/data

EXPOSE 8000

CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "30"]
