# Reliability

## Web service recovery

Abies runs Gunicorn as the main process in each hosted web container. The
Gunicorn master supervises one HTTP worker and starts a replacement if that
worker exits. This includes a worker killed by the operating system under
memory pressure. The configured 30-second worker timeout also replaces a
worker that stops responding to the master. A request in progress can fail
during the restart.
[Gunicorn supervision](https://gunicorn.org/design/#arbiter)
· [Worker timeout](https://gunicorn.org/reference/settings/#timeout)

If Gunicorn itself exits, the web container stops. The dev and production
Compose services use `restart: unless-stopped`, so Docker restarts them after
an unexpected exit. A deliberately stopped service stays stopped until
started again.
[Docker restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/)

## Terrain refresh recovery

The hosted terrain worker runs in a separate Compose service with the same
restart policy. When it starts, it requeues jobs interrupted while running
and removes their temporary staging directories. Bosco switches to a new
revision only after the worker has completed the files and committed the
new active revision ID. [Satellite publication](satellite-refresh.md#publishing-terrain-data)
