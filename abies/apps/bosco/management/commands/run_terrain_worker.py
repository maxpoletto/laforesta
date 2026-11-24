"""Run the SQLite-backed terrain coordinator in its own process."""
import fcntl
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import close_old_connections
from django.utils import timezone

from apps.base.models import TerrainRefresh
from apps.bosco import satellite_refresh, terrain_queue


class Command(BaseCommand):
    help = 'Process satellite, hillshade and print context refreshes queued in Cartografia or by the schedule.'

    def add_arguments(self, parser):
        parser.add_argument('--once', action='store_true', help='Check the schedule and process one queued job.')

    def handle(self, *args, **options):
        settings.SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
        with (settings.SATELLITE_DIR / '.worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CommandError('A terrain worker is already running for this instance.') from exc
            self.run(once=options['once'])

    def run(self, *, once):
        satellite_refresh.discard_interrupted_staging()
        TerrainRefresh.objects.filter(state=TerrainRefresh.State.RUNNING).update(
            state=TerrainRefresh.State.QUEUED,
        )
        next_check = 0

        def check_schedule():
            nonlocal next_check
            if time.monotonic() < next_check:
                return
            now = timezone.now()
            terrain_queue.schedule(now)
            satellite_refresh.prune(now)
            next_check = time.monotonic() + 60 * 60

        while True:
            close_old_connections()
            check_schedule()
            job = terrain_queue.claim()
            if job:
                satellite_refresh.refresh(job, timezone.now(), check_schedule)
            if once:
                return
            if not job:
                time.sleep(30)
