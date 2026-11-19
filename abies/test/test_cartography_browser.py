"""Opt-in Chromium run of the field-to-office workflow against the test DB."""
import os
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest


@pytest.fixture
def live_server(transactional_db, tmp_path, monkeypatch, request):
    """Run this browser and its assertions on the same private WAL snapshot."""
    from django.db import connections
    from pytest_django.live_server_helper import LiveServer

    original = connections['default']
    original.ensure_connection()
    path = tmp_path / 'browser.sqlite3'
    with closing(sqlite3.connect(path)) as snapshot:
        original.connection.backup(snapshot)
    monkeypatch.setitem(connections.databases, 'default', {**original.settings_dict, 'NAME': str(path)})
    del connections['default']
    try:
        address = request.config.getoption('liveserver') or os.getenv('DJANGO_LIVE_TEST_SERVER_ADDRESS')
        server = LiveServer(address or 'localhost')
        try:
            yield server
        finally:
            server.stop()
    finally:
        connections['default'].close()
        connections['default'] = original


@pytest.mark.skipif(not os.environ.get('ABIES_PLAYWRIGHT_MODULE'), reason='requires Playwright and Chromium')
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('width', [1280, 390])
def test_field_walk_in_browser(live_server, client, writer_user, settings, tmp_path, width):
    from apps.base.models import (
        Eclass, MapDrawing, MapFeature, MapPointSet, ObservationCategory, RoadType, Species,
    )

    Eclass.objects.get_or_create(name='B')
    Eclass.objects.get_or_create(name='A')
    RoadType.objects.get_or_create(name='Pista di prova', defaults={'sort_order': 30})
    category = ObservationCategory.objects.create(name='Controllo di prova')
    species = Species.objects.create(common_name='Pino di prova')
    settings.IPSO_SECRET = 'cartography-browser-test'
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    client.force_login(writer_user)
    result = subprocess.run(['node', str(Path(__file__).with_name('js') / 'cartography-walk.cjs')],
                            env={**os.environ, 'TEST_URL': live_server.url, 'TEST_WIDTH': str(width),
                                 'TEST_CATEGORY': str(category.pk), 'TEST_SPECIES': str(species.pk),
                                 'TEST_SESSION': client.cookies[settings.SESSION_COOKIE_NAME].value},
                            capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    road = MapFeature.objects.get(published=True, name='Strada nord')
    source = MapPointSet.objects.get()
    original = [[row['lon'], row['lat']] for row in source.records if row['segment']]
    first, gate, second, side = MapDrawing.objects.get(source=source).parts
    assert first['coordinates'] + second['coordinates'] + side['coordinates'] == original
    assert gate['fixes'][0][1] == 'Cancello'
    assert len(road.geometry['coordinates']) == len(first['coordinates']) + len(second['coordinates'])
    assert road.geometry['coordinates'][0] != first['coordinates'][0]
