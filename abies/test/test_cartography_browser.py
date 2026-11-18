"""Opt-in Chromium run of the field-to-office workflow against the test DB."""
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get('ABIES_PLAYWRIGHT_MODULE'), reason='requires Playwright and Chromium')
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('width', [1280, 390])
def test_field_walk_in_browser(live_server, client, writer_user, settings, tmp_path, width):
    from apps.base.models import MapDrawing, MapFeature, MapPointSet, RoadType

    RoadType.objects.get_or_create(name='Pista di prova', defaults={'sort_order': 30})
    settings.IPSO_SECRET = 'cartography-browser-test'
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    client.force_login(writer_user)
    result = subprocess.run(['node', str(Path(__file__).with_name('js') / 'cartography-walk.cjs')],
                            env={**os.environ, 'TEST_URL': live_server.url, 'TEST_WIDTH': str(width),
                                 'TEST_SESSION': client.cookies[settings.SESSION_COOKIE_NAME].value},
                            capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    road = MapFeature.objects.get(published=True, name='Strada nord')
    source = MapPointSet.objects.get()
    original = [[row['lon'], row['lat']] for row in source.records if row['segment']]
    assert MapDrawing.objects.get(sequence=1).geometry['coordinates'] == original
    assert MapDrawing.objects.get(sequence=2).fixes[0][1] == 'Cancello'
    assert road.geometry['coordinates'] != original
