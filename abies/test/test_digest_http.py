"""HTTP validators must identify the actual published digest, not its second."""

import gzip
import hashlib
import json
import os

import pytest
from django.test import Client

from apps.base import digests
from apps.base.models import DigestStatus, HarvestPlan

pytestmark = pytest.mark.django_db
URL = '/api/piano-di-taglio/plans/data/'
FROZEN_MTIME = 1700000000.125


@pytest.fixture
def published(reader_user, settings):
    plan = HarvestPlan.objects.create(name='Old plan', year_start=2026, year_end=2027)
    digests.generate_harvest_plans()
    path = settings.DIGEST_DIR / 'harvest_plans.json.gz'
    os.utime(path, (FROZEN_MTIME, FROZEN_MTIME))
    DigestStatus.objects.update_or_create(name='harvest_plans', defaults={'stale': False})
    client = Client()
    client.force_login(reader_user)
    return client, plan, path


def test_another_reader_regenerating_in_same_second_cannot_preserve_old_data(published):
    client, plan, path = published
    first = client.get(URL)
    old_time, old_etag = first['Last-Modified'], first.get('ETag')
    first.close()

    plan.name = 'New plan'
    plan.save()
    digests.mark_stale('harvest_plans')
    # A different reader consumes the stale flag and publishes the new file.
    client.get(URL).close()
    os.utime(path, (FROZEN_MTIME, FROZEN_MTIME))
    headers = {'HTTP_IF_MODIFIED_SINCE': old_time}
    if old_etag:
        headers['HTTP_IF_NONE_MATCH'] = old_etag
    response = client.get(URL, **headers)

    assert response.status_code == 200
    assert response['Last-Modified'] == old_time
    assert 'New plan' in json.loads(gzip.decompress(response.getvalue()))['rows'][0]
    assert response['ETag'] != old_etag


@pytest.mark.parametrize('validator', ['exact', 'weak', 'list', 'wildcard'])
def test_unchanged_digest_revalidates_with_etag(published, validator):
    client, _, path = published
    first = client.get(URL)
    tag = first.get('ETag')
    first.close()
    assert tag == f'"{hashlib.sha256(path.read_bytes()).hexdigest()}"'
    value = {
        'exact': tag, 'weak': f'W/{tag}',
        'list': f'"another-file", W/{tag}', 'wildcard': '*',
    }[validator]

    response = client.get(URL, HTTP_IF_NONE_MATCH=value)
    assert response.status_code == 304
    assert response['ETag'] == tag
    assert response['Last-Modified'] == first['Last-Modified']
    assert response['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('etag', [None, '"old-file"', 'malformed'])
def test_dates_alone_cannot_validate_a_digest(published, etag):
    client, _, _ = published
    headers = {'HTTP_IF_MODIFIED_SINCE': 'Fri, 01 Jan 2100 00:00:00 GMT'}
    if etag is not None:
        headers['HTTP_IF_NONE_MATCH'] = etag
    response = client.get(URL, **headers)
    assert response.status_code == 200
    assert response.get('ETag')
    response.close()


def test_etag_tracks_file_content_even_without_a_dirty_sequence_change(published):
    client, plan, path = published
    first = client.get(URL)
    old_tag = first.get('ETag')
    first.close()
    old_seq = DigestStatus.objects.get(name='harvest_plans').dirty_seq
    # A management-command-style regeneration replaces the published file.
    plan.name = 'Rebuilt plan'
    plan.save()
    digests.generate_harvest_plans()
    os.utime(path, (FROZEN_MTIME, FROZEN_MTIME))
    assert DigestStatus.objects.get(name='harvest_plans').dirty_seq == old_seq
    response = client.get(URL, HTTP_IF_NONE_MATCH=old_tag or '"missing-tag"')
    assert response.status_code == 200
    assert response.get('ETag') != old_tag
    response.close()


def test_etag_and_body_use_the_same_open_file(published, monkeypatch):
    from apps.base import http

    client, _, path = published
    expected_tag = f'"{hashlib.sha256(path.read_bytes()).hexdigest()}"'
    original = http._content_etag

    def replace_after_hashing(file, stat):
        tag = original(file, stat)
        digests._write_gzip_json({'rows': [['Replacement']]}, path)
        return tag

    monkeypatch.setattr(http, '_content_etag', replace_after_hashing)
    response = client.get(URL)
    assert response['ETag'] == expected_tag
    assert 'Old plan' in json.loads(gzip.decompress(response.getvalue()))['rows'][0]
    assert json.loads(gzip.decompress(path.read_bytes()))['rows'] == [['Replacement']]
