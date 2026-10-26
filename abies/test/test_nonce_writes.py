"""Retry receipts and domain writes must succeed or roll back together."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from django.db import DatabaseError, connection, connections
from django.http import JsonResponse
from django.test import Client, RequestFactory

from apps.base import responses
from apps.base.middleware import NonceMiddleware
from apps.base.models import (
    Crew, DigestStatus, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    Observation, ObservationCategory, ObservationPhoto, Tree, TreeMark, UsedNonce,
)
from config.constants import FIELD_NONCE


pytestmark = pytest.mark.django_db


@pytest.fixture
def writer_client(writer_user):
    client = Client(raise_request_exception=False)
    client.force_login(writer_user)
    return client


@pytest.fixture
def mark_body(parcels, species):
    plan = HarvestPlan.objects.create(name='Nonce test', year_start=2026, year_end=2027)
    item = HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2026,
    )
    return {
        FIELD_NONCE: 'atomic-mark', 'harvest_plan_item_id': item.id,
        'species_id': species[0].id, 'date': '2026-10-15',
        'd_cm': 30, 'h_m': '20', 'operator': 'Test',
    }


def post_json(client, url, body):
    return client.post(url, json.dumps(body), content_type='application/json')


def fail_receipt(*args, **kwargs):
    raise RuntimeError('Injected receipt failure')


def test_receipt_failure_rolls_back_mark_totals_history_and_dirty_flags(
        writer_client, mark_body):
    item = HarvestPlanItem.objects.get(pk=mark_body['harvest_plan_item_id'])
    history_count = item.history.count()
    digest_state = list(DigestStatus.objects.order_by('pk').values())
    with patch.object(responses, 'save_nonce', side_effect=fail_receipt):
        failed = post_json(writer_client, '/api/piano-di-taglio/mark/save/', mark_body)

    assert failed.status_code == 500
    assert not TreeMark.objects.exists()
    assert not Tree.objects.exists()
    assert not UsedNonce.objects.exists()
    assert list(DigestStatus.objects.order_by('pk').values()) == digest_state
    item.refresh_from_db()
    assert item.state == HarvestPlanItemState.PLANNED
    assert item.version == 1
    assert item.volume_marked_m3 is None
    assert item.history.count() == history_count

    retry = post_json(writer_client, '/api/piano-di-taglio/mark/save/', mark_body)
    assert retry.status_code == 200
    assert TreeMark.objects.count() == Tree.objects.count() == UsedNonce.objects.count() == 1


@pytest.mark.parametrize('failure', ['receipt', 'patch'])
def test_shared_save_rolls_back_when_building_or_recording_response_fails(
        writer_client, failure):
    target = 'save_nonce' if failure == 'receipt' else 'row_patch'
    with patch.object(responses, target, side_effect=fail_receipt):
        result = post_json(writer_client, '/api/squadre/crews/save/', {
            FIELD_NONCE: 'atomic-crew', 'name': 'Atomic crew', 'active': True,
        })

    assert result.status_code == 500
    assert not Crew.objects.exists()
    assert not UsedNonce.objects.exists()


def test_failure_after_receipt_is_saved_also_rolls_back(writer_client):
    original = responses.save_nonce

    def save_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('Failure after recording receipt')

    with patch.object(responses, 'save_nonce', side_effect=save_then_fail):
        result = post_json(writer_client, '/api/squadre/crews/save/', {
            FIELD_NONCE: 'receipt-written', 'name': 'Must not survive',
        })
    assert result.status_code == 500
    assert not Crew.objects.exists()
    assert not UsedNonce.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_failed_commit_does_not_leave_data_or_receipt(writer_client):
    with patch.object(connection, 'commit', side_effect=DatabaseError('Injected commit failure')):
        result = post_json(writer_client, '/api/squadre/crews/save/', {
            FIELD_NONCE: 'commit-failed', 'name': 'Must not commit',
        })
    assert result.status_code == 500
    assert not connection.in_atomic_block
    assert not Crew.objects.exists()
    assert not UsedNonce.objects.exists()


@pytest.mark.parametrize('nonce', [42, ['nonce'], {}, ' ', 'x' * 65])
def test_invalid_nonce_is_rejected_before_writing(writer_client, nonce):
    result = post_json(writer_client, '/api/squadre/crews/save/', {
        FIELD_NONCE: nonce, 'name': 'Invalid nonce',
    })
    assert result.status_code == 400
    assert not Crew.objects.exists()
    assert not UsedNonce.objects.exists()


def test_multipart_receipt_failure_rolls_back_observation_and_photo_rows(
        writer_client, regions, settings, tmp_path):
    from django.core.files.uploadedfile import SimpleUploadedFile
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    category = ObservationCategory.objects.create(name='Atomic observation')
    body = {
        FIELD_NONCE: 'atomic-observation', 'region_id': regions[0].id,
        'date': '2026-10-15', 'text': 'Atomic observation',
        'lat': '38.1', 'lon': '16.1', 'category_ids': [category.id],
    }

    from test.image_helpers import image_bytes

    def submit(content=None):
        return writer_client.post('/api/bosco/observations/save/', {
            **body, 'photos': SimpleUploadedFile(
                'photo.jpg', image_bytes() if content is None else content, 'image/jpeg'),
        })

    with patch.object(responses, 'save_nonce', side_effect=fail_receipt):
        failed = submit()
    assert failed.status_code == 500
    assert not Observation.objects.exists()
    assert not ObservationPhoto.objects.exists()
    assert not UsedNonce.objects.exists()

    retry = submit()
    assert retry.status_code == 200
    assert Observation.objects.count() == ObservationPhoto.objects.count() == 1


@pytest.mark.parametrize('changed', ['payload', 'endpoint'])
def test_successful_nonce_cannot_be_reused_for_another_request(
        writer_client, mark_body, changed):
    first = post_json(writer_client, '/api/piano-di-taglio/mark/save/', mark_body)
    assert first.status_code == 200
    if changed == 'payload':
        result = post_json(writer_client, '/api/piano-di-taglio/mark/save/', {
            **mark_body, 'd_cm': 40,
        })
    else:
        result = post_json(writer_client, '/api/squadre/crews/save/', {
            FIELD_NONCE: mark_body[FIELD_NONCE], 'name': 'Other operation',
        })

    assert result.status_code == 409
    assert TreeMark.objects.count() == 1
    assert not Crew.objects.exists()


def test_json_key_order_and_whitespace_do_not_change_request_identity(
        writer_client, mark_body):
    first = post_json(writer_client, '/api/piano-di-taglio/mark/save/', mark_body)
    second = writer_client.post('/api/piano-di-taglio/mark/save/',
        json.dumps(mark_body, indent=2, sort_keys=True), content_type='application/json')

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert TreeMark.objects.count() == 1


def test_rejected_request_does_not_consume_nonce(writer_client, mark_body):
    bad = post_json(writer_client, '/api/piano-di-taglio/mark/save/', {
        **mark_body, 'species_id': None,
    })
    assert bad.status_code == 400
    assert not UsedNonce.objects.exists()
    corrected = post_json(writer_client, '/api/piano-di-taglio/mark/save/', mark_body)
    assert corrected.status_code == 200
    assert TreeMark.objects.count() == 1


def test_changed_photo_is_not_silently_replayed(writer_client, regions, settings, tmp_path):
    from django.core.files.uploadedfile import SimpleUploadedFile
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    category = ObservationCategory.objects.create(name='Photo identity')
    body = {
        FIELD_NONCE: 'photo-identity', 'region_id': regions[0].id,
        'date': '2026-10-15', 'text': 'Photo identity',
        'lat': '38.1', 'lon': '16.1', 'category_ids': [category.id],
    }

    def submit(content):
        return writer_client.post('/api/bosco/observations/save/', {
            **body, 'photos': SimpleUploadedFile('photo.jpg', content, 'image/jpeg'),
        })

    from test.image_helpers import image_bytes
    assert submit(image_bytes(color='green')).status_code == 200
    assert submit(image_bytes(color='blue')).status_code == 409
    assert Observation.objects.count() == ObservationPhoto.objects.count() == 1


def test_error_response_rolls_back_writes_and_claim(admin_user):
    from apps.base.idempotency import begin_nonce_write
    request = RequestFactory().post('/api/test/', json.dumps({FIELD_NONCE: 'rejected'}),
                                    content_type='application/json')
    request.user = admin_user

    def view(request):
        begin_nonce_write(request)
        Crew.objects.create(name='Must roll back')
        return JsonResponse({'message': 'Rejected'}, status=400)

    assert NonceMiddleware(view)(request).status_code == 400
    assert not Crew.objects.exists()
    assert not UsedNonce.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_missing_write_boundary_fails_before_mutating_data(admin_user):
    request = RequestFactory().post('/api/test/', json.dumps({FIELD_NONCE: 'unguarded'}),
                                    content_type='application/json')
    request.user = admin_user

    def view(request):
        Crew.objects.create(name='Unguarded')
        return responses.success_response(request, {FIELD_NONCE: 'unguarded'})

    with pytest.raises(RuntimeError, match='begin_nonce_write'):
        NonceMiddleware(view)(request)
    assert not Crew.objects.exists()
    assert not UsedNonce.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_parsing_runs_outside_write_transaction(admin_user):
    from apps.base.idempotency import begin_nonce_write
    request = RequestFactory().post('/api/test/', json.dumps({FIELD_NONCE: 'deferred'}),
                                    content_type='application/json')
    request.user = admin_user

    def view(request):
        assert not connection.in_atomic_block
        begin_nonce_write(request)
        assert connection.in_atomic_block
        Crew.objects.create(name='Deferred')
        return responses.success_response(request, {FIELD_NONCE: 'deferred'})

    assert NonceMiddleware(view)(request).status_code == 200
    assert not connection.in_atomic_block
    assert Crew.objects.count() == UsedNonce.objects.count() == 1


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('first_fails', [False, True])
def test_concurrent_identical_requests_create_only_one_mark(
        writer_user, mark_body, tmp_path, monkeypatch, first_fails):
    clients = [Client(raise_request_exception=False) for _ in range(2)]
    for client in clients:
        client.force_login(writer_user)
    database = tmp_path / 'nonce-concurrency.sqlite3'
    connection.ensure_connection()
    with sqlite3.connect(database) as snapshot:
        connection.connection.backup(snapshot)
    # Only new thread-local connections use the file-backed test snapshot.
    # The main thread retains its normal pytest database/teardown connection.
    monkeypatch.setitem(connections.databases, 'default', {
        **connection.settings_dict, 'NAME': str(database),
    })
    first_at_receipt = threading.Event()
    release_first = threading.Event()
    second_at_write = threading.Event()
    original = responses.save_nonce

    def save_receipt(*args, **kwargs):
        if threading.current_thread().name.startswith('first'):
            first_at_receipt.set()
            assert release_first.wait(10)
            if first_fails:
                raise RuntimeError('First request fails before its receipt')
        return original(*args, **kwargs)

    def submit(index):
        def observe_write(execute, sql, params, many, context):
            if index == 1 and sql.lstrip().upper().startswith('INSERT'):
                second_at_write.set()
            return execute(sql, params, many, context)
        try:
            with connections['default'].execute_wrapper(observe_write):
                return post_json(clients[index], '/api/piano-di-taglio/mark/save/', mark_body)
        finally:
            connections.close_all()

    with patch.object(responses, 'save_nonce', side_effect=save_receipt):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='first') as first_pool, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix='second') as second_pool:
            first = first_pool.submit(submit, 0)
            try:
                assert first_at_receipt.wait(10)
                second = second_pool.submit(submit, 1)
                assert second_at_write.wait(10)
            finally:
                release_first.set()
            first_response, second_response = first.result(10), second.result(10)

    assert first_response.status_code == (500 if first_fails else 200)
    assert second_response.status_code == 200
    if not first_fails:
        assert first_response.json() == second_response.json()
    with sqlite3.connect(database) as snapshot:
        assert snapshot.execute('SELECT COUNT(*) FROM base_treemark').fetchone()[0] == 1
        assert snapshot.execute('SELECT COUNT(*) FROM base_usednonce').fetchone()[0] == 1
