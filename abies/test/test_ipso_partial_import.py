"""Tree import review preserves rejected rows and consumes each session once."""
import base64
import json
from decimal import Decimal
from pathlib import Path

import pytest
from django.test import Client
from django.urls import reverse

from apps.base import csv_io
from apps.base.models import HarvestPlanItemState, Sample, Survey, TreeMark, TreeSample, UsedNonce
from apps.campionamenti import csv_trees
from apps.ipso.models import IpsoUploadState
from config import strings as S
from config.constants import FIELD_H_MEASURED, FIELD_NONCE, RECORDS, ROW_ID
from test.test_ipso_views import (
    _harvest_item, _sample_survey, _stage_upload_direct, _upload_payload,
)


@pytest.fixture(params=['martellate', 'samples', 'free_survey'])
def partial_import(request, writer_user, parcels, species, settings, tmp_path):
    client = Client()
    client.force_login(writer_user)
    mode = request.param
    if mode == 'martellate':
        target = _harvest_item(parcels)
        body = {'harvest_plan_item_id': target.pk}
        route = 'ipso-upload-import-martellate'
        fields = {}
        model = TreeMark
    elif mode == 'samples':
        target, area = _sample_survey(parcels[0])
        body = {'survey_id': target.pk}
        route = 'ipso-upload-import-samples'
        fields = {'sample_area_id': area.pk}
        model = TreeSample
    else:
        target = Survey.objects.create(name='Partial free survey')
        body = {'survey_id': target.pk}
        route = 'ipso-upload-import-free-survey'
        fields = {}
        model = TreeSample
    payload = _upload_payload(parcels, species, mode=mode, record_overrides={
        **fields, 'client_record_id': '28', 'number': 25, 'h_measured': True,
    })
    payload[RECORDS].append({
        **payload[RECORDS][0], 'client_record_id': '30', 'number': 26, 'd_cm': 32,
        'h_m': None, 'h_measured': False,
    })
    upload = _stage_upload_direct(settings, tmp_path, payload)
    url = reverse(route, args=[upload.pk])
    return client, upload, target, body, url, model


def _post(client, url, body):
    return client.post(url, json.dumps(body), content_type='application/json')


def _review(client, url, body):
    response = _post(client, url, body)
    assert response.status_code == 409, response.content
    data = response.json()
    assert data['status'] == 'import_review'
    assert data['valid_count'] == data['excluded_count'] == 1
    assert len(data['errors']) == 1
    assert 'Riga 30' in data['errors'][0] and 'altezza' in data['errors'][0].lower()
    assert 'D/H/data' not in data['errors'][0]
    return data


def test_partial_import_requires_export_and_retains_recovery_csv(partial_import, writer_user):
    client, upload, target, body, url, model = partial_import
    body['nonce'] = 'partial-review'
    review = _review(client, url, body)
    assert not model.objects.exists()
    assert not UsedNonce.objects.filter(user=writer_user, nonce=body['nonce']).exists()
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    _review(client, url, {**body, 'review_token': review['review_token']})
    assert not model.objects.exists()

    export_url = reverse('ipso-upload-excluded-export', args=[upload.pk])
    exported = _post(client, export_url, {'review_token': review['review_token']})
    assert exported.status_code == 200, exported.content
    assert exported['Cache-Control'] == 'no-store'
    receipt = exported['X-Import-Export-Token']
    rejected = list(csv_io.read(exported.content))
    assert len(rejected) == 1
    assert rejected[0][S.CSV_COL_IPSO_ROW] == '30'
    assert rejected[0][S.CSV_COL_H_M] == ''
    assert rejected[0][S.CSV_COL_D_CM] == '32'
    assert 'altezza' in rejected[0][S.CSV_COL_IMPORT_ERRORS].lower()

    confirmed = {**body, 'review_token': review['review_token'], 'export_token': receipt,
                 'warnings_confirmed': True}
    response = _post(client, url, confirmed)
    assert response.status_code == 200, response.content
    assert response.json()['imported'] == response.json()['excluded_count'] == 1
    assert model.objects.count() == 1
    row = model.objects.get()
    assert row.number == 25 and row.h_m == Decimal('22')
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED
    assert upload.import_result['excluded'][0]['record_id'] == '30'
    assert upload.import_result['imported'] == 1
    assert _post(client, url, confirmed).json() == response.json()
    assert model.objects.count() == 1
    assert _post(client, url, {**body, 'nonce': 'another-attempt'}).status_code == 400
    summary = upload.error_summary
    history_count = upload.history.count()
    assert _post(client, url, {key: value for key, value in body.items() if key != 'nonce'}).status_code == 400
    upload.refresh_from_db()
    assert upload.error_summary == summary
    assert upload.history.count() == history_count
    assert client.get(export_url).content == exported.content
    detail = client.get(reverse('ipso-upload-detail', args=[upload.pk])).json()
    assert detail['upload']['state_label'] == S.IPSO_STATE_PARTIAL
    assert detail['upload']['import_result']['excluded_count'] == 1

    # The recovery CSV uses the destination's normal import format. Correcting
    # the missing height is sufficient to validate it through the real importer.
    reader = csv_io.read(exported.content)
    delimiter, _ = csv_io.export_format()
    buffer, writer = csv_io.csv_buffer(delimiter)
    writer.writerow(reader.fieldnames)
    fixed = list(reader)[0]
    fixed[S.CSV_COL_H_M] = '20'
    writer.writerow([fixed[name] for name in reader.fieldnames])
    if upload.mode != 'martellate':
        parsed, errors = csv_trees.validate_rows(csv_io.read(buffer.getvalue()), csv_trees.db_indexes(target), has_date_column=True, default_date=None)
        assert not errors, errors
        assert len(parsed) == 1 and parsed[0]['number'] == 26
    else:
        response = _post(client, '/api/piano-di-taglio/mark/import-csv/', {
            'harvest_plan_item_id': target.pk,
            'file': base64.b64encode(buffer.getvalue().encode()).decode(), 'nonce': 'csv-recovery',
        })
        assert response.status_code == 200, response.content
        assert model.objects.count() == 2


def test_changed_destination_requires_new_review_and_export(partial_import, writer_user):
    client, upload, target, body, url, model = partial_import
    review = _review(client, url, body)
    export = _post(client, reverse('ipso-upload-excluded-export', args=[upload.pk]), {
        'review_token': review['review_token'],
    })
    assert export.status_code == 200
    # A context edit after review is independent of the immutable uploaded blob.
    target.version += 1
    target.save(update_fields=['version'])
    response = _post(client, url, {
        **body, 'review_token': review['review_token'],
        'export_token': export['X-Import-Export-Token'], 'nonce': 'changed-target',
    })
    assert response.status_code == 409, response.content
    assert response.json()['status'] == 'import_review'
    assert response.json()['review_token'] != review['review_token']
    assert not model.objects.exists()
    assert not UsedNonce.objects.filter(user=writer_user, nonce='changed-target').exists()


def test_export_receipt_is_bound_to_user_and_review(partial_import, reader_user):
    client, upload, _, body, url, model = partial_import
    review = _review(client, url, body)
    exported_url = reverse('ipso-upload-excluded-export', args=[upload.pk])
    reader = Client()
    reader.force_login(reader_user)
    assert _post(reader, exported_url, {'review_token': review['review_token']}).status_code == 403
    assert _post(client, exported_url, {'review_token': 'tampered'}).status_code == 400
    response = _post(client, url, {**body, 'review_token': review['review_token'], 'export_token': 'tampered'})
    assert response.status_code == 409
    assert not model.objects.exists()


def test_partial_import_collects_measurement_problems_on_the_correct_row(partial_import, settings, tmp_path):
    client, upload, _, body, url, model = partial_import
    payload = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    payload[RECORDS][1].update(date='bad-date', d_cm=0, h_m=None)
    upload.delete()
    replacement = _stage_upload_direct(settings, tmp_path, payload)
    mode_route = {'martellate': 'ipso-upload-import-martellate',
                  'samples': 'ipso-upload-import-samples', 'free_survey': 'ipso-upload-import-free-survey'}
    url = reverse(mode_route[replacement.mode], args=[replacement.pk])
    review = _review(client, url, body)
    line = review['errors'][0].lower()
    assert all(word in line for word in ('riga 30', 'data', 'diametro', 'altezza'))
    assert not model.objects.exists()


def test_signed_review_rejects_malformed_token_types(partial_import):
    client, upload, _, body, url, model = partial_import
    review = _review(client, url, body)
    for malformed in ([1], {'bad': 'token'}, 42):
        response = _post(client, url, {**body, 'review_token': malformed})
        assert response.status_code == 409, response.content
        response = _post(client, reverse('ipso-upload-excluded-export', args=[upload.pk]), {
            'review_token': malformed,
        })
        assert response.status_code == 400, response.content
    assert not model.objects.exists()


def test_conflicting_group_is_excluded_without_arbitrary_winner(partial_import, settings, tmp_path):
    client, upload, _, body, _, model = partial_import
    payload = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    payload[RECORDS][1].update(number=25, h_m='18', h_measured=True)
    upload.delete()
    upload = _stage_upload_direct(settings, tmp_path, payload)
    route = {'martellate': 'ipso-upload-import-martellate',
             'samples': 'ipso-upload-import-samples', 'free_survey': 'ipso-upload-import-free-survey'}
    response = _post(client, reverse(route[upload.mode], args=[upload.pk]), body)
    assert response.status_code == 409, response.content
    review = response.json()
    assert review['valid_count'] == 0 and review['excluded_count'] == 2
    assert len(review['errors']) == 2
    assert not model.objects.exists()


def test_partial_import_rolls_back_if_outcome_response_fails(partial_import, monkeypatch, writer_user):
    from apps.ipso import views
    client, upload, _, body, url, model = partial_import
    review = _review(client, url, body)
    exported = _post(client, reverse('ipso-upload-excluded-export', args=[upload.pk]), {
        'review_token': review['review_token'],
    })
    def fail(*args, **kwargs):
        raise RuntimeError('response failure')
    monkeypatch.setattr(views, 'success_response', fail)
    with pytest.raises(RuntimeError, match='response failure'):
        _post(client, url, {**body, 'review_token': review['review_token'],
                           'export_token': exported['X-Import-Export-Token'],
                           'nonce': 'rollback-partial', 'warnings_confirmed': True})
    assert not model.objects.exists()
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.import_result == {}
    assert not UsedNonce.objects.filter(user=writer_user, nonce='rollback-partial').exists()


def test_recovery_of_unnumbered_free_row_preserves_blank_then_assigns_csv_number(
        writer_user, parcels, species, settings, tmp_path):
    survey = Survey.objects.create(name='Unnumbered recovery')
    payload = _upload_payload(parcels, species, mode='free_survey', record_overrides={
        'number': None, 'h_m': None, 'h_measured': False,
    })
    upload = _stage_upload_direct(settings, tmp_path, payload)
    client = Client()
    client.force_login(writer_user)
    response = _post(client, reverse('ipso-upload-import-free-survey', args=[upload.pk]), {'survey_id': survey.pk})
    assert response.status_code == 409
    response = _post(client, reverse('ipso-upload-excluded-export', args=[upload.pk]), {
        'review_token': response.json()['review_token'],
    })
    assert response.status_code == 200
    reader = csv_io.read(response.content)
    raw = list(reader)[0]
    assert raw[S.CSV_COL_TREE] == ''
    raw[S.CSV_COL_H_M] = '20'
    delimiter, _ = csv_io.export_format()
    buffer, writer = csv_io.csv_buffer(delimiter)
    writer.writerow(reader.fieldnames)
    writer.writerow([raw[name] for name in reader.fieldnames])
    parsed, errors = csv_trees.validate_rows(csv_io.read(buffer.getvalue()), csv_trees.db_indexes(survey),
                                           has_date_column=True, default_date=None)
    assert not errors, errors
    assert parsed[0]['number'] == 1

    for explicit in (1, 2 ** 63 - 1):
        buffer, writer = csv_io.csv_buffer(delimiter)
        writer.writerow(reader.fieldnames)
        writer.writerow([raw[name] for name in reader.fieldnames])
        numbered = {**raw, S.CSV_COL_TREE: str(explicit)}
        writer.writerow([numbered[name] for name in reader.fieldnames])
        parsed, errors = csv_trees.validate_rows(
            csv_io.read(buffer.getvalue()), csv_trees.db_indexes(survey),
            has_date_column=True, default_date=None,
        )
        if explicit == 1:
            assert not errors
            assert [row['number'] for row in parsed] == [2, 1]
        else:
            assert errors, 'Automatic numbering must reject storage overflow'
            assert [row['number'] for row in parsed] == [explicit]


@pytest.mark.parametrize('fields', [
    {'lat': 91}, {'lat': None, 'lon': 16.2}, {'acc_m': -1}, {'d_cm': 10 ** 18},
    {'number': 10 ** 30},
])
def test_invalid_position_or_unrepresentable_volume_is_a_row_error(
        partial_import, settings, tmp_path, fields):
    client, upload, _, body, _, model = partial_import
    payload = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    payload[RECORDS][1].update(h_m='20', h_measured=True, **fields)
    upload.delete()
    upload = _stage_upload_direct(settings, tmp_path, payload)
    route = {'martellate': 'ipso-upload-import-martellate',
             'samples': 'ipso-upload-import-samples', 'free_survey': 'ipso-upload-import-free-survey'}
    response = _post(client, reverse(route[upload.mode], args=[upload.pk]), body)
    assert response.status_code == 409, response.content
    assert response.json()['status'] == 'import_review'
    assert response.json()['valid_count'] == response.json()['excluded_count'] == 1
    assert 'Riga 30' in response.json()['errors'][0]
    assert not model.objects.exists()


@pytest.mark.parametrize('partial_import', ['samples', 'free_survey'], indirect=True)
def test_conflicting_sample_dates_exclude_the_entire_group(partial_import, settings, tmp_path):
    client, upload, _, body, _, model = partial_import
    payload = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    payload[RECORDS][1].update(h_m='20', date='2026-06-18')
    upload.delete()
    upload = _stage_upload_direct(settings, tmp_path, payload)
    route = 'ipso-upload-import-samples' if upload.mode == 'samples' else 'ipso-upload-import-free-survey'
    response = _post(client, reverse(route, args=[upload.pk]), body)
    assert response.status_code == 409, response.content
    assert response.json()['valid_count'] == 0
    assert response.json()['excluded_count'] == 2
    assert all('date' in error.lower() for error in response.json()['errors'])
    assert not model.objects.exists()


@pytest.mark.parametrize('partial_import', ['martellate', 'samples'], indirect=True)
def test_new_conflicting_tree_requires_review_even_without_parent_version_change(partial_import):
    client, upload, target, body, url, model = partial_import
    if upload.mode == 'martellate':
        target.state = HarvestPlanItemState.MARKED
        target.save(update_fields=['state'])
    review = _review(client, url, body)
    export = _post(client, reverse('ipso-upload-excluded-export', args=[upload.pk]), {
        'review_token': review['review_token'],
    })
    source = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())[RECORDS][0]
    other = {'species_id': source['species_id'], 'parcel_id': source['parcel_id'],
             'number': 25, 'd_cm': 42, 'h_m': '22', 'date': source['date'],
             'operator': 'Other entry', 'nonce': 'other-entry'}
    version = target.version
    if upload.mode == 'martellate':
        other['harvest_plan_item_id'] = target.pk
        response = _post(client, '/api/piano-di-taglio/mark/save/', other)
    else:
        other.update(survey_id=target.pk, sample_area_id=source['sample_area_id'], fustaia='true')
        response = _post(client, '/api/campionamenti/tree/save/', other)
    assert response.status_code == 200, response.content
    target.refresh_from_db()
    assert target.version == version
    response = _post(client, url, {
        **body, 'review_token': review['review_token'],
        'export_token': export['X-Import-Export-Token'], 'nonce': 'new-row-conflict',
    })
    assert response.status_code == 409, response.content
    assert response.json()['valid_count'] == 0 and response.json()['excluded_count'] == 2
    assert model.objects.count() == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


def test_destination_deleted_after_preparation_returns_an_error_without_importing(
        partial_import, monkeypatch):
    from apps.ipso import views
    client, upload, target, body, url, model = partial_import
    review = _review(client, url, body)
    exported = _post(client, reverse('ipso-upload-excluded-export', args=[upload.pk]), {
        'review_token': review['review_token'],
    })
    prepare = views._prepare_tree_review
    def remove_destination_after_read(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        target.delete()
        return prepared
    monkeypatch.setattr(views, '_prepare_tree_review', remove_destination_after_read)
    # No nonce is needed for this fault injection: the deletion represents an
    # independent committed operation between the initial read and write lock.
    response = _post(client, url, {**body, 'review_token': review['review_token'],
                                  'export_token': exported['X-Import-Export-Token'],
                                  'warnings_confirmed': True})
    assert response.status_code == 400, response.content
    assert not model.objects.exists()
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED and upload.import_result == {}
