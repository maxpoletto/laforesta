"""Tests for the global diameter-class setting and its invalidation contract."""

import json

import pytest
from django.test import Client

from apps.base.models import DigestStatus, SiteSettings
from config import strings as S
from config.constants import (
    DIAMETER_CLASS_CENTERED, DIAMETER_CLASS_SHIFTED_DOWN,
    DIAMETER_CLASS_SHIFTED_UP, DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PREFIX_MARK_TREES,
    DIGEST_PREFIX_SAMPLED_TREES, FIELD_DATA_IDS, FIELD_DIAMETER_CLASS_MODE,
    FIELD_NONCE, FIELD_PREFIXES, INVALIDATES, MESSAGE,
)


DATA_URL = '/api/impostazioni/diameter-classes/data/'
SAVE_URL = '/api/impostazioni/diameter-classes/save/'


@pytest.fixture
def writer_client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


@pytest.fixture
def reader_client(reader_user):
    client = Client()
    client.force_login(reader_user)
    return client


def post(client, data):
    return client.post(
        SAVE_URL, data=json.dumps(data), content_type='application/json',
    )


def test_settings_page_places_localized_radio_section_below_hypsometry(
        writer_client):
    response = writer_client.get('/impostazioni')

    assert response.status_code == 200
    html = response.content.decode()
    hypso = html.index('data-settings-section="hypso"')
    diameter = html.index('data-settings-section="diameter-classes"')
    users = html.index('data-settings-section="users"')
    assert hypso < diameter < users
    assert 'Le classi diametriche sono:' in html
    for value, label in (
        (
            DIAMETER_CLASS_CENTERED,
            'centrate: la classe 20 comprende i diametri 18–22 cm.',
        ),
        (
            DIAMETER_CLASS_SHIFTED_UP,
            'spostate verso l’alto: la classe 20 comprende i diametri 20–24 cm.',
        ),
        (
            DIAMETER_CLASS_SHIFTED_DOWN,
            'spostate verso il basso: la classe 20 comprende i diametri 16–20 cm.',
        ),
    ):
        assert f'value="{value}"' in html
        assert label in html


def test_writer_reads_default_mode(writer_client, db):
    response = writer_client.get(DATA_URL)

    assert response.status_code == 200
    assert response.json() == {
        FIELD_DIAMETER_CLASS_MODE: DIAMETER_CLASS_CENTERED,
    }


def test_writer_save_invalidates_all_affected_digest_families(
        writer_client, writer_user):
    affected = (
        DIGEST_PARCEL_DENDROMETRY,
        f'{DIGEST_PREFIX_MARK_TREES}17',
        f'{DIGEST_PREFIX_SAMPLED_TREES}23',
        'audit',
    )
    unaffected = (DIGEST_PARCEL_DENDROMETRY_POINTS, 'unrelated')
    for name in (*affected, *unaffected):
        DigestStatus.objects.create(name=name)

    response = post(writer_client, {
        FIELD_DIAMETER_CLASS_MODE: DIAMETER_CLASS_SHIFTED_UP,
        FIELD_NONCE: 'diameter-classes-1',
    })

    assert response.status_code == 200, response.content
    payload = response.json()
    assert payload[MESSAGE] == S.DIAMETER_CLASSES_SAVED
    assert payload[INVALIDATES] == {
        FIELD_DATA_IDS: [DIGEST_PARCEL_DENDROMETRY],
        FIELD_PREFIXES: [
            DIGEST_PREFIX_MARK_TREES,
            DIGEST_PREFIX_SAMPLED_TREES,
        ],
    }
    settings_obj = SiteSettings.load()
    assert settings_obj.diameter_class_mode == DIAMETER_CLASS_SHIFTED_UP
    assert settings_obj.history.first().history_user_id == writer_user.id
    for name in affected:
        status = DigestStatus.objects.get(name=name)
        assert status.stale is True
        assert status.dirty_seq == 1
    for name in unaffected:
        status = DigestStatus.objects.get(name=name)
        assert status.stale is False
        assert status.dirty_seq == 0


def test_unchanged_save_does_not_invalidate(writer_client, db):
    DigestStatus.objects.create(name=DIGEST_PARCEL_DENDROMETRY)

    response = post(writer_client, {
        FIELD_DIAMETER_CLASS_MODE: DIAMETER_CLASS_CENTERED,
        FIELD_NONCE: 'diameter-classes-noop',
    })

    assert response.status_code == 200
    assert INVALIDATES not in response.json()
    status = DigestStatus.objects.get(name=DIGEST_PARCEL_DENDROMETRY)
    assert status.stale is False
    assert status.dirty_seq == 0


def test_save_rejects_unknown_mode(writer_client, db):
    response = post(writer_client, {
        FIELD_DIAMETER_CLASS_MODE: 'diagonal',
        FIELD_NONCE: 'diameter-classes-invalid',
    })

    assert response.status_code == 400
    assert response.json()[MESSAGE] == S.ERR_DIAMETER_CLASS_MODE_INVALID
    assert SiteSettings.load().diameter_class_mode == DIAMETER_CLASS_CENTERED


def test_reader_cannot_read_or_change_global_mode(reader_client, db):
    assert reader_client.get(DATA_URL).status_code == 403
    response = post(reader_client, {
        FIELD_DIAMETER_CLASS_MODE: DIAMETER_CLASS_SHIFTED_DOWN,
    })
    assert response.status_code == 403
    assert SiteSettings.load().diameter_class_mode == DIAMETER_CLASS_CENTERED
