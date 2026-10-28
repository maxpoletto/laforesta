"""Administrative account edits must preserve newer decisions and access."""

import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from allauth.account.models import EmailAddress
from allauth.socialaccount.models import SocialAccount, SocialToken
from django.contrib.auth.models import update_last_login
from django.db import connection, connections
from django.test import Client

from apps.base.models import DigestStatus, LoginMethod, Role, UsedNonce, User
from config.constants import (
    COLUMNS, HTML, MESSAGE, PATCHES, RECORD, ROWS, STATUS, STATUS_CONFLICT,
    STATUS_VALIDATION_ERROR, VERSION,
)

pytestmark = pytest.mark.django_db
SAVE_URL = '/api/impostazioni/users/save/'


@pytest.fixture
def admin_client(admin_user):
    client = Client()
    client.force_login(admin_user)
    return client


def payload(user, **changes):
    return {
        'row_id': user.pk, 'version': getattr(user, 'version', 1),
        'username': user.username, 'email': user.email or f'{user.username}@example.com',
        'first_name': user.first_name, 'last_name': user.last_name,
        'role': user.role, 'login_method': user.login_method,
        'is_active': user.is_active, 'password1': '', 'password2': '',
        **changes,
    }


def post(client, body):
    return client.post(SAVE_URL, json.dumps(body), content_type='application/json')


def snapshot():
    """Rejections must leave account, OAuth, audit, and retry state untouched."""
    models = (User, User.history.model, EmailAddress, SocialAccount, SocialToken,
              DigestStatus, UsedNonce)
    return [list(model.objects.order_by('pk').values()) for model in models]


def test_user_form_and_data_expose_current_version(admin_client, writer_user):
    html = admin_client.get(
        f'/api/impostazioni/users/form/{writer_user.pk}/',
    ).json()[HTML]
    assert re.search(r'name="version" value="1"', html)
    digest = admin_client.get('/api/impostazioni/users/data/').json()
    index = digest[COLUMNS].index(VERSION)
    assert next(row[index] for row in digest[ROWS] if row[0] == writer_user.pk) == 1


def test_edit_increments_version_once_and_returns_it(admin_client, writer_user):
    before = writer_user.history.count()
    response = post(admin_client, payload(writer_user, first_name='Corrected'))
    assert response.status_code == 200
    writer_user.refresh_from_db()
    assert writer_user.version == 2
    assert writer_user.history.count() == before + 1
    assert writer_user.history.first().version == 2
    assert response.json()[PATCHES][0][RECORD][1] == 2


@pytest.mark.parametrize('changes', [
    {'role': Role.READER}, {'is_active': False},
    {'email': 'corrected@example.com'}, {'login_method': LoginMethod.OAUTH},
])
def test_stale_form_cannot_overwrite_new_admin_decisions(
        admin_client, writer_user, changes):
    old_form = payload(writer_user, first_name='Old form', nonce='stale-form')
    assert post(admin_client, payload(writer_user, **changes)).status_code == 200
    before = snapshot()
    response = post(admin_client, old_form)
    assert response.status_code == 400
    data = response.json()
    assert data[STATUS] == STATUS_CONFLICT
    assert data[RECORD][1] == 2
    assert re.search(r'name="version" value="2"', data[HTML])
    assert snapshot() == before


@pytest.mark.parametrize('version', [None, '', 0, -1, 999, 'invalid'])
def test_missing_or_wrong_version_is_stale(admin_client, writer_user, version):
    body = payload(writer_user, version=version)
    if version is None:
        body.pop(VERSION)
    before = snapshot()
    response = post(admin_client, body)
    assert response.status_code == 400
    assert response.json()[STATUS] in (STATUS_CONFLICT, STATUS_VALIDATION_ERROR)
    assert snapshot() == before


def test_unknown_account_returns_404(admin_client, writer_user):
    response = post(admin_client, payload(writer_user, row_id=999999))
    assert response.status_code == 404


@pytest.mark.parametrize('changes', [{'role': Role.READER}, {'is_active': False}])
def test_last_active_admin_cannot_be_removed(admin_client, admin_user, changes):
    # Inactive administrators and Django staff/superuser flags do not grant
    # the application's administrator role.
    User.objects.create_user(username='inactive', role=Role.ADMIN, is_active=False)
    User.objects.create_user(username='super', role=Role.READER, is_superuser=True)
    body = payload(admin_user, nonce='last-admin', **changes)
    before = snapshot()
    response = post(admin_client, body)
    assert response.status_code == 400
    assert response.json()[STATUS] == STATUS_VALIDATION_ERROR
    assert response.json()[MESSAGE]
    assert snapshot() == before


def test_last_admin_can_correct_other_fields(admin_client, admin_user):
    response = post(admin_client, payload(admin_user, email='fixed@example.com'))
    assert response.status_code == 200
    admin_user.refresh_from_db()
    assert admin_user.email == 'fixed@example.com'


@pytest.mark.parametrize('changes', [{'role': Role.READER}, {'is_active': False}])
def test_admin_can_be_removed_when_another_active_admin_exists(
        admin_client, admin_user, changes):
    User.objects.create_user(username='another', role=Role.ADMIN)
    response = post(admin_client, payload(admin_user, **changes))
    assert response.status_code == 200
    admin_user.refresh_from_db()
    assert all(getattr(admin_user, key) == value for key, value in changes.items())


@pytest.mark.parametrize('method', LoginMethod.values)
@pytest.mark.parametrize('email', [
    'invalid', 'a@', 'a b@example.com', 'a@example..com',
    'a' * 64 + '@' + 'b' * 63 + '.' + 'c' * 63 + '.' + 'd' * 58 + '.org',
])
def test_email_syntax_validated_without_mutations(admin_client, writer_user, method, email):
    before = snapshot()
    response = post(admin_client, payload(writer_user, email=email, login_method=method))
    assert response.status_code == 400
    assert response.json()[STATUS] == STATUS_VALIDATION_ERROR
    assert snapshot() == before


def test_oauth_create_has_one_version_and_one_history_entry(admin_client):
    body = {
        'email': 'new@example.com', 'login_method': LoginMethod.OAUTH,
        'role': Role.READER, 'is_active': True,
        'password1': 'ignore-this-password', 'password2': 'ignore-this-password',
    }
    assert post(admin_client, body).status_code == 200
    user = User.objects.get(email=body['email'])
    assert not user.has_usable_password()
    assert user.version == 1
    assert user.history.count() == 1


def test_user_versions_advance_on_ordinary_saves_but_not_last_login(writer_user):
    writer_user.first_name = 'Changed'
    writer_user.save(update_fields=['first_name'])
    writer_user.refresh_from_db()
    assert writer_user.version == 2
    update_last_login(None, writer_user)
    writer_user.refresh_from_db()
    assert writer_user.version == 2
    writer_user.save(update_fields=[])
    writer_user.refresh_from_db()
    assert writer_user.version == 2


def test_stale_model_instance_cannot_regress_version(writer_user):
    other = User.objects.get(pk=writer_user.pk)
    other.first_name = 'New'
    other.save(update_fields=['first_name'])
    writer_user.last_name = 'Surname'
    writer_user.save(update_fields=['last_name'])
    writer_user.refresh_from_db()
    assert writer_user.version == 3
    assert writer_user.first_name == 'New'


@pytest.mark.parametrize('field,value', [
    ('landing_page', '/bosco'), ('password', 'a-new-password-hash'),
])
def test_non_admin_account_changes_also_stale_admin_forms(
        admin_client, writer_user, field, value):
    old_form = payload(writer_user)
    setattr(writer_user, field, value)
    writer_user.save(update_fields=[field])
    response = post(admin_client, old_form)
    assert response.status_code == 400
    assert response.json()[STATUS] == STATUS_CONFLICT


def test_login_bookkeeping_does_not_stale_admin_form(admin_client, writer_user):
    body = payload(writer_user)
    update_last_login(None, writer_user)
    assert post(admin_client, body).status_code == 200


def test_late_failure_rolls_back_version_history_links_and_receipt(
        admin_client, writer_user, monkeypatch):
    writer_user.email = 'linked@example.com'
    writer_user.login_method = LoginMethod.OAUTH
    writer_user.save()
    EmailAddress.objects.create(user=writer_user, email=writer_user.email,
                                primary=True, verified=True)
    account = SocialAccount.objects.create(user=writer_user, provider='microsoft', uid='linked')
    SocialToken.objects.create(account=account, token='token')
    body = payload(writer_user, email='replacement@example.com', nonce='rollback-edit')
    before = snapshot()

    def fail(*args):
        raise RuntimeError('Failure after all account mutations')

    monkeypatch.setattr('apps.impostazioni.views.mark_stale', fail)
    admin_client.raise_request_exception = False
    assert post(admin_client, body).status_code == 500
    assert snapshot() == before


def test_stale_oauth_edit_preserves_provider_identity(admin_client, writer_user):
    writer_user.email = 'linked@example.com'
    writer_user.login_method = LoginMethod.OAUTH
    writer_user.save()
    account = SocialAccount.objects.create(user=writer_user, provider='microsoft', uid='linked')
    SocialToken.objects.create(account=account, token='token')
    assert post(admin_client, payload(writer_user, first_name='Current')).status_code == 200
    before = snapshot()
    response = post(admin_client, payload(writer_user, email='old-form@example.com', nonce='stale-oauth'))
    assert response.status_code == 400
    assert response.json()[STATUS] == STATUS_CONFLICT
    assert snapshot() == before


def test_personal_password_save_preserves_newer_admin_fields(
        admin_client, writer_user, monkeypatch):
    client = Client()
    client.force_login(writer_user)
    original = User.set_password

    def concurrent_admin_edit(user, password):
        assert post(admin_client, payload(user, role=Role.READER)).status_code == 200
        return original(user, password)

    monkeypatch.setattr(User, 'set_password', concurrent_admin_edit)
    response = client.post('/api/impostazioni/password/', json.dumps({
        'current_password': 'testpass123!',
        'password1': 'a-new-secure-password99!', 'password2': 'a-new-secure-password99!',
    }), content_type='application/json')
    assert response.status_code == 200
    writer_user.refresh_from_db()
    assert writer_user.role == Role.READER
    assert writer_user.check_password('a-new-secure-password99!')


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('targets', ['self', 'same', 'cross'])
def test_competing_admin_changes_are_serialized(
        admin_user, writer_user, tmp_path, monkeypatch, targets):
    """Real overlapping requests, including the no-nonce SQLite path."""
    if connection.vendor != 'sqlite':
        pytest.skip('This fixture snapshots SQLite for independent connections')
    from apps.impostazioni import views

    second_admin = User.objects.create_user(username='second', role=Role.ADMIN)
    clients = [Client(raise_request_exception=False), Client(raise_request_exception=False)]
    for client, actor in zip(clients, [admin_user, second_admin]):
        client.force_login(actor)
    if targets == 'same':
        bodies = [payload(writer_user, first_name=name) for name in ['First', 'Second']]
    else:
        accounts = ([admin_user, second_admin] if targets == 'self'
                    else [second_admin, admin_user])
        bodies = [payload(account, role=Role.READER) for account in accounts]
    database = tmp_path / 'account-concurrency.sqlite3'
    connection.ensure_connection()
    with sqlite3.connect(database) as db:
        connection.connection.backup(db)
    monkeypatch.setitem(connections.databases, 'default', {
        **connection.settings_dict, 'NAME': str(database),
    })
    first_before_commit = threading.Event()
    release_first = threading.Event()
    second_at_write = threading.Event()
    original_sync = views._sync_email_address

    def pause_first(user):
        if threading.current_thread().name.startswith('first'):
            first_before_commit.set()
            assert release_first.wait(10)
        return original_sync(user)

    monkeypatch.setattr(views, '_sync_email_address', pause_first)

    def submit(index):
        def observe(execute, sql, params, many, context):
            if index == 1 and sql.lstrip().upper().startswith('UPDATE "BASE_USER"'):
                second_at_write.set()
            return execute(sql, params, many, context)
        try:
            with connections['default'].execute_wrapper(observe):
                return post(clients[index], bodies[index])
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='first') as first_pool, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='second') as second_pool:
        first = first_pool.submit(submit, 0)
        try:
            assert first_before_commit.wait(10)
            second = second_pool.submit(submit, 1)
            assert second_at_write.wait(10)
        finally:
            release_first.set()
        first_response, second_response = first.result(10), second.result(10)
    assert first_response.status_code == 200, first_response.content
    assert second_response.status_code == (403 if targets == 'cross' else 400), second_response.content
    with sqlite3.connect(database) as db:
        assert db.execute(
            'SELECT COUNT(*) FROM base_user WHERE role = ? AND is_active = 1',
            [Role.ADMIN],
        ).fetchone()[0] == (2 if targets == 'same' else 1)
