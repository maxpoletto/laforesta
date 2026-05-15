"""Tests for Impostazioni (settings) views."""

import json

import pytest
from django.test import Client

from apps.base.models import Crew, LoginMethod, Role, Species, Tractor, User
from config import strings as S


# ---------------------------------------------------------------------------
# Client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def admin_client(admin_user):
    c = Client()
    c.force_login(admin_user)
    return c


@pytest.fixture
def writer_client(writer_user):
    c = Client()
    c.force_login(writer_user)
    return c


@pytest.fixture
def reader_client(reader_user):
    c = Client()
    c.force_login(reader_user)
    return c


def _post(client, url, data):
    return client.post(url, data=json.dumps(data), content_type='application/json')


# ---------------------------------------------------------------------------
# Password change
# ---------------------------------------------------------------------------

class TestPasswordView:
    URL = '/api/impostazioni/password/'

    def test_change_password(self, writer_client, writer_user):
        resp = _post(writer_client, self.URL, {
            'password1': 'newsecure99!', 'password2': 'newsecure99!',
        })
        assert resp.status_code == 200
        assert resp.json()[S.MESSAGE] == S.PASSWORD_CHANGED
        writer_user.refresh_from_db()
        assert writer_user.check_password('newsecure99!')

    def test_mismatch(self, writer_client):
        resp = _post(writer_client, self.URL, {
            'password1': 'newsecure99!', 'password2': 'different!',
        })
        assert resp.status_code == 400

    def test_too_short(self, writer_client):
        resp = _post(writer_client, self.URL, {
            'password1': '123', 'password2': '123',
        })
        assert resp.status_code == 400

    def test_requires_auth(self, db):
        resp = _post(Client(), self.URL, {
            'password1': 'newsecure99!', 'password2': 'newsecure99!',
        })
        assert resp.status_code == 302


# ---------------------------------------------------------------------------
# Crews
# ---------------------------------------------------------------------------

class TestCrews:
    def test_data(self, writer_client, crews):
        resp = writer_client.get('/api/impostazioni/crews/data/')
        assert resp.status_code == 200
        data = resp.json()
        assert data[S.COLUMNS][0] == S.ROW_ID
        assert len(data[S.ROWS]) == 2

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/api/impostazioni/crews/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[S.HTML]

    def test_form_edit(self, writer_client, crews):
        resp = writer_client.get(f'/api/impostazioni/crews/form/{crews[0].id}/')
        assert resp.status_code == 200
        assert crews[0].name in resp.json()[S.HTML]

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            S.FIELD_NAME: 'Gamma', S.FIELD_NOTES: 'test notes', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        data = resp.json()
        # _crew_row returns [id, name, notes, active]; name is at index 1.
        assert data[S.RECORD][1] == 'Gamma'
        assert Crew.objects.filter(name='Gamma').exists()

    def test_save_update(self, writer_client, crews):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            S.ROW_ID: str(crews[0].id), S.VERSION: str(crews[0].version),
            S.FIELD_NAME: 'Renamed', S.FIELD_NOTES: '', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        crews[0].refresh_from_db()
        assert crews[0].name == 'Renamed'
        assert crews[0].version == 2

    def test_save_conflict(self, writer_client, crews):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            S.ROW_ID: str(crews[0].id), S.VERSION: '999',
            S.FIELD_NAME: 'Conflict', S.FIELD_NOTES: '', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.STATUS] == 'conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            S.FIELD_NAME: '', S.FIELD_NOTES: '', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.STATUS] == 'validation_error'

    def test_reader_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/impostazioni/crews/data/')
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Tractors
# ---------------------------------------------------------------------------

class TestTractors:
    def test_data(self, writer_client, tractors):
        resp = writer_client.get('/api/impostazioni/tractors/data/')
        data = resp.json()
        assert len(data[S.ROWS]) == 2

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/api/impostazioni/tractors/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[S.HTML]

    def test_form_edit(self, writer_client, tractors):
        resp = writer_client.get(f'/api/impostazioni/tractors/form/{tractors[0].id}/')
        assert resp.status_code == 200
        assert 'Fiat' in resp.json()[S.HTML]

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            S.FIELD_MANUFACTURER: 'John Deere', S.FIELD_MODEL: '5100M', S.FIELD_YEAR: '2020', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        t = Tractor.objects.get(manufacturer='John Deere')
        assert t.year == 2020

    def test_save_create_no_year(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            S.FIELD_MANUFACTURER: 'Kubota', S.FIELD_MODEL: 'M7', S.FIELD_YEAR: '', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        assert Tractor.objects.get(manufacturer='Kubota').year is None

    def test_save_conflict(self, writer_client, tractors):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            S.ROW_ID: str(tractors[0].id), S.VERSION: '999',
            S.FIELD_MANUFACTURER: 'X', S.FIELD_MODEL: 'Y', S.FIELD_YEAR: '', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.STATUS] == 'conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            S.FIELD_MANUFACTURER: '', S.FIELD_MODEL: '', S.FIELD_YEAR: '', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.STATUS] == 'validation_error'


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class TestSpecies:
    def test_data(self, writer_client, species):
        resp = writer_client.get('/api/impostazioni/species/data/')
        data = resp.json()
        assert len(data[S.ROWS]) == 3

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/api/impostazioni/species/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[S.HTML]

    def test_form_edit(self, writer_client, species):
        resp = writer_client.get(f'/api/impostazioni/species/form/{species[0].id}/')
        assert resp.status_code == 200
        assert 'Abete' in resp.json()[S.HTML]

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            S.FIELD_COMMON_NAME: 'Faggio', S.FIELD_LATIN_NAME: 'Fagus sylvatica',
            S.FIELD_DENSITY: '10.5', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        assert Species.objects.filter(common_name='Faggio').exists()

    def test_save_conflict(self, writer_client, species):
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            S.ROW_ID: str(species[0].id), S.VERSION: '999',
            S.FIELD_COMMON_NAME: 'X', S.FIELD_LATIN_NAME: '',
            S.FIELD_DENSITY: '9.0', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.STATUS] == 'conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            S.FIELD_COMMON_NAME: '', S.FIELD_LATIN_NAME: '', S.FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.STATUS] == 'validation_error'


# ---------------------------------------------------------------------------
# Users (admin only)
# ---------------------------------------------------------------------------

class TestUsers:
    def test_data(self, admin_client, admin_user):
        resp = admin_client.get('/api/impostazioni/users/data/')
        data = resp.json()
        assert len(data[S.ROWS]) >= 1

    def test_writer_forbidden(self, writer_client):
        resp = writer_client.get('/api/impostazioni/users/data/')
        assert resp.status_code == 403

    def test_form_add(self, admin_client):
        resp = admin_client.get('/api/impostazioni/users/form/')
        assert resp.status_code == 200
        assert 'login_method' in resp.json()[S.HTML]

    def test_form_edit(self, admin_client, writer_user):
        resp = admin_client.get(f'/api/impostazioni/users/form/{writer_user.id}/')
        assert resp.status_code == 200
        assert writer_user.username in resp.json()[S.HTML]

    def test_create_password_user(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.FIELD_USERNAME: 'newuser', S.FIELD_FIRST_NAME: 'New', S.FIELD_LAST_NAME: 'User',
            S.FIELD_EMAIL: 'newuser@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            'password1': 'testpass123!', 'password2': 'testpass123!',
            S.FIELD_ROLE: Role.READER, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        u = User.objects.get(username='newuser')
        assert u.check_password('testpass123!')
        assert u.role == Role.READER

    def test_create_oauth_user(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.FIELD_USERNAME: 'oauthuser@example.com', S.FIELD_FIRST_NAME: '', S.FIELD_LAST_NAME: '',
            S.FIELD_EMAIL: 'oauthuser@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.OAUTH,
            'password1': '', 'password2': '',
            S.FIELD_ROLE: Role.WRITER, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        u = User.objects.get(username='oauthuser@example.com')
        assert not u.has_usable_password()

    def test_create_password_user_requires_password(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.FIELD_USERNAME: 'nopass', S.FIELD_FIRST_NAME: '', S.FIELD_LAST_NAME: '',
            S.FIELD_EMAIL: 'nopass@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            'password1': '', 'password2': '',
            S.FIELD_ROLE: Role.READER, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400

    def test_update_user(self, admin_client, writer_user):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.ROW_ID: str(writer_user.id),
            S.FIELD_USERNAME: 'renamed', S.FIELD_FIRST_NAME: 'A', S.FIELD_LAST_NAME: 'B',
            S.FIELD_EMAIL: 'renamed@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            'password1': '', 'password2': '',
            S.FIELD_ROLE: Role.ADMIN, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        writer_user.refresh_from_db()
        assert writer_user.username == 'renamed'
        assert writer_user.role == Role.ADMIN

    def test_update_user_with_new_password(self, admin_client, writer_user):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.ROW_ID: str(writer_user.id),
            S.FIELD_USERNAME: writer_user.username, S.FIELD_FIRST_NAME: '', S.FIELD_LAST_NAME: '',
            S.FIELD_EMAIL: 'writer@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            'password1': 'brandnew99!', 'password2': 'brandnew99!',
            S.FIELD_ROLE: Role.WRITER, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        writer_user.refresh_from_db()
        assert writer_user.check_password('brandnew99!')

    def test_username_required(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.FIELD_USERNAME: '', S.FIELD_FIRST_NAME: '', S.FIELD_LAST_NAME: '',
            S.FIELD_EMAIL: 'nouser@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            'password1': 'testpass123!', 'password2': 'testpass123!',
            S.FIELD_ROLE: Role.READER, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400

    def test_create_password_mismatch(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.FIELD_USERNAME: 'mismatch', S.FIELD_FIRST_NAME: '', S.FIELD_LAST_NAME: '',
            S.FIELD_EMAIL: 'mismatch@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            'password1': 'testpass123!', 'password2': 'different123!',
            S.FIELD_ROLE: Role.READER, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.MESSAGE] == S.PASSWORD_MISMATCH

    def test_update_password_mismatch(self, admin_client, writer_user):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            S.ROW_ID: str(writer_user.id),
            S.FIELD_USERNAME: writer_user.username, S.FIELD_FIRST_NAME: '', S.FIELD_LAST_NAME: '',
            S.FIELD_EMAIL: 'writer@example.com',
            S.FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            'password1': 'newpass123!', 'password2': 'other123!',
            S.FIELD_ROLE: Role.WRITER, S.FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[S.MESSAGE] == S.PASSWORD_MISMATCH
