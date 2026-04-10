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
    URL = '/abies/api/impostazioni/password/'

    def test_change_password(self, writer_client, writer_user):
        resp = _post(writer_client, self.URL, {
            'password1': 'newsecure99!', 'password2': 'newsecure99!',
        })
        assert resp.status_code == 200
        assert resp.json()['message'] == S.PASSWORD_CHANGED
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
        resp = writer_client.get('/abies/api/impostazioni/crews/data/')
        assert resp.status_code == 200
        data = resp.json()
        assert data['columns'][0] == 'row_id'
        assert len(data['rows']) == 2

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/abies/api/impostazioni/crews/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()['html']

    def test_form_edit(self, writer_client, crews):
        resp = writer_client.get(f'/abies/api/impostazioni/crews/form/{crews[0].id}/')
        assert resp.status_code == 200
        assert crews[0].name in resp.json()['html']

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/abies/api/impostazioni/crews/save/', {
            'name': 'Gamma', 'notes': 'test notes', 'active': 'true',
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data['record'][2] == 'Gamma'
        assert Crew.objects.filter(name='Gamma').exists()

    def test_save_update(self, writer_client, crews):
        resp = _post(writer_client, '/abies/api/impostazioni/crews/save/', {
            'row_id': str(crews[0].id), 'version': str(crews[0].version),
            'name': 'Renamed', 'notes': '', 'active': 'true',
        })
        assert resp.status_code == 200
        crews[0].refresh_from_db()
        assert crews[0].name == 'Renamed'
        assert crews[0].version == 2

    def test_save_conflict(self, writer_client, crews):
        resp = _post(writer_client, '/abies/api/impostazioni/crews/save/', {
            'row_id': str(crews[0].id), 'version': '999',
            'name': 'Conflict', 'notes': '', 'active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/abies/api/impostazioni/crews/save/', {
            'name': '', 'notes': '', 'active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'validation_error'

    def test_reader_forbidden(self, reader_client, db):
        resp = reader_client.get('/abies/api/impostazioni/crews/data/')
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Tractors
# ---------------------------------------------------------------------------

class TestTractors:
    def test_data(self, writer_client, tractors):
        resp = writer_client.get('/abies/api/impostazioni/tractors/data/')
        data = resp.json()
        assert len(data['rows']) == 2

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/abies/api/impostazioni/tractors/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()['html']

    def test_form_edit(self, writer_client, tractors):
        resp = writer_client.get(f'/abies/api/impostazioni/tractors/form/{tractors[0].id}/')
        assert resp.status_code == 200
        assert 'Fiat' in resp.json()['html']

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/abies/api/impostazioni/tractors/save/', {
            'manufacturer': 'John Deere', 'model': '5100M', 'year': '2020', 'active': 'true',
        })
        assert resp.status_code == 200
        t = Tractor.objects.get(manufacturer='John Deere')
        assert t.year == 2020

    def test_save_create_no_year(self, writer_client, db):
        resp = _post(writer_client, '/abies/api/impostazioni/tractors/save/', {
            'manufacturer': 'Kubota', 'model': 'M7', 'year': '', 'active': 'true',
        })
        assert resp.status_code == 200
        assert Tractor.objects.get(manufacturer='Kubota').year is None

    def test_save_conflict(self, writer_client, tractors):
        resp = _post(writer_client, '/abies/api/impostazioni/tractors/save/', {
            'row_id': str(tractors[0].id), 'version': '999',
            'manufacturer': 'X', 'model': 'Y', 'year': '', 'active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/abies/api/impostazioni/tractors/save/', {
            'manufacturer': '', 'model': '', 'year': '', 'active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'validation_error'


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class TestSpecies:
    def test_data(self, writer_client, species):
        resp = writer_client.get('/abies/api/impostazioni/species/data/')
        data = resp.json()
        assert len(data['rows']) == 3

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/abies/api/impostazioni/species/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()['html']

    def test_form_edit(self, writer_client, species):
        resp = writer_client.get(f'/abies/api/impostazioni/species/form/{species[0].id}/')
        assert resp.status_code == 200
        assert 'Abete' in resp.json()['html']

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/abies/api/impostazioni/species/save/', {
            'common_name': 'Faggio', 'latin_name': 'Fagus sylvatica', 'active': 'true',
        })
        assert resp.status_code == 200
        assert Species.objects.filter(common_name='Faggio').exists()

    def test_save_conflict(self, writer_client, species):
        resp = _post(writer_client, '/abies/api/impostazioni/species/save/', {
            'row_id': str(species[0].id), 'version': '999',
            'common_name': 'X', 'latin_name': '', 'active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/abies/api/impostazioni/species/save/', {
            'common_name': '', 'latin_name': '', 'active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'validation_error'


# ---------------------------------------------------------------------------
# Users (admin only)
# ---------------------------------------------------------------------------

class TestUsers:
    def test_data(self, admin_client, admin_user):
        resp = admin_client.get('/abies/api/impostazioni/users/data/')
        data = resp.json()
        assert len(data['rows']) >= 1

    def test_writer_forbidden(self, writer_client):
        resp = writer_client.get('/abies/api/impostazioni/users/data/')
        assert resp.status_code == 403

    def test_form_add(self, admin_client):
        resp = admin_client.get('/abies/api/impostazioni/users/form/')
        assert resp.status_code == 200
        assert 'login_method' in resp.json()['html']

    def test_form_edit(self, admin_client, writer_user):
        resp = admin_client.get(f'/abies/api/impostazioni/users/form/{writer_user.id}/')
        assert resp.status_code == 200
        assert writer_user.username in resp.json()['html']

    def test_create_password_user(self, admin_client):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'username': 'newuser', 'first_name': 'New', 'last_name': 'User',
            'login_method': LoginMethod.PASSWORD,
            'password1': 'testpass123!', 'password2': 'testpass123!',
            'role': Role.READER, 'is_active': 'true',
        })
        assert resp.status_code == 200
        u = User.objects.get(username='newuser')
        assert u.check_password('testpass123!')
        assert u.role == Role.READER

    def test_create_oauth_user(self, admin_client):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'username': 'oauthuser@example.com', 'first_name': '', 'last_name': '',
            'login_method': LoginMethod.OAUTH,
            'password1': '', 'password2': '',
            'role': Role.WRITER, 'is_active': 'true',
        })
        assert resp.status_code == 200
        u = User.objects.get(username='oauthuser@example.com')
        assert not u.has_usable_password()

    def test_create_password_user_requires_password(self, admin_client):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'username': 'nopass', 'first_name': '', 'last_name': '',
            'login_method': LoginMethod.PASSWORD,
            'password1': '', 'password2': '',
            'role': Role.READER, 'is_active': 'true',
        })
        assert resp.status_code == 400

    def test_update_user(self, admin_client, writer_user):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'row_id': str(writer_user.id),
            'username': 'renamed', 'first_name': 'A', 'last_name': 'B',
            'login_method': LoginMethod.PASSWORD,
            'password1': '', 'password2': '',
            'role': Role.ADMIN, 'is_active': 'true',
        })
        assert resp.status_code == 200
        writer_user.refresh_from_db()
        assert writer_user.username == 'renamed'
        assert writer_user.role == Role.ADMIN

    def test_update_user_with_new_password(self, admin_client, writer_user):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'row_id': str(writer_user.id),
            'username': writer_user.username, 'first_name': '', 'last_name': '',
            'login_method': LoginMethod.PASSWORD,
            'password1': 'brandnew99!', 'password2': 'brandnew99!',
            'role': Role.WRITER, 'is_active': 'true',
        })
        assert resp.status_code == 200
        writer_user.refresh_from_db()
        assert writer_user.check_password('brandnew99!')

    def test_username_required(self, admin_client):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'username': '', 'first_name': '', 'last_name': '',
            'login_method': LoginMethod.PASSWORD,
            'password1': 'testpass123!', 'password2': 'testpass123!',
            'role': Role.READER, 'is_active': 'true',
        })
        assert resp.status_code == 400

    def test_create_password_mismatch(self, admin_client):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'username': 'mismatch', 'first_name': '', 'last_name': '',
            'login_method': LoginMethod.PASSWORD,
            'password1': 'testpass123!', 'password2': 'different123!',
            'role': Role.READER, 'is_active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['message'] == S.PASSWORD_MISMATCH

    def test_update_password_mismatch(self, admin_client, writer_user):
        resp = _post(admin_client, '/abies/api/impostazioni/users/save/', {
            'row_id': str(writer_user.id),
            'username': writer_user.username, 'first_name': '', 'last_name': '',
            'login_method': LoginMethod.PASSWORD,
            'password1': 'newpass123!', 'password2': 'other123!',
            'role': Role.WRITER, 'is_active': 'true',
        })
        assert resp.status_code == 400
        assert resp.json()['message'] == S.PASSWORD_MISMATCH
