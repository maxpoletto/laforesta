"""Tests for Impostazioni (settings) views."""

import json
import re

import pytest
from django.test import Client

from apps.base.models import Crew, LoginMethod, Role, Species, Tractor, User
from apps.impostazioni.views import SPECIES_COLS
from config import strings as S
from config.constants import (
    COLUMNS, FIELD_ACTIVE, FIELD_COMMON_NAME, FIELD_DENSITY, FIELD_EMAIL,
    FIELD_FIRST_NAME, FIELD_IS_ACTIVE, FIELD_LAST_NAME, FIELD_LATIN_NAME,
    FIELD_LOGIN_METHOD, FIELD_MANUFACTURER, FIELD_MINOR, FIELD_MODEL, FIELD_NAME,
    FIELD_NOTES, FIELD_PASSWORD1, FIELD_PASSWORD2, FIELD_ROLE, FIELD_USERNAME,
    FIELD_YEAR, HTML, MESSAGE, PATCHES, RECORD, ROWS, ROW_ID, STATUS,
    STATUS_CONFLICT, STATUS_VALIDATION_ERROR, VERSION,
)


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
            FIELD_PASSWORD1: 'newsecure99!', FIELD_PASSWORD2: 'newsecure99!',
        })
        assert resp.status_code == 200
        assert resp.json()[MESSAGE] == S.PASSWORD_CHANGED
        writer_user.refresh_from_db()
        assert writer_user.check_password('newsecure99!')

    def test_mismatch(self, writer_client):
        resp = _post(writer_client, self.URL, {
            FIELD_PASSWORD1: 'newsecure99!', FIELD_PASSWORD2: 'different!',
        })
        assert resp.status_code == 400

    def test_too_short(self, writer_client):
        resp = _post(writer_client, self.URL, {
            FIELD_PASSWORD1: '123', FIELD_PASSWORD2: '123',
        })
        assert resp.status_code == 400

    def test_requires_auth(self, db):
        resp = _post(Client(), self.URL, {
            FIELD_PASSWORD1: 'newsecure99!', FIELD_PASSWORD2: 'newsecure99!',
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
        assert data[COLUMNS][0] == ROW_ID
        assert len(data[ROWS]) == 2

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/api/impostazioni/crews/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[HTML]

    def test_form_edit(self, writer_client, crews):
        resp = writer_client.get(f'/api/impostazioni/crews/form/{crews[0].id}/')
        assert resp.status_code == 200
        assert crews[0].name in resp.json()[HTML]

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            FIELD_NAME: 'Gamma', FIELD_NOTES: 'test notes', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        data = resp.json()
        # _crew_row returns [id, name, notes, active]; name is at index 1.
        assert data[PATCHES][0][RECORD][1] == 'Gamma'
        assert Crew.objects.filter(name='Gamma').exists()

    def test_save_update(self, writer_client, crews):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            ROW_ID: str(crews[0].id), VERSION: str(crews[0].version),
            FIELD_NAME: 'Renamed', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        crews[0].refresh_from_db()
        assert crews[0].name == 'Renamed'
        assert crews[0].version == 2

    def test_save_conflict(self, writer_client, crews):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            ROW_ID: str(crews[0].id), VERSION: '999',
            FIELD_NAME: 'Conflict', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            FIELD_NAME: '', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR

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
        assert len(data[ROWS]) == 2

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/api/impostazioni/tractors/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[HTML]

    def test_form_edit(self, writer_client, tractors):
        resp = writer_client.get(f'/api/impostazioni/tractors/form/{tractors[0].id}/')
        assert resp.status_code == 200
        assert 'Fiat' in resp.json()[HTML]

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            FIELD_MANUFACTURER: 'John Deere', FIELD_MODEL: '5100M', FIELD_YEAR: '2020', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        t = Tractor.objects.get(manufacturer='John Deere')
        assert t.year == 2020

    def test_save_create_no_year(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            FIELD_MANUFACTURER: 'Kubota', FIELD_MODEL: 'M7', FIELD_YEAR: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        assert Tractor.objects.get(manufacturer='Kubota').year is None

    def test_save_conflict(self, writer_client, tractors):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            ROW_ID: str(tractors[0].id), VERSION: '999',
            FIELD_MANUFACTURER: 'X', FIELD_MODEL: 'Y', FIELD_YEAR: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            FIELD_MANUFACTURER: '', FIELD_MODEL: '', FIELD_YEAR: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class TestSpecies:
    def test_data(self, writer_client, species):
        resp = writer_client.get('/api/impostazioni/species/data/')
        data = resp.json()
        assert len(data[ROWS]) == 4

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/api/impostazioni/species/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[HTML]

    def test_form_edit(self, writer_client, species):
        resp = writer_client.get(f'/api/impostazioni/species/form/{species[0].id}/')
        assert resp.status_code == 200
        assert 'Abete' in resp.json()[HTML]

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            FIELD_COMMON_NAME: 'Faggio', FIELD_LATIN_NAME: 'Fagus sylvatica',
            FIELD_DENSITY: '10.5', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        assert Species.objects.filter(common_name='Faggio').exists()

    def test_save_accepts_locale_comma(self, writer_client, db):
        """The it-locale form submits a comma density; stored canonically
        (density routes through the locale-aware parser)."""
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            FIELD_COMMON_NAME: 'Cerro', FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '8,25', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200, resp.content
        assert float(Species.objects.get(common_name='Cerro').density) == 8.25

    def test_save_rejects_zero_density(self, writer_client, db):
        """Density must be > 0."""
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            FIELD_COMMON_NAME: 'Zzz', FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '0', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400, resp.content
        assert not Species.objects.filter(common_name='Zzz').exists()

    def test_save_conflict(self, writer_client, species):
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            ROW_ID: str(species[0].id), VERSION: '999',
            FIELD_COMMON_NAME: 'X', FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '9.0', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            FIELD_COMMON_NAME: '', FIELD_LATIN_NAME: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR

    def test_data_exposes_minor(self, writer_client, species):
        """Each row carries its minor flag under the COL_MINOR column."""
        resp = writer_client.get('/api/impostazioni/species/data/')
        data = resp.json()
        assert S.COL_MINOR in data[COLUMNS]
        minor_idx = data[COLUMNS].index(S.COL_MINOR)
        minor_by_id = {row[0]: row[minor_idx] for row in data[ROWS]}
        for sp in species:
            assert minor_by_id[sp.id] is sp.minor

    def test_form_includes_minor(self, writer_client, db):
        resp = writer_client.get('/api/impostazioni/species/form/')
        assert f'name="{FIELD_MINOR}"' in resp.json()[HTML]

    def test_edit_form_checks_minor(self, writer_client, species):
        """The edit form reflects the stored flag: the minor checkbox is
        rendered checked for a minor species, unchecked otherwise."""
        def minor_checkbox(obj):
            html = writer_client.get(
                f'/api/impostazioni/species/form/{obj.id}/').json()[HTML]
            tag = re.search(
                rf'<input[^>]*name="{FIELD_MINOR}"[^>]*value="true"[^>]*>', html)
            assert tag, html
            return tag.group(0)

        assert 'checked' in minor_checkbox(species[2])      # Acero, minor=True
        assert 'checked' not in minor_checkbox(species[0])  # Abete, minor=False

    def test_save_persists_minor(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            FIELD_COMMON_NAME: 'Pioppo', FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '7.0', FIELD_ACTIVE: 'true', FIELD_MINOR: 'true',
        })
        assert resp.status_code == 200, resp.content
        assert Species.objects.get(common_name='Pioppo').minor is True

    def test_save_create_minor_false(self, writer_client, db):
        """Unchecked minor stores False — both when the form omits the key
        and when it sends the hidden field's 'false' (the real submission)."""
        for name, extra in [('Ontano', {}), ('Frassino', {FIELD_MINOR: 'false'})]:
            resp = _post(writer_client, '/api/impostazioni/species/save/', {
                FIELD_COMMON_NAME: name, FIELD_LATIN_NAME: '',
                FIELD_DENSITY: '7.0', FIELD_ACTIVE: 'true', **extra,
            })
            assert resp.status_code == 200, resp.content
            assert Species.objects.get(common_name=name).minor is False

    def test_save_update_toggles_minor(self, writer_client, species):
        """Editing a species flips minor both ways through the optimistic
        save, and the returned record carries minor at the SPECIES_COLS
        index the in-place table update reads."""
        minor_idx = SPECIES_COLS.index(S.COL_MINOR)

        acero = species[2]   # starts minor=True
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            ROW_ID: str(acero.id), VERSION: str(acero.version),
            FIELD_COMMON_NAME: acero.common_name, FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '9.0', FIELD_ACTIVE: 'true', FIELD_MINOR: 'false',
        })
        assert resp.status_code == 200, resp.content
        acero.refresh_from_db()
        assert acero.minor is False
        assert acero.version == 2
        assert resp.json()[PATCHES][0][RECORD][minor_idx] is False

        abete = species[0]   # starts minor=False
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            ROW_ID: str(abete.id), VERSION: str(abete.version),
            FIELD_COMMON_NAME: abete.common_name, FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '9.0', FIELD_ACTIVE: 'true', FIELD_MINOR: 'true',
        })
        assert resp.status_code == 200, resp.content
        abete.refresh_from_db()
        assert abete.minor is True
        assert resp.json()[PATCHES][0][RECORD][minor_idx] is True

        # A later edit that leaves minor checked preserves it (the form always
        # resubmits the flag's current state), bumping the version again.
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            ROW_ID: str(abete.id), VERSION: str(abete.version),
            FIELD_COMMON_NAME: abete.common_name, FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '8.0', FIELD_ACTIVE: 'true', FIELD_MINOR: 'true',
        })
        assert resp.status_code == 200, resp.content
        abete.refresh_from_db()
        assert abete.minor is True
        assert abete.version == 3

    def test_other_species_cannot_be_minor(self, writer_client, species):
        """The 'Altro' bucket species backs the minor-aggregation column;
        flagging it minor would break prelievi generation, so the save is
        rejected and the flag stays False."""
        altro = next(s for s in species if s.common_name == S.SPECIES_OTHER)
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            ROW_ID: str(altro.id), VERSION: str(altro.version),
            FIELD_COMMON_NAME: altro.common_name, FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '9.0', FIELD_ACTIVE: 'true', FIELD_MINOR: 'true',
        })
        assert resp.status_code == 400, resp.content
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        altro.refresh_from_db()
        assert altro.minor is False


# ---------------------------------------------------------------------------
# Users (admin only)
# ---------------------------------------------------------------------------

class TestUsers:
    def test_data(self, admin_client, admin_user):
        resp = admin_client.get('/api/impostazioni/users/data/')
        data = resp.json()
        assert len(data[ROWS]) >= 1

    def test_writer_forbidden(self, writer_client):
        resp = writer_client.get('/api/impostazioni/users/data/')
        assert resp.status_code == 403

    def test_form_add(self, admin_client):
        resp = admin_client.get('/api/impostazioni/users/form/')
        assert resp.status_code == 200
        assert 'login_method' in resp.json()[HTML]

    def test_form_edit(self, admin_client, writer_user):
        resp = admin_client.get(f'/api/impostazioni/users/form/{writer_user.id}/')
        assert resp.status_code == 200
        assert writer_user.username in resp.json()[HTML]

    def test_create_password_user(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            FIELD_USERNAME: 'newuser', FIELD_FIRST_NAME: 'New', FIELD_LAST_NAME: 'User',
            FIELD_EMAIL: 'newuser@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            FIELD_PASSWORD1: 'testpass123!', FIELD_PASSWORD2: 'testpass123!',
            FIELD_ROLE: Role.READER, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        u = User.objects.get(username='newuser')
        assert u.check_password('testpass123!')
        assert u.role == Role.READER

    def test_create_oauth_user(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            FIELD_USERNAME: 'oauthuser@example.com', FIELD_FIRST_NAME: '', FIELD_LAST_NAME: '',
            FIELD_EMAIL: 'oauthuser@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.OAUTH,
            FIELD_PASSWORD1: '', FIELD_PASSWORD2: '',
            FIELD_ROLE: Role.WRITER, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        u = User.objects.get(username='oauthuser@example.com')
        assert not u.has_usable_password()

    def test_create_password_user_requires_password(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            FIELD_USERNAME: 'nopass', FIELD_FIRST_NAME: '', FIELD_LAST_NAME: '',
            FIELD_EMAIL: 'nopass@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            FIELD_PASSWORD1: '', FIELD_PASSWORD2: '',
            FIELD_ROLE: Role.READER, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400

    def test_update_user(self, admin_client, writer_user):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            ROW_ID: str(writer_user.id),
            FIELD_USERNAME: 'renamed', FIELD_FIRST_NAME: 'A', FIELD_LAST_NAME: 'B',
            FIELD_EMAIL: 'renamed@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            FIELD_PASSWORD1: '', FIELD_PASSWORD2: '',
            FIELD_ROLE: Role.ADMIN, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        writer_user.refresh_from_db()
        assert writer_user.username == 'renamed'
        assert writer_user.role == Role.ADMIN

    def test_update_user_with_new_password(self, admin_client, writer_user):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            ROW_ID: str(writer_user.id),
            FIELD_USERNAME: writer_user.username, FIELD_FIRST_NAME: '', FIELD_LAST_NAME: '',
            FIELD_EMAIL: 'writer@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            FIELD_PASSWORD1: 'brandnew99!', FIELD_PASSWORD2: 'brandnew99!',
            FIELD_ROLE: Role.WRITER, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        writer_user.refresh_from_db()
        assert writer_user.check_password('brandnew99!')

    def test_username_required(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            FIELD_USERNAME: '', FIELD_FIRST_NAME: '', FIELD_LAST_NAME: '',
            FIELD_EMAIL: 'nouser@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            FIELD_PASSWORD1: 'testpass123!', FIELD_PASSWORD2: 'testpass123!',
            FIELD_ROLE: Role.READER, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400

    def test_create_password_mismatch(self, admin_client):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            FIELD_USERNAME: 'mismatch', FIELD_FIRST_NAME: '', FIELD_LAST_NAME: '',
            FIELD_EMAIL: 'mismatch@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            FIELD_PASSWORD1: 'testpass123!', FIELD_PASSWORD2: 'different123!',
            FIELD_ROLE: Role.READER, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[MESSAGE] == S.PASSWORD_MISMATCH

    def test_update_password_mismatch(self, admin_client, writer_user):
        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            ROW_ID: str(writer_user.id),
            FIELD_USERNAME: writer_user.username, FIELD_FIRST_NAME: '', FIELD_LAST_NAME: '',
            FIELD_EMAIL: 'writer@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.PASSWORD,
            FIELD_PASSWORD1: 'newpass123!', FIELD_PASSWORD2: 'other123!',
            FIELD_ROLE: Role.WRITER, FIELD_IS_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[MESSAGE] == S.PASSWORD_MISMATCH
