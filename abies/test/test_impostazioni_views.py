"""Tests for Impostazioni (settings) views."""

import json
import re
from datetime import date
from decimal import Decimal

import pytest
from django.test import Client

from apps.base.models import (
    Crew, DigestStatus, HarvestPlan, LoginMethod, Role, Sample, SampleArea,
    SampleGrid, Species, Survey, Tractor, Tree, TreeSample, User,
)
from apps.impostazioni.views import SPECIES_COLS
from config import strings as S
from config.constants import (
    COLUMNS, DATA_ID, DIGEST_FUTURE_PRODUCTION, DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES,
    FIELD_ACTIVE, FIELD_COMMON_NAME, FIELD_DENSITY, FIELD_EMAIL,
    FIELD_FIRST_NAME, FIELD_HARVEST_PLAN_ID, FIELD_IS_ACTIVE, FIELD_LAST_NAME,
    FIELD_LATIN_NAME, FIELD_LOGIN_METHOD, FIELD_MANUFACTURER, FIELD_MINOR,
    FIELD_PRESSLER_DEFAULT, FIELD_SPECIES,
    FIELD_MODEL, FIELD_NAME, FIELD_NONCE, FIELD_NOTES, FIELD_PASSWORD1,
    FIELD_PASSWORD2, FIELD_ROLE, FIELD_SURVEY_IDS, FIELD_USERNAME, FIELD_YEAR,
    HTML, MESSAGE, PATCHES, RECORD, ROWS, ROW_ID, STATUS, STATUS_CONFLICT,
    STATUS_VALIDATION_ERROR, VERSION,
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

    def test_oauth_user_cannot_change_local_password(self, db):
        user = User.objects.create_user(
            username='oauthuser@example.com',
            email='oauthuser@example.com',
            password='oldpass123!',
            login_method=LoginMethod.OAUTH,
            role=Role.WRITER,
        )
        client = Client()
        client.force_login(user)

        resp = _post(client, self.URL, {
            FIELD_PASSWORD1: 'newsecure99!', FIELD_PASSWORD2: 'newsecure99!',
        })

        assert resp.status_code == 400
        assert resp.json()[MESSAGE] == S.ERR_FORBIDDEN
        user.refresh_from_db()
        assert user.check_password('oldpass123!')
        assert not user.check_password('newsecure99!')


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

    def test_save_non_numeric_version_conflicts(self, writer_client, crews):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            ROW_ID: str(crews[0].id), VERSION: 'not-a-number',
            FIELD_NAME: 'Conflict', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        crews[0].refresh_from_db()
        assert crews[0].name != 'Conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/crews/save/', {
            FIELD_NAME: '', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR

    def test_save_malformed_json_returns_validation_error(self, writer_client, db):
        resp = writer_client.post(
            '/api/impostazioni/crews/save/',
            data='{',
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert resp.json()[MESSAGE] == S.ERR_JSON_INVALID

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

    def test_save_sets_name(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            FIELD_MANUFACTURER: 'Fiat', FIELD_MODEL: '110-90',
            FIELD_NAME: 'Fiat 110-90',
            FIELD_YEAR: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        assert Tractor.objects.get(name='Fiat 110-90').manufacturer == 'Fiat'

    def test_save_name_optional(self, writer_client, db):
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            FIELD_MANUFACTURER: 'Kubota', FIELD_MODEL: 'M7',
            FIELD_YEAR: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        assert Tractor.objects.get(manufacturer='Kubota').name is None

    def test_save_duplicate_name_rejected(self, writer_client, db):
        Tractor.objects.create(manufacturer='Fiat', model='110-90', name='Fiat 110-90')
        resp = _post(writer_client, '/api/impostazioni/tractors/save/', {
            FIELD_MANUFACTURER: 'Fiat', FIELD_MODEL: '90-90',
            FIELD_NAME: 'Fiat 110-90',
            FIELD_YEAR: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert resp.json()[MESSAGE] == S.ERR_TRACTOR_NAME_DUPLICATE


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
        for name in (
                'prelievi', FIELD_SPECIES, DIGEST_PARCEL_DENDROMETRY,
                DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES,
                'audit',
        ):
            assert DigestStatus.objects.get(name=name).stale is True

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

    def test_save_rejects_zero_pressler(self, writer_client, db):
        """Default Pressler coefficient must be > 0."""
        resp = _post(writer_client, '/api/impostazioni/species/save/', {
            FIELD_COMMON_NAME: 'Zzz', FIELD_LATIN_NAME: '',
            FIELD_DENSITY: '7.0', FIELD_PRESSLER_DEFAULT: '0',
            FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400, resp.content
        assert resp.json()[MESSAGE] == S.ERR_PRESSLER_POSITIVE
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
# Bosco source settings
# ---------------------------------------------------------------------------

class TestFutureProductionSettings:
    DATA_URL = '/api/impostazioni/future-production/data/'
    SAVE_URL = '/api/impostazioni/future-production/save/'

    def test_data_defaults_to_current_plan_with_greatest_end_year(
            self, writer_client, monkeypatch,
    ):
        monkeypatch.setattr(
            'apps.base.selectors.timezone.localdate',
            lambda: date(2026, 6, 10),
        )
        HarvestPlan.objects.create(name='Old', year_start=2010, year_end=2020)
        short = HarvestPlan.objects.create(
            name='Current short', year_start=2020, year_end=2030,
        )
        long = HarvestPlan.objects.create(
            name='Current long', year_start=2020, year_end=2035,
        )

        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert data['active_id'] == long.id
        active_by_id = {p['id']: p['active'] for p in data['plans']}
        assert active_by_id[long.id] is True
        assert active_by_id[short.id] is False

    def test_data_prefers_runtime_active_plan(self, writer_client, monkeypatch):
        monkeypatch.setattr(
            'apps.base.selectors.timezone.localdate',
            lambda: date(2026, 6, 10),
        )
        manual = HarvestPlan.objects.create(
            name='Manual', year_start=2010, year_end=2020, active=True,
        )
        current = HarvestPlan.objects.create(
            name='Current', year_start=2020, year_end=2035,
        )

        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert data['active_id'] == manual.id
        active_by_id = {p['id']: p['active'] for p in data['plans']}
        assert active_by_id[manual.id] is True
        assert active_by_id[current.id] is False

    def test_data_empty_is_robust(self, writer_client, db):
        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        assert resp.json() == {'active_id': None, 'plans': []}

    def test_save_sets_single_active_plan_and_marks_digests(self, writer_client):
        old = HarvestPlan.objects.create(
            name='Old active', year_start=2010, year_end=2020, active=True,
        )
        new = HarvestPlan.objects.create(
            name='New active', year_start=2021, year_end=2030,
        )

        resp = _post(writer_client, self.SAVE_URL, {
            FIELD_HARVEST_PLAN_ID: str(new.id), FIELD_NONCE: 'future-1',
        })

        assert resp.status_code == 200, resp.content
        assert resp.json()[MESSAGE] == S.FUTURE_PRODUCTION_SAVED
        old.refresh_from_db()
        new.refresh_from_db()
        assert old.active is False
        assert new.active is True
        assert old.version == 2
        assert new.version == 2
        patches = {
            p[ROW_ID]: p[RECORD]
            for p in resp.json()[PATCHES]
            if p[DATA_ID] == 'harvest_plans'
        }
        assert patches[old.id][-1] is False
        assert patches[new.id][-1] is True
        for name in (
                DIGEST_FUTURE_PRODUCTION, 'harvest_plans', 'audit',
        ):
            assert DigestStatus.objects.get(name=name).stale is True

    def test_reader_forbidden(self, reader_client, db):
        assert reader_client.get(self.DATA_URL).status_code == 403


class TestDendrometrySettings:
    DATA_URL = '/api/impostazioni/dendrometry/data/'
    SAVE_URL = '/api/impostazioni/dendrometry/save/'

    def test_data_defaults_to_first_survey_and_counts(
            self, writer_client, parcels, species,
    ):
        grid = SampleGrid.objects.create(name='Grid')
        beta = Survey.objects.create(name='Beta', sample_grid=grid)
        alpha = Survey.objects.create(name='Alpha', sample_grid=grid)
        area = SampleArea.objects.create(
            sample_grid=grid, parcel=parcels[0], number='1', lat=0, lon=0,
        )
        sample = Sample.objects.create(
            sample_area=area, survey=alpha, date=date(2024, 9, 1),
        )
        for number in (1, 2):
            tree = Tree.objects.create(species=species[0], parcel=parcels[0])
            TreeSample.objects.create(
                sample=sample, tree=tree, shoot=0, standard=False,
                number=number, d_cm=30, h_m=Decimal('20.00'), l10_mm=10,
                volume_m3=Decimal('1.0000'), mass_q=Decimal('9.000'),
            )

        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert data['active_ids'] == [alpha.id]
        assert data['counts'] == {'trees': 2, 'regions': 1, 'parcels': 1}
        active_by_id = {s['id']: s['active'] for s in data['surveys']}
        assert active_by_id[alpha.id] is True
        assert active_by_id[beta.id] is False

    def test_data_empty_is_robust(self, writer_client, db):
        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        assert resp.json() == {
            'active_ids': [],
            'counts': {'trees': 0, 'regions': 0, 'parcels': 0},
            'surveys': [],
        }

    def test_save_sets_active_surveys_and_marks_digests(self, writer_client):
        grid = SampleGrid.objects.create(name='Grid')
        first = Survey.objects.create(name='First', sample_grid=grid)
        second = Survey.objects.create(name='Second', sample_grid=grid)
        old = Survey.objects.create(name='Old', sample_grid=grid, active=True)

        resp = _post(writer_client, self.SAVE_URL, {
            FIELD_SURVEY_IDS: [str(first.id), str(second.id)],
            FIELD_NONCE: 'dendro-1',
        })

        assert resp.status_code == 200, resp.content
        assert resp.json()[MESSAGE] == S.DENDROMETRY_SAVED
        first.refresh_from_db()
        second.refresh_from_db()
        old.refresh_from_db()
        assert first.active is True
        assert second.active is True
        assert old.active is False
        assert first.version == 2
        assert second.version == 2
        assert old.version == 2
        patches = {
            p[ROW_ID]: p[RECORD]
            for p in resp.json()[PATCHES]
            if p[DATA_ID] == 'surveys'
        }
        assert patches[first.id][-1] is True
        assert patches[second.id][-1] is True
        assert patches[old.id][-1] is False
        for name in (
                DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS,
                'surveys', 'audit',
        ):
            assert DigestStatus.objects.get(name=name).stale is True

    def test_save_rejects_empty_selection_when_surveys_exist(self, writer_client):
        grid = SampleGrid.objects.create(name='Grid')
        Survey.objects.create(name='Survey', sample_grid=grid)

        resp = _post(writer_client, self.SAVE_URL, {
            FIELD_SURVEY_IDS: [], FIELD_NONCE: 'dendro-empty',
        })

        assert resp.status_code == 400
        assert resp.json()[MESSAGE] == S.ERR_DENDROMETRY_SURVEYS_REQUIRED

    def test_save_allows_empty_selection_when_no_surveys_exist(self, writer_client, db):
        resp = _post(writer_client, self.SAVE_URL, {
            FIELD_SURVEY_IDS: [], FIELD_NONCE: 'dendro-no-surveys',
        })

        assert resp.status_code == 200, resp.content
        assert resp.json()[MESSAGE] == S.DENDROMETRY_SAVED

    def test_reader_forbidden(self, reader_client, db):
        assert reader_client.get(self.DATA_URL).status_code == 403


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

    def test_update_to_oauth_clears_usable_password(
        self, admin_client, writer_user,
    ):
        assert writer_user.has_usable_password()

        resp = _post(admin_client, '/api/impostazioni/users/save/', {
            ROW_ID: str(writer_user.id),
            FIELD_USERNAME: writer_user.username, FIELD_FIRST_NAME: '', FIELD_LAST_NAME: '',
            FIELD_EMAIL: 'writer-oauth@example.com',
            FIELD_LOGIN_METHOD: LoginMethod.OAUTH,
            FIELD_PASSWORD1: 'brandnew99!', FIELD_PASSWORD2: 'brandnew99!',
            FIELD_ROLE: Role.WRITER, FIELD_IS_ACTIVE: 'true',
        })

        assert resp.status_code == 200
        writer_user.refresh_from_db()
        assert writer_user.login_method == LoginMethod.OAUTH
        assert writer_user.username == 'writer-oauth@example.com'
        assert not writer_user.has_usable_password()

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
