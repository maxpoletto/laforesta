"""Tests for Campionamenti API views (data endpoints, M3b)."""

import gzip
import json
from datetime import date
from decimal import Decimal

import pytest
from django.test import Client

from apps.base.models import (
    Parcel, Sample, SampleArea, SampleGrid, Survey, Tree, TreeSample,
)


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


@pytest.fixture
def sample_setup(db, regions, eclasses, species):
    """A tiny grid + survey + sample + tree fixture."""
    parcel = Parcel.objects.create(
        name='1', region=regions[0], eclass=eclasses[0],
        area_ha=Decimal('5.0'),
    )
    grid = SampleGrid.objects.create(name='Test grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcel, number='1',
        lat=0.0, lng=0.0, r_m=12,
    )
    survey = Survey.objects.create(name='Test survey', sample_grid=grid)
    sample = Sample.objects.create(
        sample_area=area, survey=survey, date=date(2024, 9, 15),
    )
    tree = Tree.objects.create(
        species=species[0], parcel=parcel, preserved=False, coppice=False,
    )
    TreeSample.objects.create(
        sample=sample, tree=tree, shoot=0, standard=False,
        number=1, d_cm=30, h_m=Decimal('20.00'),
        l10_mm=10, volume_m3=Decimal('0.7022'), mass_q=Decimal('6.32'),
    )
    return {
        'grid': grid, 'area': area, 'survey': survey,
        'sample': sample, 'tree': tree,
    }


def _read_gzip_json(resp):
    return json.loads(gzip.decompress(resp.getvalue()))


class TestDataEndpoints:
    def test_grids_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/grids/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert 'Nome' in d['columns']
        assert any(r[d['columns'].index('Nome')] == 'Test grid' for r in d['rows'])

    def test_surveys_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/surveys/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert d['rows'][0][d['columns'].index('N. aree visitate')] == 1
        assert d['rows'][0][d['columns'].index('N. aree totali')] == 1

    def test_sample_areas_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/sample-areas/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert len(d['rows']) == 1
        assert d['rows'][0][d['columns'].index('Raggio')] == 12

    def test_samples_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/samples/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert d['rows'][0][d['columns'].index('N. alberi')] == 1

    def test_trees_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        survey_id = sample_setup['survey'].id
        resp = writer_client.get(f'/api/campionamenti/trees/{survey_id}/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert len(d['rows']) == 1
        row = d['rows'][0]
        assert row[d['columns'].index('Specie')] == 'Abete'
        assert row[d['columns'].index('Tipo')] == 'fustaia'
        assert row[d['columns'].index('D (cm)')] == 30

    def test_trees_data_unknown_survey(self, writer_client, sample_setup,
                                       tmp_path, settings):
        """Requesting a non-existent survey id returns an empty digest."""
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/trees/9999/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert d['rows'] == []

    def test_requires_auth(self, db):
        resp = Client().get('/api/campionamenti/surveys/data/')
        assert resp.status_code == 302    # redirected to login

    def test_reader_can_read(self, reader_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = reader_client.get('/api/campionamenti/surveys/data/')
        assert resp.status_code == 200


class TestTreeForm:
    def test_form_add_requires_survey_and_area(self, writer_client, sample_setup):
        # Missing survey / area → 404
        resp = writer_client.get('/api/campionamenti/tree/form/')
        assert resp.status_code == 404

    def test_form_add_returns_html(self, writer_client, sample_setup):
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}&area={s["area"].id}'
        )
        assert resp.status_code == 200
        html = resp.json()['html']
        assert '<form' in html
        assert s['area'].parcel.name in html

    def test_form_add_rejects_mismatched_grid(self, writer_client, sample_setup,
                                              regions, eclasses):
        """Survey on grid A + area on grid B → 404."""
        from apps.base.models import Parcel, SampleArea, SampleGrid
        from decimal import Decimal
        other_grid = SampleGrid.objects.create(name='Other')
        other_parcel = Parcel.objects.create(
            name='99', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        other_area = SampleArea.objects.create(
            sample_grid=other_grid, parcel=other_parcel, number='1',
            lat=0.0, lng=0.0, r_m=12,
        )
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}&area={other_area.id}'
        )
        assert resp.status_code == 404


class TestTreeSave:
    @staticmethod
    def _post(client, body):
        import json
        return client.post(
            '/api/campionamenti/tree/save/',
            data=json.dumps(body), content_type='application/json',
        )

    def test_create_fustaia_tree(self, writer_client, sample_setup):
        from apps.base.models import Tree, TreeSample
        s = sample_setup
        n_before = TreeSample.objects.count()
        resp = self._post(writer_client, {
            'survey_id': str(s['survey'].id),
            'sample_area_id': str(s['area'].id),
            'species_id': str(s['tree'].species_id),
            'number': '42',
            'd_cm': '30', 'h_m': '20.5', 'l10_mm': '12',
            'volume_m3': '0.7022', 'mass_q': '6.32',
            'fustaia': 'true',
            'lat': '0.001', 'lng': '0.001',
            'preserved': '',
        })
        assert resp.status_code == 200, resp.content
        data = resp.json()
        assert TreeSample.objects.count() == n_before + 1
        ts = TreeSample.objects.get(id=data['row_id'])
        assert ts.number == 42
        assert ts.tree.coppice is False
        assert ts.tree.preserved is False
        assert ts.volume_m3 is not None and ts.mass_q is not None

    def test_create_rejects_mismatched_grid(self, writer_client, sample_setup,
                                            regions, eclasses):
        from apps.base.models import Parcel, SampleArea, SampleGrid
        from decimal import Decimal
        other_grid = SampleGrid.objects.create(name='Other')
        other_parcel = Parcel.objects.create(
            name='99', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        other_area = SampleArea.objects.create(
            sample_grid=other_grid, parcel=other_parcel, number='1',
            lat=0.0, lng=0.0, r_m=12,
        )
        s = sample_setup
        resp = self._post(writer_client, {
            'survey_id': str(s['survey'].id),
            'sample_area_id': str(other_area.id),
            'species_id': str(s['tree'].species_id),
            'number': '1', 'd_cm': '30', 'h_m': '20', 'l10_mm': '0',
            'volume_m3': '0.5', 'mass_q': '4.7',
            'fustaia': 'true',
        })
        assert resp.status_code == 400
        assert 'griglia' in resp.json()['message'].lower()

    def test_create_rejects_zero_diameter(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, {
            'survey_id': str(s['survey'].id),
            'sample_area_id': str(s['area'].id),
            'species_id': str(s['tree'].species_id),
            'number': '1', 'd_cm': '0', 'h_m': '20', 'l10_mm': '0',
            'volume_m3': '0', 'mass_q': '0', 'fustaia': 'true',
        })
        assert resp.status_code == 400

    def test_reader_cannot_save(self, reader_client, sample_setup):
        s = sample_setup
        resp = self._post(reader_client, {
            'survey_id': str(s['survey'].id),
            'sample_area_id': str(s['area'].id),
            'species_id': str(s['tree'].species_id),
            'number': '1', 'd_cm': '30', 'h_m': '20',
            'l10_mm': '0', 'volume_m3': '0.5', 'mass_q': '4.7',
            'fustaia': 'true',
        })
        assert resp.status_code == 403
