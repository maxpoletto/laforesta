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

    def test_rejects_duplicate_number_in_sample(self, writer_client, sample_setup):
        """Spec: within a single Sample, tree numbers must be unique."""
        s = sample_setup
        # The sample fixture already has tree with number=1.
        resp = self._post(writer_client, {
            'survey_id': str(s['survey'].id),
            'sample_area_id': str(s['area'].id),
            'species_id': str(s['tree'].species_id),
            'number': '1', 'd_cm': '40', 'h_m': '25',
            'l10_mm': '0', 'volume_m3': '0.8', 'mass_q': '5.7',
            'fustaia': 'true',
        })
        assert resp.status_code == 400
        assert 'già utilizzato' in resp.json()['message']

    def test_existing_tree_reuses_tree_id(self, writer_client, sample_setup):
        """Picking an existing tree from the pulldown must NOT create a new
        Tree row — same physical tree across surveys (cross-sample identity).
        """
        s = sample_setup
        # Make a second survey on the same grid + sample on the same area.
        second_survey = Survey.objects.create(
            name='Second campaign', sample_grid=s['grid'],
        )
        n_trees_before = Tree.objects.count()
        n_ts_before = TreeSample.objects.count()

        resp = self._post(writer_client, {
            'survey_id': str(second_survey.id),
            'sample_area_id': str(s['area'].id),
            'tree_pick': str(s['tree'].id),
            'species_id': str(s['tree'].species_id),
            'number': '1',         # propagated from the existing tree
            'd_cm': '35', 'h_m': '21', 'l10_mm': '0',
            'volume_m3': '0.9', 'mass_q': '7.1',
            'fustaia': 'true',
            'lat': str(s['tree'].lat or 0.0),
            'lng': str(s['tree'].lng or 0.0),
        })
        assert resp.status_code == 200, resp.content

        # One new TreeSample, zero new Trees.
        assert Tree.objects.count() == n_trees_before
        assert TreeSample.objects.count() == n_ts_before + 1
        ts = TreeSample.objects.get(id=resp.json()['row_id'])
        assert ts.tree_id == s['tree'].id
        assert ts.number == 1                         # propagated
        assert ts.sample.survey_id == second_survey.id
        assert ts.d_cm == 35                          # new measurement

    def test_existing_tree_not_in_area_rejected(self, writer_client, sample_setup,
                                                regions, eclasses, species):
        """A tree in a *different* sample area cannot be picked here."""
        s = sample_setup
        # Build a parallel area + sample + tree in a different SampleGrid+area.
        other_parcel = Parcel.objects.create(
            name='99', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        other_grid = SampleGrid.objects.create(name='Other grid')
        other_area = SampleArea.objects.create(
            sample_grid=other_grid, parcel=other_parcel, number='1',
            lat=0.0, lng=0.0, r_m=12,
        )
        other_survey = Survey.objects.create(
            name='Other survey', sample_grid=other_grid,
        )
        other_sample = Sample.objects.create(
            sample_area=other_area, survey=other_survey, date=date(2024, 1, 1),
        )
        other_tree = Tree.objects.create(
            species=species[0], parcel=other_parcel,
            preserved=False, coppice=False,
        )
        TreeSample.objects.create(
            sample=other_sample, tree=other_tree, shoot=0, standard=False,
            number=7, d_cm=50, h_m=Decimal('30.00'), l10_mm=0,
            volume_m3=Decimal('1.0'), mass_q=Decimal('9.0'),
        )

        resp = self._post(writer_client, {
            'survey_id': str(s['survey'].id),
            'sample_area_id': str(s['area'].id),    # ours, not other_area
            'tree_pick': str(other_tree.id),
            'species_id': str(s['tree'].species_id),
            'number': '7', 'd_cm': '40', 'h_m': '20', 'l10_mm': '0',
            'volume_m3': '0.5', 'mass_q': '4.0',
            'fustaia': 'true',
        })
        assert resp.status_code == 400


class TestTreeFormPriorTrees:
    """Form GET reflects the prior-trees pulldown contents."""

    def test_lists_prior_trees(self, writer_client, sample_setup):
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}'
            f'&area={s["area"].id}'
        )
        assert resp.status_code == 200
        html = resp.json()['html']
        # The existing fixture has tree number=1 in this area.
        assert 'id="id_tree_pick"' in html
        assert '+ nuovo albero' in html
        assert 'n.1' in html
        # next_number = max(existing)+1 = 2
        assert 'data-next="2"' in html

    def test_fustaia_default_off_for_coppice_parcel(
        self, writer_client, sample_setup, regions, eclasses,
    ):
        """For parcels whose eclass is coppice, fustaia defaults to OFF
        (spec §"Manual tree + sample entry": "Defaults to fustaia, except
        in parcels whose eclass.coppice = true, where it defaults to ceduo")."""
        coppice_eclass = next(e for e in eclasses if e.coppice)
        coppice_parcel = Parcel.objects.create(
            name='99', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'),
        )
        coppice_area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=coppice_parcel,
            number='9', lat=0.0, lng=0.0, r_m=12,
        )
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={sample_setup["survey"].id}'
            f'&area={coppice_area.id}'
        )
        html = resp.json()['html']
        # The fustaia checkbox should NOT be checked for coppice areas.
        # In the form template, "checked" only appears on the fustaia
        # checkbox.  Match the input line.
        idx = html.find('id="id_fustaia"')
        assert idx >= 0
        # Look at the entire input tag (up to the closing >).
        tag = html[max(0, idx - 200):idx + 200]
        assert 'checked' not in tag

    def test_fustaia_default_on_for_non_coppice_parcel(
        self, writer_client, sample_setup,
    ):
        """Non-coppice areas default fustaia=on (regression: the existing
        fixture's parcel uses non-coppice eclass A)."""
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={sample_setup["survey"].id}'
            f'&area={sample_setup["area"].id}'
        )
        html = resp.json()['html']
        idx = html.find('id="id_fustaia"')
        assert idx >= 0
        tag = html[max(0, idx - 200):idx + 200]
        assert 'checked' in tag

    def test_empty_area_has_only_new(self, writer_client, sample_setup,
                                     regions, eclasses):
        """An area with no prior trees shows only '+ nuovo albero'."""
        s = sample_setup
        new_parcel = Parcel.objects.create(
            name='2', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        empty_area = SampleArea.objects.create(
            sample_grid=s['grid'], parcel=new_parcel, number='2',
            lat=0.0, lng=0.0, r_m=12,
        )
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}'
            f'&area={empty_area.id}'
        )
        html = resp.json()['html']
        assert '+ nuovo albero' in html
        assert 'data-next="1"' in html
        # `data-number` is only emitted on prior-tree options.
        assert 'data-number=' not in html


class TestGridSave:
    @staticmethod
    def _post(client, body):
        import json
        return client.post(
            '/api/campionamenti/grid/save/',
            data=json.dumps(body), content_type='application/json',
        )

    def test_form_renders(self, writer_client, db):
        resp = writer_client.get('/api/campionamenti/grid/form/')
        assert resp.status_code == 200
        html = resp.json()['html']
        # Modal carries all three creation paths per campionamenti.md §1.
        assert 'data-path="empty"' in html
        assert 'data-path="auto"' in html
        assert 'data-path="csv"' in html
        # Default-active body is the empty-grid create form.
        assert 'campionamenti-grid-form-empty' in html

    def test_create_empty_grid(self, writer_client, db):
        from apps.base.models import SampleGrid
        resp = self._post(writer_client, {
            'name': 'Griglia di prova', 'description': 'desc',
        })
        assert resp.status_code == 200
        data = resp.json()
        g = SampleGrid.objects.get(id=data['row_id'])
        assert g.name == 'Griglia di prova'

    def test_name_required(self, writer_client, db):
        resp = self._post(writer_client, {'name': '', 'description': ''})
        assert resp.status_code == 400

    def test_name_duplicate_rejected(self, writer_client, db):
        from apps.base.models import SampleGrid
        SampleGrid.objects.create(name='Dup')
        resp = self._post(writer_client, {'name': 'Dup', 'description': ''})
        assert resp.status_code == 400

    def test_reader_forbidden(self, reader_client, db):
        resp = self._post(reader_client, {'name': 'X'})
        assert resp.status_code == 403


class TestGridSaveAuto:
    URL = '/api/campionamenti/grid/save-auto/'

    @staticmethod
    def _post(client, body):
        import json
        return client.post(
            TestGridSaveAuto.URL,
            data=json.dumps(body), content_type='application/json',
        )

    def test_create_auto_grid(self, writer_client, sample_setup):
        s = sample_setup
        # Re-use the fixture parcel via its compresa+name.
        resp = self._post(writer_client, {
            'name': 'Auto grid',
            'description': '',
            'r_m': 12,
            'points': [
                {'compresa': s['area'].parcel.region.name,
                 'particella': s['area'].parcel.name,
                 'lat': 38.5, 'lng': 16.1},
                {'compresa': s['area'].parcel.region.name,
                 'particella': s['area'].parcel.name,
                 'lat': 38.51, 'lng': 16.11},
            ],
        })
        assert resp.status_code == 200, resp.content
        data = resp.json()
        grid = SampleGrid.objects.get(id=data['row_id'])
        assert grid.name == 'Auto grid'
        # bulk_create made 2 SampleAreas with sequential numbers.
        areas = list(SampleArea.objects.filter(sample_grid=grid).order_by('number'))
        assert len(areas) == 2
        assert {a.number for a in areas} == {'1', '2'}
        assert all(a.r_m == 12 for a in areas)

    def test_unknown_compresa_aborts(self, writer_client, sample_setup):
        n_before = SampleGrid.objects.count()
        resp = self._post(writer_client, {
            'name': 'Bad grid', 'r_m': 12,
            'points': [
                {'compresa': 'Nessuna', 'particella': '1',
                 'lat': 38.5, 'lng': 16.1},
            ],
        })
        assert resp.status_code == 400
        assert SampleGrid.objects.count() == n_before     # no partial commit

    def test_unknown_particella_aborts(self, writer_client, sample_setup):
        s = sample_setup
        n_before = SampleGrid.objects.count()
        resp = self._post(writer_client, {
            'name': 'Bad grid 2', 'r_m': 12,
            'points': [
                {'compresa': s['area'].parcel.region.name,
                 'particella': 'ZZZ',
                 'lat': 38.5, 'lng': 16.1},
            ],
        })
        assert resp.status_code == 400
        assert SampleGrid.objects.count() == n_before

    def test_duplicate_name_rejected(self, writer_client, sample_setup):
        SampleGrid.objects.create(name='Dup auto')
        resp = self._post(writer_client, {
            'name': 'Dup auto', 'r_m': 12,
            'points': [
                {'compresa': sample_setup['area'].parcel.region.name,
                 'particella': sample_setup['area'].parcel.name,
                 'lat': 38.5, 'lng': 16.1},
            ],
        })
        assert resp.status_code == 400

    def test_empty_points_rejected(self, writer_client, db):
        resp = self._post(writer_client, {
            'name': 'Empty', 'r_m': 12, 'points': [],
        })
        assert resp.status_code == 400

    def test_reader_forbidden(self, reader_client, sample_setup):
        resp = self._post(reader_client, {
            'name': 'X', 'r_m': 12,
            'points': [
                {'compresa': sample_setup['area'].parcel.region.name,
                 'particella': sample_setup['area'].parcel.name,
                 'lat': 0.0, 'lng': 0.0},
            ],
        })
        assert resp.status_code == 403


class TestSurveySave:
    @staticmethod
    def _post(client, body):
        import json
        return client.post(
            '/api/campionamenti/survey/save/',
            data=json.dumps(body), content_type='application/json',
        )

    def test_form_renders(self, writer_client, sample_setup):
        resp = writer_client.get('/api/campionamenti/survey/form/')
        assert resp.status_code == 200
        html = resp.json()['html']
        # Modal carries both creation paths per campionamenti.md §2.
        assert 'data-path="empty"' in html
        assert 'data-path="csv"' in html
        assert 'campionamenti-survey-form-empty' in html
        # Grid pulldown contains the fixture's grid.
        assert sample_setup['grid'].name in html

    def test_create_empty_survey(self, writer_client, sample_setup):
        from apps.base.models import Survey
        resp = self._post(writer_client, {
            'name': 'Rilevamento di prova',
            'sample_grid_id': str(sample_setup['grid'].id),
            'description': 'desc',
        })
        assert resp.status_code == 200
        data = resp.json()
        sv = Survey.objects.get(id=data['row_id'])
        assert sv.name == 'Rilevamento di prova'
        assert sv.sample_grid_id == sample_setup['grid'].id

    def test_name_required(self, writer_client, sample_setup):
        resp = self._post(writer_client, {
            'name': '', 'sample_grid_id': str(sample_setup['grid'].id),
        })
        assert resp.status_code == 400

    def test_grid_required(self, writer_client, db):
        resp = self._post(writer_client, {'name': 'X', 'sample_grid_id': ''})
        assert resp.status_code == 400

    def test_grid_must_exist(self, writer_client, db):
        resp = self._post(writer_client, {'name': 'X', 'sample_grid_id': '9999'})
        assert resp.status_code == 400

    def test_name_duplicate_rejected(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, {
            'name': s['survey'].name,
            'sample_grid_id': str(s['grid'].id),
        })
        assert resp.status_code == 400

    def test_reader_forbidden(self, reader_client, sample_setup):
        s = sample_setup
        resp = self._post(reader_client, {
            'name': 'X', 'sample_grid_id': str(s['grid'].id),
        })
        assert resp.status_code == 403
