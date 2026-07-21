"""Tests for Campionamenti API views."""

import base64
import gzip
import json
from datetime import date
from decimal import Decimal
import pytest
from django.test import Client

from apps.base.models import (
    DigestStatus, Parcel, Sample, SampleArea, SampleGrid, Survey, Tree,
    TreeSample, UsedNonce,
)
from config import strings as S
from config.constants import (
    COLUMNS, COL_COPPICE, COL_SURVEY_ID, DATA_ID, DELETES,
    DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS,
    DIGEST_PRESERVED_TREES,
    FIELD_ALTITUDE_M, FIELD_DATE,
    FIELD_DEFAULT_DATE, FIELD_DESCRIPTION, FIELD_D_CM, FIELD_ERRORS, FIELD_FILE,
    FIELD_HIGHFOREST, FIELD_H_M, FIELD_H_MEASURED,
    FIELD_LAT, FIELD_LON, FIELD_MASS_Q, FIELD_NAME, FIELD_NONCE, FIELD_NOTE,
    FIELD_NUMBER, FIELD_PARCEL_ID, FIELD_POINTS, FIELD_PRESERVED, FIELD_R_M,
    FIELD_SAMPLE_AREA_ID, FIELD_SAMPLE_GRID_ID, FIELD_SHOOT, FIELD_SPECIES_ID,
    FIELD_STANDARD, FIELD_SURVEY_ID, FIELD_TREE_PICK, HTML, MESSAGE, PATCHES,
    RECORD, ROWS, ROW_ID, SAMPLE_GRID_UNSTRUCTURED, STATUS, STATUS_CONFLICT,
    STATUS_VALIDATION_ERROR, VERSION,
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
        lat=0.0, lon=0.0, r_m=12,
    )
    survey = Survey.objects.create(name='Test survey', sample_grid=grid)
    sample = Sample.objects.create(
        sample_area=area, survey=survey, date=date(2024, 9, 15),
    )
    tree = Tree.objects.create(
        species=species[0], parcel=parcel, preserved=False, coppice=False,
    )
    TreeSample.objects.create(
        sample=sample, tree=tree, parcel=parcel, shoot=0, standard=False,
        number=1, d_cm=30, h_m=Decimal('20.00'),
        l10_mm=10, volume_m3=Decimal('0.7022'), mass_q=Decimal('6.32'),
    )
    return {
        'grid': grid, 'area': area, 'survey': survey,
        'sample': sample, 'tree': tree,
    }


def _read_gzip_json(resp):
    return json.loads(gzip.decompress(resp.getvalue()))


def _patch(payload, data_id, row_id=None):
    for patch in payload[PATCHES]:
        if patch[DATA_ID] == data_id and (row_id is None or patch[ROW_ID] == row_id):
            return patch
    raise AssertionError(f'missing patch for {data_id}:{row_id}')


def _assert_stale(*names):
    for name in names:
        assert DigestStatus.objects.get(name=name).stale is True


def _csv_b64(csv_text):
    raw = csv_text if isinstance(csv_text, bytes) else csv_text.encode('utf-8-sig')
    return base64.b64encode(raw).decode('ascii')


class TestDataEndpoints:
    def test_grids_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/grids/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert 'Nome' in d[COLUMNS]
        assert any(r[d[COLUMNS].index(S.COL_NAME)] == 'Test grid' for r in d[ROWS])

    def test_surveys_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/surveys/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert d[ROWS][0][d[COLUMNS].index(S.COL_N_AREAS_VISITED)] == 1
        assert d[ROWS][0][d[COLUMNS].index(S.COL_N_AREAS_TOTAL)] == 1

    def test_survey_build_record_matches_generator(
            self, sample_setup, tmp_path, settings, django_assert_num_queries,
    ):
        from apps.base.digests import build_survey_record, generate_surveys

        settings.DIGEST_DIR = tmp_path
        with django_assert_num_queries(2):
            generate_surveys()
        with gzip.open(tmp_path / 'surveys.json.gz', 'rt') as f:
            data = json.load(f)

        survey = Survey.objects.get(pk=sample_setup['survey'].pk)
        gen_row = next(
            row for row in data[ROWS]
            if row[data[COLUMNS].index(ROW_ID)] == survey.id
        )
        assert build_survey_record(survey) == gen_row

    def test_survey_build_record_with_precomputed_aggregates_avoids_queries(
            self, sample_setup, django_assert_num_queries,
    ):
        from apps.base.digests import build_survey_record

        survey = Survey.objects.get(pk=sample_setup['survey'].pk)
        sample_date = sample_setup['sample'].date

        with django_assert_num_queries(0):
            row = build_survey_record(
                survey, n_visited=1, n_total=1,
                first_date=sample_date, last_date=sample_date,
            )

        assert row[0] == survey.id
        assert row[5:] == [
            1, 1, sample_date.isoformat(), sample_date.isoformat(), False,
        ]

    def test_sample_areas_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/sample-areas/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert len(d[ROWS]) == 1
        assert d[ROWS][0][d[COLUMNS].index(S.COL_RADIUS)] == 12

    def test_samples_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/samples/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert d[ROWS][0][d[COLUMNS].index(S.COL_N_TREES)] == 1

    def test_trees_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        survey_id = sample_setup['survey'].id
        resp = writer_client.get(f'/api/campionamenti/trees/{survey_id}/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert len(d[ROWS]) == 1
        row = d[ROWS][0]
        assert row[d[COLUMNS].index(S.COL_SPECIES)] == 'Abete'
        assert row[d[COLUMNS].index(S.COL_PRODUCT)] == S.TYPE_HIGHFOREST
        assert row[d[COLUMNS].index(S.COL_D_CM)] == 30

    def test_trees_data_unknown_survey(self, writer_client, sample_setup,
                                       tmp_path, settings):
        """Requesting a non-existent survey id does not create a dynamic digest."""
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/trees/9999/')
        assert resp.status_code == 404

    def test_requires_auth(self, db):
        resp = Client().get('/api/campionamenti/surveys/data/')
        assert resp.status_code == 302    # redirected to login

    def test_reader_can_read(self, reader_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = reader_client.get('/api/campionamenti/surveys/data/')
        assert resp.status_code == 200


class TestTreeForm:
    def test_form_add_requires_survey_and_area(self, writer_client, sample_setup):
        # Missing survey / area -> 404
        resp = writer_client.get('/api/campionamenti/tree/form/')
        assert resp.status_code == 404

    def test_form_add_rejects_malformed_query_ids(self, writer_client, sample_setup):
        resp = writer_client.get('/api/campionamenti/tree/form/?survey=x&area=1')
        assert resp.status_code == 404

    def test_reader_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/campionamenti/tree/form/')
        assert resp.status_code == 403

    def test_form_add_returns_html(self, writer_client, sample_setup):
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}&area={s["area"].id}'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert '<form' in html
        assert s['area'].parcel.name in html

    def test_unstructured_form_uses_parcel_and_number_fields(
        self, writer_client, parcels, species,
    ):
        survey = Survey.objects.create(name='Free survey')

        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={survey.id}'
        )

        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert 'name="parcel_id"' in html
        assert 'name="number" id="id_number"' in html
        assert 'id="id_tree_pick"' not in html
        assert parcels[0].name in html

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
            lat=0.0, lon=0.0, r_m=12,
        )
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}&area={other_area.id}'
        )
        assert resp.status_code == 404

    def test_fustaia_parcel_defaults_to_abete(
        self, writer_client, sample_setup, species,
    ):
        """Fustaia parcel (eclass.coppice=False) → Abete is preselected
        in the species pulldown.  See views._default_species_id."""
        s = sample_setup
        assert s['area'].parcel.eclass.coppice is False
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}&area={s["area"].id}'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        abete = next(sp for sp in species if sp.common_name == 'Abete')
        assert f'value="{abete.id}"' in html
        # The selected attribute must appear on Abete's option.
        # Use a minimal contextual match — the option's data-name carries
        # the common name, so we look for "data-name=\"Abete\" selected".
        assert f'data-name="Abete"' in html
        # selected attribute appears between value="..." and >, so check
        # the option line wholesale.
        import re
        match = re.search(
            r'<option[^>]*data-name="Abete"[^>]*>', html,
        )
        assert match is not None and 'selected' in match.group(0)

    def test_ceduo_parcel_defaults_to_castagno(
        self, writer_client, sample_setup, regions, eclasses, species,
    ):
        """Ceduo parcel (eclass.coppice=True) → Castagno is preselected."""
        from apps.base.models import Parcel, SampleArea, Survey
        ceduo_eclass = next(e for e in eclasses if e.coppice)
        ceduo_parcel = Parcel.objects.create(
            name='C1', region=regions[0], eclass=ceduo_eclass,
            area_ha=Decimal('2.0'),
            intervention_interval=18, standards_per_ha=75,
        )
        grid = sample_setup['grid']
        ceduo_area = SampleArea.objects.create(
            sample_grid=grid, parcel=ceduo_parcel, number='C1',
            lat=0.0, lon=0.0, r_m=12,
        )
        survey = sample_setup['survey']
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={survey.id}&area={ceduo_area.id}'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        castagno = next(sp for sp in species if sp.common_name == 'Castagno')
        assert f'value="{castagno.id}"' in html
        import re
        match = re.search(
            r'<option[^>]*data-name="Castagno"[^>]*>', html,
        )
        assert match is not None and 'selected' in match.group(0)

    def test_edit_form_hidden_species_carries_density_and_name(
        self, writer_client, sample_setup,
    ):
        """On the edit-tree form, #id_species is a hidden <input> rather
        than a <select>.  wireVMPreview in JS reads density / common_name
        from the element's dataset, so the hidden input must carry
        data-density and data-name — otherwise the live V/m preview
        crashes and the whole form-wiring chain (cancel handler, submit
        interceptor) aborts."""
        ts = TreeSample.objects.get(sample=sample_setup['sample'])
        resp = writer_client.get(f'/api/campionamenti/tree/form/{ts.id}/')
        assert resp.status_code == 200
        html = resp.json()[HTML]
        import re
        match = re.search(
            r'<input[^>]*id="tf-species"[^>]*>', html,
        )
        assert match is not None
        tag = match.group(0)
        assert 'type="hidden"' in tag
        assert 'data-density="' in tag
        assert 'data-name="' in tag
        assert f'name="version" value="{ts.version}"' in html


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
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '42',
            FIELD_D_CM: '30', FIELD_H_M: '20.5', FIELD_H_MEASURED: 'true',
            'l10_mm': '12', 'volume_m3': '0.7022', FIELD_MASS_Q: '6.32',
            FIELD_HIGHFOREST: 'true',
            FIELD_LAT: '0.001', FIELD_LON: '0.001',
            FIELD_PRESERVED: '',
        })
        assert resp.status_code == 200, resp.content
        data = resp.json()
        assert TreeSample.objects.count() == n_before + 1
        ts = TreeSample.objects.get(id=data[ROW_ID])
        assert ts.number == 42
        assert ts.tree.coppice is False
        assert ts.tree.preserved is False
        assert ts.h_measured is True
        assert ts.volume_m3 is not None and ts.mass_q is not None

    def test_create_unstructured_tree_creates_null_area_sample(
            self, writer_client, parcels, species,
    ):
        survey = Survey.objects.create(name='Free tree survey')
        n_trees_before = Tree.objects.count()

        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(survey.id),
            FIELD_PARCEL_ID: str(parcels[0].id),
            FIELD_SPECIES_ID: str(species[0].id),
            FIELD_NUMBER: '1',
            FIELD_DATE: '2026-07-17',
            FIELD_D_CM: '30', FIELD_H_M: '20.5', FIELD_H_MEASURED: 'true',
            'l10_mm': '12', 'volume_m3': '0.7022', FIELD_MASS_Q: '6.32',
            FIELD_HIGHFOREST: 'true',
            FIELD_LAT: '38.5', FIELD_LON: '16.1',
            FIELD_PRESERVED: '',
        })

        assert resp.status_code == 200, resp.content
        sample = Sample.objects.get(survey=survey)
        assert sample.sample_area_id is None
        assert sample.date.isoformat() == '2026-07-17'
        ts = TreeSample.objects.select_related('tree').get(sample=sample)
        assert ts.tree.parcel == parcels[0]
        assert ts.tree.lat == 38.5
        assert ts.tree.lon == 16.1
        assert ts.h_measured is True
        assert Tree.objects.count() == n_trees_before + 1

    def test_unstructured_tree_same_date_reuses_sample_but_not_tree(
            self, writer_client, parcels, species,
    ):
        survey = Survey.objects.create(name='Free tree survey')
        base = {
            FIELD_SURVEY_ID: str(survey.id),
            FIELD_PARCEL_ID: str(parcels[0].id),
            FIELD_SPECIES_ID: str(species[0].id),
            FIELD_DATE: '2026-07-17',
            FIELD_D_CM: '30', FIELD_H_M: '20.5',
            'l10_mm': '12', 'volume_m3': '0.7022', FIELD_MASS_Q: '6.32',
            FIELD_HIGHFOREST: 'true',
        }

        first = self._post(writer_client, {**base, FIELD_NUMBER: '1'})
        second = self._post(writer_client, {**base, FIELD_NUMBER: '2'})

        assert first.status_code == 200, first.content
        assert second.status_code == 200, second.content
        assert Sample.objects.filter(survey=survey, sample_area__isnull=True).count() == 1
        samples = TreeSample.objects.filter(sample__survey=survey).order_by(FIELD_NUMBER)
        assert samples.count() == 2
        assert samples[0].sample_id == samples[1].sample_id
        assert samples[0].tree_id != samples[1].tree_id

    def test_create_rejects_malformed_parent_ids(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: 'not-a-survey',
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '42',
            FIELD_D_CM: '30', FIELD_H_M: '20.5', 'l10_mm': '12',
            FIELD_HIGHFOREST: 'true',
        })

        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR

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
            lat=0.0, lon=0.0, r_m=12,
        )
        s = sample_setup
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(other_area.id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '1', FIELD_D_CM: '30', FIELD_H_M: '20', 'l10_mm': '0',
            'volume_m3': '0.5', FIELD_MASS_Q: '4.7',
            FIELD_HIGHFOREST: 'true',
        })
        assert resp.status_code == 400
        assert 'griglia' in resp.json()[MESSAGE].lower()

    def test_create_rejects_zero_diameter(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '1', FIELD_D_CM: '0', FIELD_H_M: '20', 'l10_mm': '0',
            'volume_m3': '0', FIELD_MASS_Q: '0', FIELD_HIGHFOREST: 'true',
        })
        assert resp.status_code == 400

    def test_reader_cannot_save(self, reader_client, sample_setup):
        s = sample_setup
        resp = self._post(reader_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '1', FIELD_D_CM: '30', FIELD_H_M: '20',
            'l10_mm': '0', 'volume_m3': '0.5', FIELD_MASS_Q: '4.7',
            FIELD_HIGHFOREST: 'true',
        })
        assert resp.status_code == 403

    def test_rejects_duplicate_number_in_sample(self, writer_client, sample_setup):
        """Spec: within a single Sample, tree numbers must be unique."""
        s = sample_setup
        # The sample fixture already has tree with number=1.
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '1', FIELD_D_CM: '40', FIELD_H_M: '25',
            'l10_mm': '0', 'volume_m3': '0.8', FIELD_MASS_Q: '5.7',
            FIELD_HIGHFOREST: 'true',
        })
        assert resp.status_code == 400
        assert 'già utilizzato' in resp.json()[MESSAGE]

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
            FIELD_SURVEY_ID: str(second_survey.id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_TREE_PICK: str(s['tree'].id),
            FIELD_NUMBER: '1',         # propagated from the existing tree
            FIELD_D_CM: '35', FIELD_H_M: '21', 'l10_mm': '0',
            'volume_m3': '0.9', FIELD_MASS_Q: '7.1',
        })
        assert resp.status_code == 200, resp.content

        # One new TreeSample, zero new Trees.
        assert Tree.objects.count() == n_trees_before
        assert TreeSample.objects.count() == n_ts_before + 1
        ts = TreeSample.objects.get(id=resp.json()[ROW_ID])
        assert ts.tree_id == s['tree'].id
        assert ts.number == 1                         # propagated
        assert ts.sample.survey_id == second_survey.id
        assert ts.d_cm == 35                          # new measurement

    def test_existing_fustaia_tree_already_in_sample_has_clear_error(
        self, writer_client, sample_setup,
    ):
        s = sample_setup
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_TREE_PICK: str(s['tree'].id),
            FIELD_NUMBER: '1',
            FIELD_D_CM: '35', FIELD_H_M: '21', 'l10_mm': '0',
            'volume_m3': '0.9', FIELD_MASS_Q: '7.1',
        })
        assert resp.status_code == 400
        assert S.ERR_TREE_ALREADY_IN_SAMPLE.format(1) in resp.json()[MESSAGE]

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
            lat=0.0, lon=0.0, r_m=12,
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
            sample=other_sample, tree=other_tree, parcel=other_parcel,
            shoot=0, standard=False,
            number=7, d_cm=50, h_m=Decimal('30.00'), l10_mm=0,
            volume_m3=Decimal('1.0'), mass_q=Decimal('9.0'),
        )

        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),    # ours, not other_area
            FIELD_TREE_PICK: str(other_tree.id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '7', FIELD_D_CM: '40', FIELD_H_M: '20', 'l10_mm': '0',
            'volume_m3': '0.5', FIELD_MASS_Q: '4.0',
            FIELD_HIGHFOREST: 'true',
        })
        assert resp.status_code == 400

    # --- Date editing (round 4) --------------------------------------

    def _save_payload(self, s, number, date_str):
        """Common scaffolding for tree-save tests below."""
        return {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: str(number),
            FIELD_D_CM: '30', FIELD_H_M: '20', 'l10_mm': '0',
            'volume_m3': '0.5', FIELD_MASS_Q: '4.7',
            FIELD_HIGHFOREST: 'true',
            FIELD_DATE: date_str,
        }

    def test_create_uses_user_date_on_new_sample(
        self, writer_client, sample_setup, regions, eclasses,
    ):
        """A tree saved on a fresh (survey, area) creates the parent
        Sample with the user-chosen date — replaces what the deleted
        sample_date_save_view used to cover."""
        s = sample_setup
        new_parcel = Parcel.objects.create(
            name='2', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        new_area = SampleArea.objects.create(
            sample_grid=s['grid'], parcel=new_parcel, number='2',
            lat=0.0, lon=0.0, r_m=12,
        )
        assert not Sample.objects.filter(
            survey=s['survey'], sample_area=new_area,
        ).exists()
        payload = self._save_payload(s, 1, '2025-03-10')
        payload[FIELD_SAMPLE_AREA_ID] = str(new_area.id)
        resp = self._post(writer_client, payload)
        assert resp.status_code == 200, resp.content
        new_sample = Sample.objects.get(
            survey=s['survey'], sample_area=new_area,
        )
        assert new_sample.date.isoformat() == '2025-03-10'

    def test_create_rejects_sample_date_change_when_sample_exists(
        self, writer_client, sample_setup,
    ):
        s = sample_setup
        n_before = TreeSample.objects.filter(sample=s['sample']).count()
        resp = self._post(writer_client, self._save_payload(s, 99, '2025-04-01'))
        assert resp.status_code == 400
        assert S.ERR_SAMPLE_DATE_CONFLICT.format(
            s['area'].parcel.region.name, s['area'].parcel.name,
            s['area'].number, '2024-09-15',
        ) in resp.json()[MESSAGE]
        s['sample'].refresh_from_db()
        assert s['sample'].date.isoformat() == '2024-09-15'
        assert TreeSample.objects.filter(sample=s['sample']).count() == n_before

    def test_edit_rejects_sample_date_change(self, writer_client, sample_setup):
        s = sample_setup
        ts = TreeSample.objects.get(sample=s['sample'], number=1)
        payload = self._save_payload(s, 1, '2025-05-20')
        payload[ROW_ID] = str(ts.id)
        payload[VERSION] = ts.version
        resp = self._post(writer_client, payload)
        assert resp.status_code == 400
        assert S.ERR_SAMPLE_DATE_CONFLICT.format(
            s['area'].parcel.region.name, s['area'].parcel.name,
            s['area'].number, '2024-09-15',
        ) in resp.json()[MESSAGE]
        s['sample'].refresh_from_db()
        assert s['sample'].date.isoformat() == '2024-09-15'

    def test_edit_stale_version_conflicts(self, writer_client, sample_setup):
        s = sample_setup
        ts = TreeSample.objects.get(sample=s['sample'], number=1)
        payload = self._save_payload(s, 1, '2024-09-15')
        payload[ROW_ID] = str(ts.id)
        payload[VERSION] = ts.version + 1

        resp = self._post(writer_client, payload)

        assert resp.status_code == 400
        data = resp.json()
        assert data[STATUS] == STATUS_CONFLICT
        assert data[DATA_ID] == f'sampled_trees_{s["survey"].id}'
        assert data[ROW_ID] == ts.id
        assert HTML in data
        ts.refresh_from_db()
        assert ts.d_cm == 30

    def test_rejects_invalid_date(self, writer_client, sample_setup):
        payload = self._save_payload(sample_setup, 1, 'not-a-date')
        resp = self._post(writer_client, payload)
        assert resp.status_code == 400
        assert 'Data' in resp.json()[MESSAGE]

    def test_rejects_zero_diameter_or_height(self, writer_client, sample_setup):
        """A measured (fustaia) tree needs D and h > 0."""
        for field in (FIELD_D_CM, FIELD_H_M):
            payload = self._save_payload(sample_setup, 1, '2025-03-10')
            payload[field] = '0'
            resp = self._post(writer_client, payload)
            assert resp.status_code == 400, (field, resp.content)

    def test_response_sample_patch_reflects_current_sample(
        self, writer_client, sample_setup,
    ):
        """The generic patches envelope carries the current sample row
        for client cache updates."""
        from apps.base.digests import build_sample_record
        s = sample_setup
        resp = self._post(writer_client, self._save_payload(s, 50, '2024-09-15'))
        assert resp.status_code == 200, resp.content
        s['sample'].refresh_from_db()
        sample_patch = _patch(resp.json(), 'samples', s['sample'].id)
        assert sample_patch[RECORD] == build_sample_record(s['sample'])


class TestTreeFormPriorTrees:
    """Form GET reflects the prior-trees pulldown contents."""

    def test_lists_eligible_prior_trees_with_survey_and_date(
        self, writer_client, sample_setup,
    ):
        s = sample_setup
        second_survey = Survey.objects.create(
            name='Second campaign', sample_grid=s['grid'],
        )
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={second_survey.id}'
            f'&area={s["area"].id}'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert 'id="id_tree_pick"' in html
        assert '+ nuovo albero' in html
        assert 'n.1' in html
        assert s['survey'].name in html
        assert s['sample'].date.isoformat() in html
        # next_number = max(existing)+1 = 2
        assert 'data-next="2"' in html

    def test_omits_tree_numbers_already_in_active_sample(
        self, writer_client, sample_setup, species,
    ):
        s = sample_setup
        other_survey = Survey.objects.create(
            name='Other historical survey', sample_grid=s['grid'],
        )
        other_sample = Sample.objects.create(
            sample_area=s['area'], survey=other_survey, date=date(2025, 1, 10),
        )
        other_tree = Tree.objects.create(
            species=species[0], parcel=s['area'].parcel, preserved=False,
            coppice=False,
        )
        TreeSample.objects.create(
            sample=other_sample, tree=other_tree, parcel=s['area'].parcel,
            shoot=0, standard=False,
            number=1, d_cm=36, h_m=Decimal('22.00'), l10_mm=0,
        )

        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}'
            f'&area={s["area"].id}'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert '+ nuovo albero' in html
        assert 'data-number="1"' not in html
        assert 'Other historical survey' not in html
        # The next new-tree number still considers trees already in the sample.
        assert 'data-next="2"' in html

    def test_ceduo_default_on_for_coppice_parcel(
        self, writer_client, sample_setup, regions, eclasses,
    ):
        """For parcels whose eclass is coppice, the Ceduo checkbox
        defaults to CHECKED (spec §"Manual tree + sample entry":
        "Defaults to fustaia, except in parcels whose eclass.coppice =
        true, where it defaults to ceduo")."""
        coppice_eclass = next(e for e in eclasses if e.coppice)
        coppice_parcel = Parcel.objects.create(
            name='99', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'), intervention_interval=18,
            standards_per_ha=75,
        )
        coppice_area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=coppice_parcel,
            number='9', lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={sample_setup["survey"].id}'
            f'&area={coppice_area.id}'
        )
        html = resp.json()[HTML]
        idx = html.find('id="tf-ceduo"')
        assert idx >= 0
        tag = html[max(0, idx - 200):idx + 200]
        assert 'checked' in tag

    def test_ceduo_default_off_for_non_coppice_parcel(
        self, writer_client, sample_setup,
    ):
        """Non-coppice areas default Ceduo=off (the common case).
        Regression: the existing fixture's parcel uses non-coppice
        eclass A."""
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={sample_setup["survey"].id}'
            f'&area={sample_setup["area"].id}'
        )
        html = resp.json()[HTML]
        idx = html.find('id="tf-ceduo"')
        assert idx >= 0
        tag = html[max(0, idx - 200):idx + 200]
        assert 'checked' not in tag

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
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}'
            f'&area={empty_area.id}'
        )
        html = resp.json()[HTML]
        assert '+ nuovo albero' in html
        assert 'data-next="1"' in html
        # `data-number` is only emitted on prior-tree options.
        assert 'data-number=' not in html

    def test_coppice_tree_listed_with_next_shoot(
        self, writer_client, sample_setup, species,
    ):
        """A coppice tree with shoots [1,2] in this area must appear in
        the pulldown labelled "ceduo" and carry data-next-shoot=3."""
        s = sample_setup
        coppice_tree = Tree.objects.create(
            species=species[0], parcel=s['area'].parcel,
            preserved=False, coppice=True,
        )
        for sh in (1, 2):
            TreeSample.objects.create(
                sample=s['sample'], tree=coppice_tree, parcel=s['area'].parcel,
                shoot=sh, standard=(sh == 2), number=7,
                d_cm=5 + sh, h_m=Decimal('8.00'), l10_mm=0,
            )
        second_survey = Survey.objects.create(
            name='Second campaign for coppice', sample_grid=s['grid'],
        )
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={second_survey.id}'
            f'&area={s["area"].id}'
        )
        html = resp.json()[HTML]
        assert 'data-next-shoot="3"' in html
        assert 'ceduo' in html
        assert 'n.7' in html
        assert s['survey'].name in html
        assert s['sample'].date.isoformat() in html


class TestTreeSaveCoppice:
    """Coppice (per-shoot) tree+sample creation via `shoots` JSON."""

    @staticmethod
    def _post(client, body):
        import json
        return client.post(
            '/api/campionamenti/tree/save/',
            data=json.dumps(body), content_type='application/json',
        )

    def test_create_coppice_tree_with_three_shoots(
        self, writer_client, sample_setup, species, regions, eclasses,
    ):
        """One Tree + N TreeSamples, all sharing the same number / tree."""
        # Use a fresh coppice area so the existing fustaia tree (number=1)
        # doesn't collide with the coppice tree's number.
        coppice_eclass = next(e for e in eclasses if e.coppice)
        parcel = Parcel.objects.create(
            name='c', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'), intervention_interval=18,
            standards_per_ha=75,
        )
        area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=parcel,
            number='1', lat=0.0, lon=0.0, r_m=12,
        )
        n_trees_before = Tree.objects.count()
        n_ts_before = TreeSample.objects.count()
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(sample_setup['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(area.id),
            FIELD_SPECIES_ID: str(species[1].id),     # Castagno
            FIELD_NUMBER: '1', FIELD_HIGHFOREST: 'false',
            'shoots': json.dumps([
                {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 5,
                 FIELD_H_M: '8.0', 'l10_mm': 0},
                {FIELD_SHOOT: 2, FIELD_STANDARD: True,  FIELD_D_CM: 7,
                 FIELD_H_M: '9.5', 'l10_mm': 12},
                {FIELD_SHOOT: 3, FIELD_STANDARD: False, FIELD_D_CM: 4,
                 FIELD_H_M: '7.0', 'l10_mm': 0},
            ]),
            FIELD_LAT: '0', FIELD_LON: '0', FIELD_PRESERVED: '',
        })
        assert resp.status_code == 200, resp.content
        # One new Tree (coppice=True), three new TreeSamples.
        assert Tree.objects.count() == n_trees_before + 1
        assert TreeSample.objects.count() == n_ts_before + 3
        new_tree = Tree.objects.order_by('-id').first()
        assert new_tree.coppice is True
        shoots = list(TreeSample.objects.filter(tree=new_tree)
                      .order_by('shoot').values_list('shoot', 'standard',
                                                     'd_cm', 'volume_m3',
                                                     'mass_q'))
        assert shoots == [
            (1, False, 5, None, None),
            (2, True,  7, None, None),
            (3, False, 4, None, None),
        ]
        # All TreeSamples share number=1.
        assert TreeSample.objects.filter(
            tree=new_tree,
        ).values_list('number', flat=True).distinct().count() == 1

    def test_coppice_shoot_height_locale_and_garbage(
        self, writer_client, sample_setup, species, regions, eclasses,
    ):
        """Per-shoot validation: a locale-comma height parses; garbage height
        and zero diameter are clean 400s (regression for the unimported
        InvalidOperation that 500'd the height path)."""
        coppice_eclass = next(e for e in eclasses if e.coppice)
        parcel = Parcel.objects.create(
            name='cc', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'), intervention_interval=18,
            standards_per_ha=75,
        )
        area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=parcel,
            number='1', lat=0.0, lon=0.0, r_m=12,
        )
        base = {
            FIELD_SURVEY_ID: str(sample_setup['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(area.id),
            FIELD_SPECIES_ID: str(species[1].id),
            FIELD_NUMBER: '1', FIELD_HIGHFOREST: 'false',
            FIELD_LAT: '0', FIELD_LON: '0', FIELD_PRESERVED: '',
        }
        ok = self._post(writer_client, {**base, 'shoots': json.dumps([
            {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 5,
             FIELD_H_M: '8,5', 'l10_mm': 0},
        ])})
        assert ok.status_code == 200, ok.content
        assert TreeSample.objects.order_by('-id').first().h_m == Decimal('8.50')

        bad = self._post(writer_client, {**base, 'shoots': json.dumps([
            {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 5,
             FIELD_H_M: 'abc', 'l10_mm': 0},
        ])})
        assert bad.status_code == 400  # clean validation error, never a 500

        bad_d = self._post(writer_client, {**base, 'shoots': json.dumps([
            {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 0,
             FIELD_H_M: '8,0', 'l10_mm': 0},
        ])})
        assert bad_d.status_code == 400  # a coppice shoot's diameter must be > 0

    def test_coppice_requires_at_least_one_shoot(
        self, writer_client, sample_setup, regions, eclasses,
    ):
        coppice_eclass = next(e for e in eclasses if e.coppice)
        parcel = Parcel.objects.create(
            name='c2', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'), intervention_interval=18,
            standards_per_ha=75,
        )
        area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=parcel, number='1',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(sample_setup['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(area.id),
            FIELD_SPECIES_ID: str(sample_setup['tree'].species_id),
            FIELD_NUMBER: '1', FIELD_HIGHFOREST: 'false',
            'shoots': json.dumps([]),
        })
        assert resp.status_code == 400
        assert 'pollone' in resp.json()[MESSAGE].lower()

    def test_coppice_existing_tree_appends_shoots(
        self, writer_client, sample_setup, species,
    ):
        """Picking an existing coppice tree from the pulldown adds new
        shoots under the same tree_id (no new Tree row)."""
        s = sample_setup
        existing = Tree.objects.create(
            species=species[1], parcel=s['area'].parcel,
            preserved=False, coppice=True,
        )
        TreeSample.objects.create(
            sample=s['sample'], tree=existing, parcel=s['area'].parcel, shoot=1,
            standard=False, number=42,
            d_cm=5, h_m=Decimal('8.00'), l10_mm=0,
        )
        n_trees_before = Tree.objects.count()
        # Add a second sample in a new survey so we can append shoots.
        second_survey = Survey.objects.create(
            name='Second campaign', sample_grid=s['grid'],
        )
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(second_survey.id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_TREE_PICK: str(existing.id),
            FIELD_SPECIES_ID: str(existing.species_id),
            FIELD_NUMBER: '42', FIELD_HIGHFOREST: 'false',
            'shoots': json.dumps([
                {FIELD_SHOOT: 2, FIELD_STANDARD: False, FIELD_D_CM: 6,
                 FIELD_H_M: '8.5', 'l10_mm': 0},
                {FIELD_SHOOT: 3, FIELD_STANDARD: True,  FIELD_D_CM: 8,
                 FIELD_H_M: '9.0', 'l10_mm': 14},
            ]),
        })
        assert resp.status_code == 200, resp.content
        assert Tree.objects.count() == n_trees_before    # no new tree
        new_shoots = TreeSample.objects.filter(
            tree=existing, sample__survey=second_survey,
        ).order_by('shoot')
        assert [ts.shoot for ts in new_shoots] == [2, 3]
        assert [ts.standard for ts in new_shoots] == [False, True]

    def test_coppice_shoot_duplicate_in_sample_refused(
        self, writer_client, sample_setup, species,
    ):
        """Trying to add a pollone that already exists on the same
        (sample, tree) raises a clean validation error."""
        s = sample_setup
        existing = Tree.objects.create(
            species=species[1], parcel=s['area'].parcel,
            preserved=False, coppice=True,
        )
        TreeSample.objects.create(
            sample=s['sample'], tree=existing, parcel=s['area'].parcel,
            shoot=1, standard=False,
            number=99, d_cm=5, h_m=Decimal('8.00'), l10_mm=0,
        )
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_TREE_PICK: str(existing.id),
            FIELD_SPECIES_ID: str(existing.species_id),
            FIELD_NUMBER: '99', FIELD_HIGHFOREST: 'false',
            'shoots': json.dumps([
                {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 6,
                 FIELD_H_M: '8.5', 'l10_mm': 0},     # collides
            ]),
        })
        assert resp.status_code == 400
        assert S.ERR_TREE_ALREADY_IN_SAMPLE.format(99) in resp.json()[MESSAGE]

    def test_edit_coppice_single_shoot(self, writer_client, sample_setup,
                                       species):
        """Editing a single coppice TreeSample updates only that row."""
        s = sample_setup
        tree = Tree.objects.create(
            species=species[1], parcel=s['area'].parcel,
            preserved=False, coppice=True,
        )
        ts1 = TreeSample.objects.create(
            sample=s['sample'], tree=tree, parcel=s['area'].parcel,
            shoot=1, standard=False,
            number=15, d_cm=5, h_m=Decimal('8.00'), l10_mm=0,
        )
        ts2 = TreeSample.objects.create(
            sample=s['sample'], tree=tree, parcel=s['area'].parcel,
            shoot=2, standard=True,
            number=15, d_cm=6, h_m=Decimal('8.50'), l10_mm=12,
        )
        n_ts_before = TreeSample.objects.count()
        resp = self._post(writer_client, {
            ROW_ID: str(ts1.id),
            VERSION: ts1.version,
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_TREE_PICK: str(tree.id),
            FIELD_SPECIES_ID: str(species[0].id),         # changed
            FIELD_NUMBER: '15', FIELD_HIGHFOREST: 'false',
            'shoots': json.dumps([
                {FIELD_SHOOT: 1, FIELD_STANDARD: True, FIELD_D_CM: 9,
                 FIELD_H_M: '10.0', 'l10_mm': 5},
            ]),
            FIELD_PRESERVED: '',
            FIELD_LAT: '0', FIELD_LON: '0',
        })
        assert resp.status_code == 200, resp.content
        assert TreeSample.objects.count() == n_ts_before
        ts1.refresh_from_db()
        ts2.refresh_from_db()
        # Edited row has the new measurements.
        assert ts1.d_cm == 9 and ts1.standard is True
        # Sibling shoot is untouched.
        assert ts2.d_cm == 6 and ts2.standard is True
        # The shared Tree's species was updated by the edit.
        tree.refresh_from_db()
        assert tree.species_id == species[0].id


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
        html = resp.json()[HTML]
        # Modal carries two creation paths (CSV import moved to pencil modal).
        assert 'data-path="empty"' in html
        assert 'data-path="auto"' in html
        # Default-active body is the empty-grid create form.
        assert 'campionamenti-grid-form-empty' in html

    def test_reader_form_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/campionamenti/grid/form/')
        assert resp.status_code == 403

    def test_create_empty_grid(self, writer_client, db):
        from apps.base.models import SampleGrid
        resp = self._post(writer_client, {
            FIELD_NAME: 'Griglia di prova', FIELD_DESCRIPTION: 'desc',
        })
        assert resp.status_code == 200
        data = resp.json()
        g = SampleGrid.objects.get(id=data[ROW_ID])
        assert g.name == 'Griglia di prova'

    def test_name_required(self, writer_client, db):
        resp = self._post(writer_client, {FIELD_NAME: '', FIELD_DESCRIPTION: ''})
        assert resp.status_code == 400

    def test_name_duplicate_rejected(self, writer_client, db):
        from apps.base.models import SampleGrid
        SampleGrid.objects.create(name='Dup')
        resp = self._post(writer_client, {FIELD_NAME: 'Dup', FIELD_DESCRIPTION: ''})
        assert resp.status_code == 400

    def test_reader_forbidden(self, reader_client, db):
        resp = self._post(reader_client, {FIELD_NAME: 'X'})
        assert resp.status_code == 403


class TestAreaCRUD:
    """Section 1 SampleArea add/edit/delete."""

    @staticmethod
    def _post(client, url, body):
        import json
        return client.post(
            url, data=json.dumps(body), content_type='application/json',
        )

    def test_form_add_requires_grid(self, writer_client, db):
        resp = writer_client.get('/api/campionamenti/area/form/')
        assert resp.status_code == 404

    def test_form_add_rejects_malformed_grid(self, writer_client, db):
        resp = writer_client.get('/api/campionamenti/area/form/?grid=abc')
        assert resp.status_code == 404

    def test_reader_form_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/campionamenti/area/form/')
        assert resp.status_code == 403

    def test_form_add_renders(self, writer_client, sample_setup):
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
            f'&lat=38.5&lon=16.1'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert '<form' in html
        assert 'value="38,50000"' in html          # it-locale: comma decimal

    def test_add_form_rounds_click_coords_to_5dp(self, writer_client, sample_setup):
        """Map-click passes full-precision lat/lon; the form must display them
        at 5 dp (coordinates are 5 dp everywhere — spec)."""
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
            f'&lat=38.123456789&lon=16.987654321'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert 'value="38,12346"' in html          # it-locale: comma, 5 dp
        assert 'value="16,98765"' in html
        assert '38.123456789' not in html        # full precision must not leak

    def test_form_edit_renders(self, writer_client, sample_setup):
        s = sample_setup
        resp = writer_client.get(f'/api/campionamenti/area/form/{s["area"].id}/')
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert s['area'].number in html

    def test_create_area(self, writer_client, sample_setup):
        s = sample_setup
        n_before = SampleArea.objects.filter(sample_grid=s['grid']).count()
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: s['area'].parcel_id,
            FIELD_NUMBER: '42',
            FIELD_LAT: '38,6', FIELD_LON: '16,2',       # Italian comma input
            FIELD_ALTITUDE_M: '500',
            FIELD_R_M: '15', FIELD_NOTE: 'test',
        })
        assert resp.status_code == 200, resp.content
        assert SampleArea.objects.filter(sample_grid=s['grid']).count() == n_before + 1
        # The comma input is parsed back to a canonical float (not 386).
        area = SampleArea.objects.filter(sample_grid=s['grid']).latest('id')
        assert (area.lat, area.lon) == (38.6, 16.2)

    def test_update_area(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            ROW_ID: s['area'].id,
            VERSION: s['area'].version,
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: s['area'].parcel_id,
            FIELD_NUMBER: 'edited',
            FIELD_LAT: '38.6', FIELD_LON: '16.2',
            FIELD_R_M: '14',
        })
        assert resp.status_code == 200
        s['area'].refresh_from_db()
        assert s['area'].number == 'edited'
        assert s['area'].r_m == 14

    def test_update_area_stale_version_conflicts(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            ROW_ID: s['area'].id,
            VERSION: s['area'].version + 1,
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: s['area'].parcel_id,
            FIELD_NUMBER: 'edited',
            FIELD_LAT: '38.6', FIELD_LON: '16.2', FIELD_R_M: '14',
        })

        assert resp.status_code == 400
        data = resp.json()
        assert data[STATUS] == STATUS_CONFLICT
        assert data[DATA_ID] == 'sample_areas'
        assert data[ROW_ID] == s['area'].id
        s['area'].refresh_from_db()
        assert s['area'].number == '1'

    def test_number_required(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: s['area'].parcel_id,
            FIELD_NUMBER: '',
            FIELD_LAT: '38.5', FIELD_LON: '16.1',
            FIELD_R_M: '12',
        })
        assert resp.status_code == 400

    def test_duplicate_number_same_region_other_parcel_rejected(
        self, writer_client, sample_setup, eclasses,
    ):
        """Area numbers are unique per region: '1' (used by the fixture parcel)
        can't be reused by another parcel in the same region."""
        s = sample_setup
        region = s['area'].parcel.region
        other = Parcel.objects.create(
            name='1bis', region=region, eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: other.id,
            FIELD_NUMBER: '1',
            FIELD_LAT: '38.6', FIELD_LON: '16.2', FIELD_R_M: '12',
        })
        assert resp.status_code == 400
        assert SampleArea.objects.filter(parcel=other).count() == 0

    def test_same_number_different_region_ok(
        self, writer_client, sample_setup, regions, eclasses,
    ):
        s = sample_setup
        other = Parcel.objects.create(
            name='1', region=regions[1], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: other.id,
            FIELD_NUMBER: '1',
            FIELD_LAT: '38.6', FIELD_LON: '16.2', FIELD_R_M: '12',
        })
        assert resp.status_code == 200, resp.content
        assert SampleArea.objects.filter(parcel=other, number='1').count() == 1

    def test_edit_area_keeps_own_number(self, writer_client, sample_setup):
        """Editing an area without changing its number must not trip the
        per-region duplicate check against itself."""
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            ROW_ID: s['area'].id,
            VERSION: s['area'].version,
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: s['area'].parcel_id,
            FIELD_NUMBER: s['area'].number,          # unchanged
            FIELD_LAT: '38.7', FIELD_LON: '16.3', FIELD_R_M: '13',
        })
        assert resp.status_code == 200, resp.content

    def test_delete_unused_area(self, writer_client, sample_setup,
                                regions, eclasses):
        """An area with no samples can be deleted."""
        s = sample_setup
        new_parcel = Parcel.objects.create(
            name='2', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        unused = SampleArea.objects.create(
            sample_grid=s['grid'], parcel=new_parcel, number='3',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.post(
            f'/api/campionamenti/area/delete/{unused.id}/',
            data=json.dumps({VERSION: unused.version}),
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert not SampleArea.objects.filter(id=unused.id).exists()

    def test_delete_saves_nonce(self, writer_client, sample_setup, regions, eclasses):
        s = sample_setup
        new_parcel = Parcel.objects.create(
            name='2', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        unused = SampleArea.objects.create(
            sample_grid=s['grid'], parcel=new_parcel, number='3',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.post(
            f'/api/campionamenti/area/delete/{unused.id}/',
            data=json.dumps({
                VERSION: unused.version,
                FIELD_NONCE: 'area-delete-nonce',
            }),
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='area-delete-nonce').exists()

    def test_delete_stale_version_conflicts(self, writer_client, sample_setup,
                                            regions, eclasses):
        s = sample_setup
        new_parcel = Parcel.objects.create(
            name='2', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        unused = SampleArea.objects.create(
            sample_grid=s['grid'], parcel=new_parcel, number='3',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.post(
            f'/api/campionamenti/area/delete/{unused.id}/',
            data=json.dumps({VERSION: unused.version + 1}),
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert SampleArea.objects.filter(id=unused.id).exists()

    def test_delete_in_use_refused(self, writer_client, sample_setup):
        """An area referenced by any Sample is protected from delete."""
        s = sample_setup
        resp = writer_client.post(
            f'/api/campionamenti/area/delete/{s["area"].id}/',
            data=json.dumps({VERSION: s['area'].version}),
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert SampleArea.objects.filter(id=s['area'].id).exists()

    def test_reader_forbidden(self, reader_client, sample_setup):
        s = sample_setup
        resp = self._post(reader_client, '/api/campionamenti/area/save/', {
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: s['area'].parcel_id,
            FIELD_NUMBER: '99',
            FIELD_LAT: '38.5', FIELD_LON: '16.1', FIELD_R_M: '12',
        })
        assert resp.status_code == 403


class TestAreaFormNextNumber:
    """Add-area form auto-suggests the next free integer area number,
    scoped to (grid, REGION) — area numbers are unique per region, not per
    parcel.  Per docs/page-campionamenti.md §Section 1."""

    @staticmethod
    def _parcel_option(html, parcel_id):
        """The <option> open tag for a given parcel id.  Parcel options carry
        data-region-id, which disambiguates them from region options that may
        share the same numeric id."""
        anchor = html.find(f'value="{parcel_id}" data-region-id')
        assert anchor >= 0, html
        return html[anchor:html.find('>', anchor) + 1]

    def test_suggests_max_integer_plus_one(self, writer_client, sample_setup):
        s = sample_setup
        parcel = s['area'].parcel          # fixture already has area number '1'
        SampleArea.objects.create(
            sample_grid=s['grid'], parcel=parcel, number='2',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
        )
        assert resp.status_code == 200
        assert 'data-next-number="3"' in self._parcel_option(
            resp.json()[HTML], parcel.id,
        )

    def test_region_scoped_across_parcels(self, writer_client, sample_setup,
                                          eclasses):
        """Numbers are unique per region: an area in ANOTHER parcel of the same
        region bumps the suggestion for every parcel in that region."""
        s = sample_setup
        region = s['area'].parcel.region          # already holds area '1'
        other_parcel = Parcel.objects.create(
            name='1bis', region=region, eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        SampleArea.objects.create(
            sample_grid=s['grid'], parcel=other_parcel, number='5',
            lat=0.0, lon=0.0, r_m=12,
        )
        html = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
        ).json()[HTML]
        # Region max is 5 → every parcel in the region suggests 6.
        assert 'data-next-number="6"' in self._parcel_option(
            html, s['area'].parcel_id,
        )
        assert 'data-next-number="6"' in self._parcel_option(html, other_parcel.id)

    def test_unused_region_starts_at_one(self, writer_client, sample_setup,
                                         regions, eclasses):
        s = sample_setup
        fresh = Parcel.objects.create(          # regions[1] holds no areas
            name='77', region=regions[1], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
        )
        assert 'data-next-number="1"' in self._parcel_option(
            resp.json()[HTML], fresh.id,
        )

    def test_non_integer_numbers_ignored(self, writer_client, sample_setup,
                                         regions, eclasses):
        """Non-integer labels like 'C1' don't count, so a region holding only
        'C1' still starts at 1."""
        s = sample_setup
        coppice = next(e for e in eclasses if e.coppice)
        parcel = Parcel.objects.create(          # regions[2], isolated
            name='88', region=regions[2], eclass=coppice,
            area_ha=Decimal('1.0'),
            intervention_interval=18, standards_per_ha=75,
        )
        SampleArea.objects.create(
            sample_grid=s['grid'], parcel=parcel, number='C1',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
        )
        assert 'data-next-number="1"' in self._parcel_option(
            resp.json()[HTML], parcel.id,
        )

    def test_scoped_to_grid(self, writer_client, sample_setup):
        """Areas in other grids don't affect this grid's suggestion."""
        s = sample_setup
        parcel = s['area'].parcel          # number '1' in this grid
        other_grid = SampleGrid.objects.create(name='Other grid')
        SampleArea.objects.create(
            sample_grid=other_grid, parcel=parcel, number='9',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
        )
        # Max integer in THIS grid for the region is 1 → suggest 2, not 10.
        assert 'data-next-number="2"' in self._parcel_option(
            resp.json()[HTML], parcel.id,
        )


class TestAreaFormPreselect:
    """Click-to-create passes the clicked parcel's compresa+particella; the
    form pre-selects that region+parcel (docs/page-campionamenti.md
    §Section 1)."""

    @staticmethod
    def _is_selected(html, select_id, value):
        """True if the <option value=`value`> inside <select id=`select_id`>
        carries the `selected` attribute."""
        sidx = html.find(f'id="{select_id}"')
        assert sidx >= 0, html
        block = html[sidx:html.find('</select>', sidx)]
        vidx = block.find(f'value="{value}"')
        if vidx < 0:
            return False
        tag = block[block.rfind('<option', 0, vidx):block.find('>', vidx)]
        return 'selected' in tag

    def test_preselects_parcel_from_names(self, writer_client, sample_setup):
        s = sample_setup
        parcel = s['area'].parcel
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
            f'&compresa={parcel.region.name}&particella={parcel.name}'
        )
        html = resp.json()[HTML]
        assert self._is_selected(html, 'id_area_parcel', parcel.id)
        assert self._is_selected(html, 'id_area_region', parcel.region_id)

    def test_unknown_names_no_preselect(self, writer_client, sample_setup):
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={sample_setup["grid"].id}'
            f'&compresa=Nowhere&particella=999'
        )
        assert resp.status_code == 200          # graceful: just no pre-selection
        assert not self._is_selected(
            resp.json()[HTML], 'id_area_parcel', sample_setup['area'].parcel_id,
        )


class TestRecordShape:
    """Per CLAUDE.md §"Optimistic table updates" — every write view
    returns a `record` (or `records`) shaped identically to the
    corresponding JSON digest row.  Locks the column-shape contract
    between the digest generators and the write views.
    """

    @staticmethod
    def _post(client, url, body):
        import json
        return client.post(
            url, data=json.dumps(body), content_type='application/json',
        )

    def test_tree_save_record_matches_digest(self, writer_client, sample_setup):
        """build_tree_sample_record output == digest row for the same ts."""
        from apps.base.digests import (
            SAMPLED_TREE_COLUMNS, build_tree_sample_record,
        )
        from apps.base.models import TreeSample
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/tree/save/', {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '42',
            FIELD_D_CM: '30', FIELD_H_M: '20.5', 'l10_mm': '12',
            'volume_m3': '0.7022', FIELD_MASS_Q: '6.32',
            FIELD_HIGHFOREST: 'true', FIELD_PRESERVED: 'true',
            FIELD_LAT: '38.1', FIELD_LON: '16.2',
        })
        assert resp.status_code == 200, resp.content
        payload = resp.json()
        tree_patches = [p for p in payload[PATCHES]
                        if p[DATA_ID] == payload[DATA_ID]]
        assert len(tree_patches) == 1
        record = tree_patches[0][RECORD]
        # Match against the canonical row built from the freshly-saved ts.
        ts = TreeSample.objects.select_related(
            'sample__survey', 'sample__sample_area__parcel__region',
            'tree__species', 'tree__parcel',
        ).get(id=payload[ROW_ID])
        assert record == build_tree_sample_record(ts)
        assert len(record) == len(SAMPLED_TREE_COLUMNS)
        assert record[SAMPLED_TREE_COLUMNS.index(COL_COPPICE)] is False
        _assert_stale(
            DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS,
            DIGEST_PRESERVED_TREES,
        )

    def test_tree_save_coppice_records_match_digest(
        self, writer_client, sample_setup, species, regions, eclasses,
    ):
        """Coppice multi-shoot creates return one record per shoot."""
        from apps.base.digests import (
            SAMPLED_TREE_COLUMNS, build_tree_sample_record,
        )
        from apps.base.models import Parcel, SampleArea, TreeSample
        coppice_eclass = next(e for e in eclasses if e.coppice)
        parcel = Parcel.objects.create(
            name='cs', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'), intervention_interval=18,
            standards_per_ha=75,
        )
        area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=parcel, number='1',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = self._post(writer_client, '/api/campionamenti/tree/save/', {
            FIELD_SURVEY_ID: str(sample_setup['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(area.id),
            FIELD_SPECIES_ID: str(species[1].id),
            FIELD_NUMBER: '1', FIELD_HIGHFOREST: 'false',
            'shoots': json.dumps([
                {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 5, FIELD_H_M: '8.0'},
                {FIELD_SHOOT: 2, FIELD_STANDARD: True,  FIELD_D_CM: 7, FIELD_H_M: '9.0'},
            ]),
            FIELD_LAT: '0', FIELD_LON: '0',
        })
        assert resp.status_code == 200, resp.content
        payload = resp.json()
        tree_patches = [p for p in payload[PATCHES]
                        if p[DATA_ID] == payload[DATA_ID]]
        assert len(tree_patches) == 2
        ids = [p[ROW_ID] for p in tree_patches]
        canonical = {
            ts.id: build_tree_sample_record(ts)
            for ts in TreeSample.objects.filter(id__in=ids).select_related(
                'sample__sample_area__parcel__region',
                'tree__species', 'tree__parcel',
            )
        }
        for patch in tree_patches:
            assert patch[RECORD] == canonical[patch[ROW_ID]]
            assert patch[RECORD][SAMPLED_TREE_COLUMNS.index(COL_COPPICE)] is True

    def test_tree_save_includes_sample_and_survey_patches(
        self, writer_client, sample_setup,
    ):
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/tree/save/', {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '42',
            FIELD_D_CM: '30', FIELD_H_M: '20.5', 'l10_mm': '12',
            'volume_m3': '0.7022', FIELD_MASS_Q: '6.32',
            FIELD_HIGHFOREST: 'true',
        })
        from apps.base.digests import (
            SAMPLE_COLUMNS, SURVEY_COLUMNS, build_sample_record,
            build_survey_record,
        )
        payload = resp.json()
        sample_record = _patch(payload, 'samples', s['sample'].id)[RECORD]
        survey_record = _patch(payload, 'surveys', s['survey'].id)[RECORD]
        assert sample_record[0] == s['sample'].id
        assert len(sample_record) == len(SAMPLE_COLUMNS)
        s['sample'].refresh_from_db()
        assert sample_record == build_sample_record(s['sample'])
        s['survey'].refresh_from_db()
        assert survey_record == build_survey_record(s['survey'])
        assert len(survey_record) == len(SURVEY_COLUMNS)

    def test_area_save_returns_records(self, writer_client, sample_setup):
        from apps.base.digests import (
            build_grid_record, build_sample_area_record, build_survey_record,
        )
        from apps.base.models import SampleArea
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_PARCEL_ID: s['area'].parcel_id,
            FIELD_NUMBER: '7',
            FIELD_LAT: '38.5', FIELD_LON: '16.1', FIELD_R_M: '12',
        })
        assert resp.status_code == 200, resp.content
        payload = resp.json()
        area = SampleArea.objects.select_related(
            'parcel__region',
        ).get(id=payload[ROW_ID])
        assert _patch(payload, 'sample_areas', area.id)[RECORD] == build_sample_area_record(area)
        s['grid'].refresh_from_db()
        assert _patch(payload, 'grids', s['grid'].id)[RECORD] == build_grid_record(s['grid'])
        s['survey'].refresh_from_db()
        assert _patch(payload, 'surveys', s['survey'].id)[RECORD] == build_survey_record(s['survey'])
        _assert_stale(DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS)

    def test_grid_save_returns_record(self, writer_client, db):
        from apps.base.digests import build_grid_record
        from apps.base.models import SampleGrid
        resp = self._post(writer_client, '/api/campionamenti/grid/save/', {
            FIELD_NAME: 'Griglia X', FIELD_DESCRIPTION: 'd',
        })
        payload = resp.json()
        grid = SampleGrid.objects.get(id=payload[ROW_ID])
        assert _patch(payload, 'grids', grid.id)[RECORD] == build_grid_record(grid)

    def test_grid_edit_returns_record(self, writer_client, sample_setup):
        from apps.base.digests import build_grid_record
        s = sample_setup
        resp = self._post(
            writer_client,
            f'/api/campionamenti/grid/edit/{s["grid"].id}/',
            {
                FIELD_NAME: 'Renamed', FIELD_DESCRIPTION: '',
                VERSION: str(s['grid'].version),
            },
        )
        payload = resp.json()
        s['grid'].refresh_from_db()
        assert _patch(payload, 'grids', s['grid'].id)[RECORD] == build_grid_record(s['grid'])

    def test_grid_edit_conflict_returns_current_patch(self, writer_client, sample_setup):
        from apps.base.digests import build_grid_record
        s = sample_setup
        resp = self._post(
            writer_client,
            f'/api/campionamenti/grid/edit/{s["grid"].id}/',
            {FIELD_NAME: 'Renamed', FIELD_DESCRIPTION: '', VERSION: '999'},
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload[STATUS] == STATUS_CONFLICT
        s['grid'].refresh_from_db()
        assert _patch(payload, 'grids', s['grid'].id)[RECORD] == build_grid_record(s['grid'])

    def test_survey_save_returns_record(self, writer_client, sample_setup):
        from apps.base.digests import build_grid_record, build_survey_record
        from apps.base.models import Survey
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/survey/save/', {
            FIELD_NAME: 'New survey', FIELD_SAMPLE_GRID_ID: s['grid'].id,
            FIELD_DESCRIPTION: '',
        })
        payload = resp.json()
        survey = Survey.objects.get(id=payload[ROW_ID])
        assert _patch(payload, 'surveys', survey.id)[RECORD] == build_survey_record(survey)
        s['grid'].refresh_from_db()
        assert _patch(payload, 'grids', s['grid'].id)[RECORD] == build_grid_record(s['grid'])

    def test_survey_edit_returns_record(self, writer_client, sample_setup):
        from apps.base.digests import build_survey_record
        s = sample_setup
        resp = self._post(
            writer_client,
            f'/api/campionamenti/survey/edit/{s["survey"].id}/',
            {
                FIELD_NAME: 'Renamed', FIELD_DESCRIPTION: '',
                VERSION: str(s['survey'].version),
            },
        )
        payload = resp.json()
        s['survey'].refresh_from_db()
        assert _patch(payload, 'surveys', s['survey'].id)[RECORD] == build_survey_record(s['survey'])
        _assert_stale(DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS)

    def test_survey_edit_conflict_returns_current_patch(self, writer_client, sample_setup):
        from apps.base.digests import build_survey_record
        s = sample_setup
        resp = self._post(
            writer_client,
            f'/api/campionamenti/survey/edit/{s["survey"].id}/',
            {FIELD_NAME: 'Renamed', FIELD_DESCRIPTION: '', VERSION: '999'},
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload[STATUS] == STATUS_CONFLICT
        s['survey'].refresh_from_db()
        assert _patch(payload, 'surveys', s['survey'].id)[RECORD] == build_survey_record(s['survey'])

    def test_tree_delete_returns_sample_and_survey_patches(
        self, writer_client, sample_setup,
    ):
        from apps.base.digests import build_sample_record, build_survey_record
        from apps.base.models import TreeSample
        s = sample_setup
        ts_id = TreeSample.objects.first().id
        ts = TreeSample.objects.get(id=ts_id)
        resp = writer_client.post(
            f'/api/campionamenti/tree/delete/{ts_id}/',
            data=json.dumps({VERSION: ts.version}),
            content_type='application/json',
        )
        payload = resp.json()
        assert {DATA_ID: f'sampled_trees_{s["survey"].id}', ROW_ID: ts_id} in payload[DELETES]
        s['sample'].refresh_from_db()
        s['survey'].refresh_from_db()
        assert _patch(payload, 'samples', s['sample'].id)[RECORD] == build_sample_record(s['sample'])
        assert _patch(payload, 'surveys', s['survey'].id)[RECORD] == build_survey_record(s['survey'])
        _assert_stale(DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS)


class TestTreeDelete:
    @staticmethod
    def _post(client, ts_id, body=None):
        ts = TreeSample.objects.filter(id=ts_id).first()
        if body is None:
            body = {VERSION: ts.version if ts else 0}
        return client.post(
            f'/api/campionamenti/tree/delete/{ts_id}/',
            data=json.dumps(body),
            content_type='application/json',
        )

    def test_delete_tree_sample(self, writer_client, sample_setup):
        from apps.base.models import Tree, TreeSample
        s = sample_setup
        ts_id = TreeSample.objects.first().id
        tree_count_before = Tree.objects.count()
        resp = self._post(writer_client, ts_id)
        assert resp.status_code == 200
        # TreeSample gone, but the parent Tree row survives.
        assert not TreeSample.objects.filter(id=ts_id).exists()
        assert Tree.objects.count() == tree_count_before

    def test_delete_saves_nonce(self, writer_client, sample_setup):
        from apps.base.models import TreeSample
        ts_id = TreeSample.objects.first().id
        ts = TreeSample.objects.get(id=ts_id)
        resp = writer_client.post(
            f'/api/campionamenti/tree/delete/{ts_id}/',
            data=json.dumps({
                VERSION: ts.version,
                FIELD_NONCE: 'tree-delete-nonce',
            }),
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='tree-delete-nonce').exists()

    def test_delete_stale_version_conflicts(self, writer_client, sample_setup):
        ts = TreeSample.objects.first()
        resp = self._post(writer_client, ts.id, {VERSION: ts.version + 1})
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert TreeSample.objects.filter(id=ts.id).exists()

    def test_delete_nonexistent(self, writer_client, db):
        resp = self._post(writer_client, 99999)
        assert resp.status_code == 404

    def test_reader_forbidden(self, reader_client, sample_setup):
        ts_id = TreeSample.objects.first().id
        resp = self._post(reader_client, ts_id)
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
            FIELD_NAME: 'Auto grid',
            FIELD_DESCRIPTION: '',
            FIELD_R_M: 12,
            FIELD_POINTS: [
                {'compresa': s['area'].parcel.region.name,
                 'particella': s['area'].parcel.name,
                 FIELD_LAT: 38.5, FIELD_LON: 16.1},
                {'compresa': s['area'].parcel.region.name,
                 'particella': s['area'].parcel.name,
                 FIELD_LAT: 38.51, FIELD_LON: 16.11},
            ],
        })
        assert resp.status_code == 200, resp.content
        data = resp.json()
        grid = SampleGrid.objects.get(id=data[ROW_ID])
        assert grid.name == 'Auto grid'
        # bulk_create made 2 SampleAreas with sequential numbers.
        areas = list(SampleArea.objects.filter(sample_grid=grid).order_by('number'))
        assert len(areas) == 2
        assert {a.number for a in areas} == {'1', '2'}
        assert all(a.r_m == 12 for a in areas)
        # Coordinates round-trip into the lat/lon schema columns.  Regression
        # guard for the client contract: the planner maps geo.js's `lng` to
        # `lon` before posting, and the server reads `lon` (FIELD_LON).
        by_number = {a.number: a for a in areas}
        assert (by_number['1'].lat, by_number['1'].lon) == pytest.approx((38.5, 16.1))
        assert (by_number['2'].lat, by_number['2'].lon) == pytest.approx((38.51, 16.11))

    def test_numbers_restart_per_compresa(
        self, writer_client, sample_setup, regions, eclasses,
    ):
        """Auto-grid area numbers restart within each compresa (per-compresa
        uniqueness, matching manual numbering) — not 1..N overall.  Points are
        interleaved across comprese to prove the per-compresa counter holds."""
        s = sample_setup
        p_a = s['area'].parcel                       # regions[0]
        p_b = Parcel.objects.create(                 # regions[1]
            name='9', region=regions[1], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        resp = self._post(writer_client, {
            FIELD_NAME: 'Multi-compresa grid', FIELD_R_M: 12,
            FIELD_POINTS: [
                {'compresa': p_a.region.name, 'particella': p_a.name,
                 FIELD_LAT: 38.5, FIELD_LON: 16.1},
                {'compresa': p_b.region.name, 'particella': p_b.name,
                 FIELD_LAT: 38.6, FIELD_LON: 16.2},
                {'compresa': p_a.region.name, 'particella': p_a.name,
                 FIELD_LAT: 38.7, FIELD_LON: 16.3},
            ],
        })
        assert resp.status_code == 200, resp.content
        grid = SampleGrid.objects.get(id=resp.json()[ROW_ID])
        nums_a = sorted(SampleArea.objects.filter(
            sample_grid=grid, parcel__region=regions[0],
        ).values_list('number', flat=True))
        nums_b = sorted(SampleArea.objects.filter(
            sample_grid=grid, parcel__region=regions[1],
        ).values_list('number', flat=True))
        assert nums_a == ['1', '2']      # two points in compresa A
        assert nums_b == ['1']           # restarts at 1 in compresa B

    def test_unknown_compresa_aborts(self, writer_client, sample_setup):
        n_before = SampleGrid.objects.count()
        resp = self._post(writer_client, {
            FIELD_NAME: 'Bad grid', FIELD_R_M: 12,
            FIELD_POINTS: [
                {'compresa': 'Nessuna', 'particella': '1',
                 FIELD_LAT: 38.5, FIELD_LON: 16.1},
            ],
        })
        assert resp.status_code == 400
        assert SampleGrid.objects.count() == n_before     # no partial commit

    def test_unknown_particella_aborts(self, writer_client, sample_setup):
        s = sample_setup
        n_before = SampleGrid.objects.count()
        resp = self._post(writer_client, {
            FIELD_NAME: 'Bad grid 2', FIELD_R_M: 12,
            FIELD_POINTS: [
                {'compresa': s['area'].parcel.region.name,
                 'particella': 'ZZZ',
                 FIELD_LAT: 38.5, FIELD_LON: 16.1},
            ],
        })
        assert resp.status_code == 400
        assert SampleGrid.objects.count() == n_before

    def test_duplicate_name_rejected(self, writer_client, sample_setup):
        SampleGrid.objects.create(name='Dup auto')
        resp = self._post(writer_client, {
            FIELD_NAME: 'Dup auto', FIELD_R_M: 12,
            FIELD_POINTS: [
                {'compresa': sample_setup['area'].parcel.region.name,
                 'particella': sample_setup['area'].parcel.name,
                 FIELD_LAT: 38.5, FIELD_LON: 16.1},
            ],
        })
        assert resp.status_code == 400

    def test_nonce_replay_does_not_duplicate_auto_grid(self, writer_client, sample_setup):
        s = sample_setup
        body = {
            FIELD_NAME: 'Auto grid nonce',
            FIELD_DESCRIPTION: '',
            FIELD_R_M: 12,
            FIELD_POINTS: [
                {'compresa': s['area'].parcel.region.name,
                 'particella': s['area'].parcel.name,
                 FIELD_LAT: 38.5, FIELD_LON: 16.1},
            ],
            FIELD_NONCE: 'auto-grid-nonce',
        }

        first = self._post(writer_client, body)
        second = self._post(writer_client, body)

        assert first.status_code == 200, first.content
        assert second.status_code == 200, second.content
        assert second.json() == first.json()
        assert SampleGrid.objects.filter(name='Auto grid nonce').count() == 1
        assert UsedNonce.objects.filter(nonce='auto-grid-nonce').exists()

    def test_empty_points_rejected(self, writer_client, db):
        resp = self._post(writer_client, {
            FIELD_NAME: 'Empty', FIELD_R_M: 12, FIELD_POINTS: [],
        })
        assert resp.status_code == 400

    def test_reader_forbidden(self, reader_client, sample_setup):
        resp = self._post(reader_client, {
            FIELD_NAME: 'X', FIELD_R_M: 12,
            FIELD_POINTS: [
                {'compresa': sample_setup['area'].parcel.region.name,
                 'particella': sample_setup['area'].parcel.name,
                 FIELD_LAT: 0.0, FIELD_LON: 0.0},
            ],
        })
        assert resp.status_code == 403


class TestDigestInvalidation:
    """Regression tests for digest staleness: every write that affects a
    digest's content must mark that digest stale.  Symptom of a missed
    invalidation is a stale on-disk file served as-is on the next read.
    """

    @staticmethod
    def _grids_n_rilev(client, grid_id, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = client.get('/api/campionamenti/grids/data/')
        d = _read_gzip_json(resp)
        row = next(r for r in d[ROWS]
                   if r[d[COLUMNS].index(ROW_ID)] == grid_id)
        return row[d[COLUMNS].index(S.COL_N_SURVEYS)]

    @staticmethod
    def _surveys_n_totali(client, survey_id, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = client.get('/api/campionamenti/surveys/data/')
        d = _read_gzip_json(resp)
        row = next(r for r in d[ROWS]
                   if r[d[COLUMNS].index(ROW_ID)] == survey_id)
        return row[d[COLUMNS].index(S.COL_N_AREAS_TOTAL)]

    def test_survey_create_invalidates_grids(self, writer_client, sample_setup,
                                             tmp_path, settings):
        """User-reported bug: after creating a survey on a grid, the grids
        digest must regenerate so `N. rilevamenti` reflects the new count.
        Without this, the next read returns a stale file even after page
        reload."""
        import json
        s = sample_setup
        # Baseline (fixture has 1 survey on this grid).
        n_before = self._grids_n_rilev(writer_client, s['grid'].id,
                                       tmp_path, settings)
        # Create a second survey.
        writer_client.post(
            '/api/campionamenti/survey/save/',
            data=json.dumps({
                FIELD_NAME: 'Survey two',
                FIELD_SAMPLE_GRID_ID: str(s['grid'].id),
            }),
            content_type='application/json',
        )
        n_after = self._grids_n_rilev(writer_client, s['grid'].id,
                                      tmp_path, settings)
        assert n_after == n_before + 1, (
            f'grids.N_rilevamenti should reflect new survey '
            f'(was {n_before}, now {n_after}); survey_save_view '
            f'is not marking grids stale.'
        )

    def test_survey_delete_invalidates_grids(self, writer_client, sample_setup,
                                             tmp_path, settings):
        """After deleting a survey, grids.N_rilevamenti must drop."""
        s = sample_setup
        # Add a second survey and immediately delete it.
        extra = Survey.objects.create(name='Extra', sample_grid=s['grid'])
        n_with_extra = self._grids_n_rilev(writer_client, s['grid'].id,
                                           tmp_path, settings)
        writer_client.post(
            f'/api/campionamenti/survey/delete/{extra.id}/',
            data=json.dumps({VERSION: extra.version}),
            content_type='application/json',
        )
        n_after = self._grids_n_rilev(writer_client, s['grid'].id,
                                      tmp_path, settings)
        assert n_after == n_with_extra - 1

    def test_area_create_invalidates_surveys(self, writer_client, sample_setup,
                                             regions, eclasses,
                                             tmp_path, settings):
        """Adding a SampleArea changes N. aree totali for every Survey on
        that grid → surveys digest must be invalidated."""
        import json
        s = sample_setup
        n_before = self._surveys_n_totali(writer_client, s['survey'].id,
                                          tmp_path, settings)
        writer_client.post(
            '/api/campionamenti/area/save/',
            data=json.dumps({
                FIELD_SAMPLE_GRID_ID: s['grid'].id,
                FIELD_PARCEL_ID: s['area'].parcel_id,
                FIELD_NUMBER: '777',
                FIELD_LAT: '0.0', FIELD_LON: '0.0', FIELD_R_M: '12',
            }),
            content_type='application/json',
        )
        n_after = self._surveys_n_totali(writer_client, s['survey'].id,
                                         tmp_path, settings)
        assert n_after == n_before + 1

    def test_area_delete_invalidates_surveys(self, writer_client, sample_setup,
                                             regions, eclasses,
                                             tmp_path, settings):
        s = sample_setup
        # Create an unused area, then delete it.
        unused_parcel = Parcel.objects.create(
            name='unused', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        unused = SampleArea.objects.create(
            sample_grid=s['grid'], parcel=unused_parcel, number='12',
            lat=0.0, lon=0.0, r_m=12,
        )
        n_before = self._surveys_n_totali(writer_client, s['survey'].id,
                                          tmp_path, settings)
        writer_client.post(
            f'/api/campionamenti/area/delete/{unused.id}/',
            data=json.dumps({VERSION: unused.version}),
            content_type='application/json',
        )
        n_after = self._surveys_n_totali(writer_client, s['survey'].id,
                                         tmp_path, settings)
        assert n_after == n_before - 1


class TestGridCsvImport:
    URL = '/api/campionamenti/grid/import-csv/'

    @staticmethod
    def _post(client, grid_id, csv_text):
        return client.post(
            TestGridCsvImport.URL,
            data=json.dumps({
                FIELD_SAMPLE_GRID_ID: str(grid_id),
                FIELD_FILE: _csv_b64(csv_text),
            }),
            content_type='application/json',
        )

    def test_happy_path_appends_to_existing(self, writer_client, sample_setup):
        """CSV import adds new areas to the chosen grid; no new grid created."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='Import target')
        n_grids_before = SampleGrid.objects.count()
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},10,16.1,38.5,500,12\n'
            f'{compresa},{particella},11,16.11,38.51,510,12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 200, resp.content
        data = resp.json()
        assert data['n_areas'] == 2
        assert data[ROW_ID] == grid.id
        assert SampleGrid.objects.count() == n_grids_before  # no new grid
        assert SampleArea.objects.filter(sample_grid=grid).count() == 2
        area_patches = [p for p in data[PATCHES] if p[DATA_ID] == 'sample_areas']
        assert len(area_patches) == 2

    def test_semicolon_comma_decimal_import(self, writer_client, sample_setup):
        """A ';'-delimited, comma-decimal file imports; lat/lon parsed."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='IT-format target')
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv_text = (
            'Compresa;Particella;Area saggio;Lon;Lat;Quota;Raggio\n'
            f'{compresa};{particella};10;16,1;38,5;500;12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 200, resp.content
        area = SampleArea.objects.get(sample_grid=grid, number='10')
        assert area.lat == 38.5 and area.lon == 16.1
        assert area.altitude_m == 500 and area.r_m == 12

    def test_unit_headers_import(self, writer_client, sample_setup):
        """Unit-bearing display headers are accepted as aliases."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='Unit headers')
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Alt. (m),Raggio (m)\n'
            f'{compresa},{particella},10,16.1,38.5,500,12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 200, resp.content
        area = SampleArea.objects.get(sample_grid=grid, number='10')
        assert area.altitude_m == 500 and area.r_m == 12

    def test_legacy_sample_grid_headers_import(self, writer_client, sample_setup):
        """Legacy-shaped setup CSV headers are accepted when Raggio is supplied."""
        s = sample_setup
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name

        grid = SampleGrid.objects.create(name='Legacy header import')
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},10,16.1,38.5,500,12\n'
            f'{compresa},{particella},11,16.2,38.6,501,12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 200, resp.content
        areas = list(SampleArea.objects.filter(sample_grid=grid).order_by('number'))
        assert len(areas) == 2
        assert {a.r_m for a in areas} == {12}
        assert {a.altitude_m for a in areas} == {500, 501}

    def test_unparseable_radius_flagged(self, writer_client, sample_setup):
        """A present-but-garbage Raggio is flagged, not silently defaulted."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='Bad radius')
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},10,16.1,38.5,500,abc\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 400
        assert SampleArea.objects.filter(sample_grid=grid).count() == 0

    def test_blank_radius_flagged(self, writer_client, sample_setup):
        """A blank Raggio cell is flagged (Raggio is required, no default)."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='Blank radius')
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},10,16.1,38.5,500,\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 400
        assert SampleArea.objects.filter(sample_grid=grid).count() == 0

    def test_second_import_appends_more(self, writer_client, sample_setup):
        """Two successive imports on the same grid should accumulate."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='Two-pass target')
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv1 = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},10,16.1,38.5,500,12\n'
        )
        csv2 = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},20,16.2,38.6,510,12\n'
        )
        assert self._post(writer_client, grid.id, csv1).status_code == 200
        assert self._post(writer_client, grid.id, csv2).status_code == 200
        assert SampleArea.objects.filter(sample_grid=grid).count() == 2

    def test_duplicate_area_rejected(self, writer_client, sample_setup):
        """Row with (parcel, number) matching an existing area in this grid
        is rejected with a per-row error; transaction does not commit any
        rows from the upload."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='Dup target')
        SampleArea.objects.create(
            sample_grid=grid, parcel=s['area'].parcel, number='10',
            lat=16.1, lon=38.5, r_m=12,
        )
        n_areas_before = SampleArea.objects.filter(sample_grid=grid).count()
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},10,16.1,38.5,500,12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 400
        body = resp.json()
        assert FIELD_ERRORS in body
        assert any('già presente' in e for e in body[FIELD_ERRORS])
        # Transaction rolled back — no rows added.
        assert SampleArea.objects.filter(sample_grid=grid).count() == n_areas_before

    def test_duplicate_within_csv_rejected(self, writer_client, sample_setup):
        """Two rows with the same (parcel, number) in one upload — second
        flagged as duplicate."""
        s = sample_setup
        grid = SampleGrid.objects.create(name='Intra-csv dup')
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{compresa},{particella},10,16.1,38.5,500,12\n'
            f'{compresa},{particella},10,16.2,38.6,520,12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 400
        assert SampleArea.objects.filter(sample_grid=grid).count() == 0

    def test_duplicate_number_other_parcel_same_region_rejected(
        self, writer_client, sample_setup, eclasses,
    ):
        """Per-region uniqueness: two parcels of the same region can't share a
        number, even though they're different parcels."""
        s = sample_setup
        region = s['area'].parcel.region
        other = Parcel.objects.create(
            name='1bis', region=region, eclass=eclasses[0],
            area_ha=Decimal('1.0'),
        )
        grid = SampleGrid.objects.create(name='Region dup')
        SampleArea.objects.create(
            sample_grid=grid, parcel=s['area'].parcel, number='10',
            lat=16.1, lon=38.5, r_m=12,
        )
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            f'{region.name},{other.name},10,16.2,38.6,500,12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 400
        assert SampleArea.objects.filter(sample_grid=grid, parcel=other).count() == 0

    def test_missing_grid_id(self, writer_client, db):
        resp = writer_client.post(
            self.URL,
            data=json.dumps({FIELD_FILE: _csv_b64(b'Compresa\n')}),
            content_type='application/json',
        )
        assert resp.status_code == 400

    def test_grid_not_found(self, writer_client, db):
        resp = writer_client.post(
            self.URL,
            data=json.dumps({
                FIELD_SAMPLE_GRID_ID: '999999',
                FIELD_FILE: _csv_b64(b'Compresa\n'),
            }),
            content_type='application/json',
        )
        assert resp.status_code == 404

    def test_missing_required_column(self, writer_client, sample_setup):
        grid = SampleGrid.objects.create(name='Bad cols')
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat\n'    # missing Quota
            'X,Y,1,16,38\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 400
        assert SampleArea.objects.filter(sample_grid=grid).count() == 0

    def test_unknown_parcel_reports_per_row_errors(self, writer_client, db):
        grid = SampleGrid.objects.create(name='Bad parcel')
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
            'Nessuna,1,1,16.0,38.0,500,12\n'
        )
        resp = self._post(writer_client, grid.id, csv_text)
        assert resp.status_code == 400
        body = resp.json()
        assert FIELD_ERRORS in body
        assert any('Nessuna' in e for e in body[FIELD_ERRORS])

    def test_reader_forbidden(self, reader_client, sample_setup):
        grid = sample_setup['grid']
        resp = reader_client.post(
            self.URL,
            data=json.dumps({
                FIELD_SAMPLE_GRID_ID: str(grid.id),
                FIELD_FILE: _csv_b64(b'a,b\n1,2'),
            }),
            content_type='application/json',
        )
        assert resp.status_code == 403


class TestTreeCsvImport:
    URL = '/api/campionamenti/survey/import-csv/'

    @staticmethod
    def _post(client, survey_id, csv_text, default_date=''):
        return client.post(
            TestTreeCsvImport.URL,
            data=json.dumps({
                FIELD_SURVEY_ID: str(survey_id),
                FIELD_DEFAULT_DATE: default_date,
                FIELD_FILE: _csv_b64(csv_text),
            }),
            content_type='application/json',
        )

    def test_unstructured_survey_import_creates_free_sample(
            self, writer_client, parcels, species, db,
    ):
        from apps.base.models import Sample, Survey, Tree, TreeSample

        survey = Survey.objects.create(name='Unstructured CSV target')
        parcel = parcels[0]
        csv_text = (
            'Compresa,Particella,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,Data,H_measured,Lat,Lon\n'
            f'{parcel.region.name},{parcel.name},1,0,false,30,20.5,10,2,'
            f'{species[0].common_name},true,2024-09-15,true,38.5,16.1\n'
        )

        resp = self._post(writer_client, survey.id, csv_text)

        assert resp.status_code == 200, resp.content
        assert resp.json()['n_samples'] == 1
        assert resp.json()['n_trees'] == 1
        sample = Sample.objects.get(survey=survey, sample_area__isnull=True)
        row = TreeSample.objects.select_related('tree').get(sample=sample)
        assert row.parcel == parcel
        assert row.tree.parcel == parcel
        assert row.tree.species == species[0]
        assert row.h_measured is True
        assert row.lat == pytest.approx(38.5)
        assert row.lon == pytest.approx(16.1)
        assert Tree.objects.filter(id=row.tree_id).count() == 1

    def test_happy_path(
        self, writer_client, sample_setup, tmp_path, settings,
    ):
        """Importing rows updates every affected public tree digest."""
        from apps.base.models import Survey, Tree, TreeSample
        s = sample_setup
        settings.DIGEST_DIR = tmp_path
        # Make the empty target the dendrometry source so both Bosco
        # tree-derived digests have an observable before/after transition.
        empty_survey = Survey.objects.create(
            name='CSV import target', sample_grid=s['grid'], active=True,
        )
        urls = {
            'trees': f'/api/campionamenti/trees/{empty_survey.id}/',
            'samples': '/api/campionamenti/samples/data/',
            'surveys': '/api/campionamenti/surveys/data/',
            'dendrometry': '/api/bosco/parcel-dendrometry/data/',
            'points': '/api/bosco/parcel-dendrometry-points/data/',
        }

        def read_digest(name):
            response = writer_client.get(urls[name])
            assert response.status_code == 200
            return _read_gzip_json(response)

        before = {name: read_digest(name) for name in urls}
        assert before['trees'][ROWS] == []
        assert before['dendrometry'][ROWS] == []
        assert before['points'][ROWS] == []
        assert not any(
            row[before['samples'][COLUMNS].index(S.COL_SURVEY)] == empty_survey.id
            for row in before['samples'][ROWS]
        )
        survey_before = next(
            row for row in before['surveys'][ROWS]
            if row[before['surveys'][COLUMNS].index(ROW_ID)] == empty_survey.id
        )
        assert survey_before[
            before['surveys'][COLUMNS].index(S.COL_N_AREAS_VISITED)
        ] == 0

        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        adc = s['area'].number
        n_trees_before = Tree.objects.count()
        n_preserved_before = Tree.objects.filter(preserved=True).count()
        n_ts_before = TreeSample.objects.count()
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,Data,PAI\n'
            f'{compresa},{particella},{adc},10,0,false,30,20.5,10,2,Abete,true,'
            '2024-09-15,true\n'
            f'{compresa},{particella},{adc},11,0,false,32,22.5,11,2,Abete,true,'
            '2024-09-15,false\n'
        )
        resp = self._post(writer_client, empty_survey.id, csv_text)
        assert resp.status_code == 200, resp.content
        data = resp.json()
        assert data['n_samples'] == 1
        assert data['n_trees'] == 2
        assert Tree.objects.count() == n_trees_before + 2
        assert TreeSample.objects.count() == n_ts_before + 2
        assert Tree.objects.filter(preserved=True).count() == n_preserved_before + 1
        assert list(
            TreeSample.objects
            .filter(sample__survey=empty_survey)
            .order_by(FIELD_NUMBER)
            .values_list(FIELD_H_MEASURED, flat=True)
        ) == [False, False]
        _assert_stale(
            DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS,
            DIGEST_PRESERVED_TREES,
        )

        trees = read_digest('trees')
        assert [
            row[trees[COLUMNS].index(S.COL_TREE_NUM)] for row in trees[ROWS]
        ] == [10, 11]

        samples = read_digest('samples')
        sample_row = next(
            row for row in samples[ROWS]
            if row[samples[COLUMNS].index(S.COL_SURVEY)] == empty_survey.id
        )
        assert sample_row[samples[COLUMNS].index(S.COL_N_TREES)] == 2

        surveys = read_digest('surveys')
        survey_row = next(
            row for row in surveys[ROWS]
            if row[surveys[COLUMNS].index(ROW_ID)] == empty_survey.id
        )
        assert survey_row[
            surveys[COLUMNS].index(S.COL_N_AREAS_VISITED)
        ] == 1
        assert survey_row[surveys[COLUMNS].index(S.COL_DATE_FIRST)] == '2024-09-15'
        assert survey_row[surveys[COLUMNS].index(S.COL_DATE_LAST)] == '2024-09-15'

        dendrometry = read_digest('dendrometry')
        dendrometry_row = next(
            row for row in dendrometry[ROWS]
            if row[dendrometry[COLUMNS].index(COL_SURVEY_ID)] == empty_survey.id
        )
        assert dendrometry_row[dendrometry[COLUMNS].index(S.COL_N_TREES)] == 2

        points = read_digest('points')
        assert len([
            row for row in points[ROWS]
            if row[points[COLUMNS].index(COL_SURVEY_ID)] == empty_survey.id
        ]) == 2

    def test_h_measured_column_persists_when_present(
            self, writer_client, sample_setup,
    ):
        s = sample_setup
        target = Survey.objects.create(
            name='CSV h measured target', sample_grid=s['grid'],
        )
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        adc = s['area'].number
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,H_measured\n'
            f'{compresa},{particella},{adc},20,0,false,30,20.5,10,2,Abete,true,true\n'
            f'{compresa},{particella},{adc},21,0,false,32,22.5,11,2,Abete,true,false\n'
        )

        resp = self._post(writer_client, target.id, csv_text, default_date='2024-09-15')

        assert resp.status_code == 200, resp.content
        assert list(
            TreeSample.objects
            .filter(sample__survey=target)
            .order_by(FIELD_NUMBER)
            .values_list(FIELD_H_MEASURED, flat=True)
        ) == [True, False]

    def test_duplicate_number_shoot_within_csv_rejected(
        self, writer_client, sample_setup,
    ):
        from apps.base.models import Survey, TreeSample
        s = sample_setup
        empty_survey = Survey.objects.create(
            name='CSV duplicate tree target', sample_grid=s['grid'],
        )
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        adc = s['area'].number
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,Data\n'
            f'{compresa},{particella},{adc},10,0,false,30,20.5,10,2,Abete,true,'
            '2024-09-15\n'
            f'{compresa},{particella},{adc},10,0,false,32,22.5,11,2,Abete,true,'
            '2024-09-15\n'
        )

        resp = self._post(writer_client, empty_survey.id, csv_text)

        assert resp.status_code == 400
        assert (S.ERR_CSV_ROW_TREE_NUMBER_DUPLICATE.format(3, 10, 0)
                in resp.json()[FIELD_ERRORS])
        assert TreeSample.objects.filter(sample__survey=empty_survey).count() == 0

    def test_conflicting_dates_for_same_area_rejected(
        self, writer_client, sample_setup,
    ):
        from apps.base.models import Survey, TreeSample
        s = sample_setup
        empty_survey = Survey.objects.create(
            name='CSV date conflict target', sample_grid=s['grid'],
        )
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        adc = s['area'].number
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,Data\n'
            f'{compresa},{particella},{adc},10,0,false,30,20.5,10,2,Abete,true,'
            '2024-09-15\n'
            f'{compresa},{particella},{adc},11,0,false,32,22.5,11,2,Abete,true,'
            '2024-09-16\n'
        )
        resp = self._post(writer_client, empty_survey.id, csv_text)
        assert resp.status_code == 400
        body = resp.json()
        assert S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
            3, compresa, particella, adc, '2024-09-15',
        ) in body[FIELD_ERRORS]
        assert Sample.objects.filter(survey=empty_survey).count() == 0
        assert TreeSample.objects.filter(sample__survey=empty_survey).count() == 0

    def test_existing_sample_date_conflict_rejected(
        self, writer_client, sample_setup,
    ):
        s = sample_setup
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        adc = s['area'].number
        n_trees_before = TreeSample.objects.filter(sample=s['sample']).count()
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,Data\n'
            f'{compresa},{particella},{adc},99,0,false,30,20.5,10,2,Abete,true,'
            '2025-01-01\n'
        )
        resp = self._post(writer_client, s['survey'].id, csv_text)
        assert resp.status_code == 400
        body = resp.json()
        assert S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
            2, compresa, particella, adc, '2024-09-15',
        ) in body[FIELD_ERRORS]
        assert TreeSample.objects.filter(sample=s['sample']).count() == n_trees_before

    def test_nonce_replay_does_not_duplicate_rows(self, writer_client, sample_setup):
        from apps.base.models import Survey, TreeSample
        s = sample_setup
        empty_survey = Survey.objects.create(
            name='CSV nonce target', sample_grid=s['grid'],
        )
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,Data\n'
            f'{s["area"].parcel.region.name},{s["area"].parcel.name},'
            f'{s["area"].number},10,0,false,30,20.5,10,2,Abete,true,'
            '2024-09-15\n'
        )
        body = {
            FIELD_SURVEY_ID: str(empty_survey.id),
            FIELD_DEFAULT_DATE: '',
            FIELD_FILE: _csv_b64(csv_text),
            FIELD_NONCE: 'tree-import-nonce',
        }
        resp1 = writer_client.post(
            self.URL, data=json.dumps(body), content_type='application/json',
        )
        resp2 = writer_client.post(
            self.URL, data=json.dumps(body), content_type='application/json',
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert TreeSample.objects.filter(sample__survey=empty_survey).count() == 1

    def test_missing_required_column(self, writer_client, sample_setup):
        csv_text = (
            'Compresa,Particella,Area saggio,Albero\n'  # missing many
            'X,Y,1,1\n'
        )
        resp = self._post(writer_client, sample_setup['survey'].id, csv_text)
        assert resp.status_code == 400

    def test_missing_data_column_without_default_date(self, writer_client,
                                                     sample_setup):
        s = sample_setup
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia\n'
            f'{s["area"].parcel.region.name},{s["area"].parcel.name},'
            f'{s["area"].number},10,0,false,30,20.5,10,2,Abete,true\n'
        )
        resp = self._post(writer_client, s['survey'].id, csv_text)
        assert resp.status_code == 400

    def test_default_date_used_when_no_data_column(self, writer_client,
                                                  sample_setup):
        from apps.base.models import Sample, Survey
        s = sample_setup
        empty_survey = Survey.objects.create(
            name='CSV default date target', sample_grid=s['grid'],
        )
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia\n'
            f'{s["area"].parcel.region.name},{s["area"].parcel.name},'
            f'{s["area"].number},99,0,false,30,20.5,10,2,Abete,true\n'
        )
        resp = self._post(writer_client, empty_survey.id, csv_text,
                          default_date='2025-06-01')
        assert resp.status_code == 200, resp.content
        sample = Sample.objects.get(survey=empty_survey, sample_area=s['area'])
        assert sample.date.isoformat() == '2025-06-01'

    def test_empty_survey_id_returns_clean_400(self, writer_client, sample_setup):
        """User-reported bug: leaving the target-survey pulldown on
        '— Seleziona —' used to silently fail (HTML5 `required` blocked the
        submit and our JS handler never ran).  The form is now `novalidate`
        and we rely on the server to return a friendly 400 — make sure
        that path renders the error message clients can show."""
        resp = writer_client.post(
            self.URL,
            data=json.dumps({
                FIELD_SURVEY_ID: '',      # empty: user didn't pick a survey
                FIELD_DEFAULT_DATE: '',
                FIELD_FILE: _csv_b64(b'a,b\n1,2'),
            }),
            content_type='application/json',
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body[MESSAGE]    # non-empty user-facing message

    def test_unknown_area_reports_error(self, writer_client, sample_setup):
        s = sample_setup
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia,Data\n'
            f'{s["area"].parcel.region.name},{s["area"].parcel.name},'
            'ZZZ,1,0,false,30,20,0,2,Abete,true,2024-09-15\n'
        )
        resp = self._post(writer_client, s['survey'].id, csv_text)
        assert resp.status_code == 400
        body = resp.json()
        assert any('ZZZ' in e for e in body[FIELD_ERRORS])

    def test_reader_forbidden(self, reader_client, sample_setup):
        resp = reader_client.post(
            self.URL,
            data=json.dumps({
                FIELD_SURVEY_ID: str(sample_setup['survey'].id),
                FIELD_FILE: _csv_b64(b'a,b\n1,2'),
            }),
            content_type='application/json',
        )
        assert resp.status_code == 403


class TestGridEditDelete:
    """Pencil/garbage on Section 1 pulldown (Bucket 2)."""

    @staticmethod
    def _post(client, url, body=None):
        import json
        return client.post(
            url, data=json.dumps(body or {}),
            content_type='application/json',
        )

    def test_edit_grid_renames(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/edit/{s["grid"].id}/',
                          {
                              FIELD_NAME: 'Rinominata', FIELD_DESCRIPTION: 'desc',
                              VERSION: str(s['grid'].version),
                          })
        assert resp.status_code == 200
        s['grid'].refresh_from_db()
        assert s['grid'].name == 'Rinominata'

    def test_edit_grid_name_required(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/edit/{s["grid"].id}/',
                          {FIELD_NAME: '', FIELD_DESCRIPTION: ''})
        assert resp.status_code == 400

    def test_edit_grid_duplicate_name(self, writer_client, sample_setup):
        s = sample_setup
        SampleGrid.objects.create(name='Other grid X')
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/edit/{s["grid"].id}/',
                          {
                              FIELD_NAME: 'Other grid X',
                              VERSION: str(s['grid'].version),
                          })
        assert resp.status_code == 400

    def test_delete_grid_in_use_refused(self, writer_client, sample_setup):
        """Grid with surveys is protected (Survey.sample_grid on_delete=PROTECT
        plus the explicit pre-check)."""
        s = sample_setup
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/delete/{s["grid"].id}/',
                          {VERSION: s['grid'].version})
        assert resp.status_code == 400
        assert SampleGrid.objects.filter(id=s['grid'].id).exists()

    def test_delete_unused_grid(self, writer_client, db):
        """An empty grid (no surveys, no areas) deletes cleanly."""
        g = SampleGrid.objects.create(name='Empty grid')
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/delete/{g.id}/',
                          {VERSION: g.version})
        assert resp.status_code == 200
        assert not SampleGrid.objects.filter(id=g.id).exists()

    def test_delete_saves_nonce(self, writer_client, db):
        g = SampleGrid.objects.create(name='Empty grid')
        resp = self._post(
            writer_client, f'/api/campionamenti/grid/delete/{g.id}/',
            {VERSION: g.version, FIELD_NONCE: 'grid-delete-nonce'},
        )
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='grid-delete-nonce').exists()

    def test_delete_grid_cascades_to_areas(self, writer_client, db,
                                           regions, eclasses):
        """A grid with areas but NO surveys cascades to its areas."""
        from apps.base.models import Parcel
        g = SampleGrid.objects.create(name='Just areas')
        p = Parcel.objects.create(name='5', region=regions[0],
                                  eclass=eclasses[0], area_ha=Decimal('1.0'))
        SampleArea.objects.create(sample_grid=g, parcel=p, number='1',
                                  lat=0, lon=0, r_m=12)
        SampleArea.objects.create(sample_grid=g, parcel=p, number='2',
                                  lat=0, lon=0, r_m=12)
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/delete/{g.id}/',
                          {VERSION: g.version})
        assert resp.status_code == 200
        assert not SampleArea.objects.filter(sample_grid=g).exists()

    def test_delete_stale_version_conflicts(self, writer_client, db):
        g = SampleGrid.objects.create(name='Empty grid')
        resp = self._post(
            writer_client, f'/api/campionamenti/grid/delete/{g.id}/',
            {VERSION: g.version + 1},
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert SampleGrid.objects.filter(id=g.id).exists()

    def test_reader_forbidden(self, reader_client, sample_setup):
        s = sample_setup
        resp = self._post(reader_client,
                          f'/api/campionamenti/grid/edit/{s["grid"].id}/',
                          {FIELD_NAME: 'X'})
        assert resp.status_code == 403


class TestSurveyEditDelete:
    """Pencil/garbage on Section 2 pulldown (Bucket 2)."""

    @staticmethod
    def _post(client, url, body=None):
        import json
        return client.post(
            url, data=json.dumps(body or {}),
            content_type='application/json',
        )

    def test_edit_survey_renames(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client,
                          f'/api/campionamenti/survey/edit/{s["survey"].id}/',
                          {
                              FIELD_NAME: 'Sv rinominato',
                              FIELD_DESCRIPTION: 'desc',
                              VERSION: str(s['survey'].version),
                          })
        assert resp.status_code == 200
        s['survey'].refresh_from_db()
        assert s['survey'].name == 'Sv rinominato'

    def test_edit_survey_duplicate_name(self, writer_client, sample_setup):
        s = sample_setup
        Survey.objects.create(name='Other survey X', sample_grid=s['grid'])
        resp = self._post(writer_client,
                          f'/api/campionamenti/survey/edit/{s["survey"].id}/',
                          {
                              FIELD_NAME: 'Other survey X',
                              VERSION: str(s['survey'].version),
                          })
        assert resp.status_code == 400

    def test_delete_survey_cascades_to_samples_and_tree_samples(
        self, writer_client, sample_setup,
    ):
        """Deleting a survey cascades to its samples and their tree_samples;
        the Tree row itself remains (TreeSample.tree=PROTECT, but the
        cascade goes via Sample, not Tree)."""
        from apps.base.models import Tree, TreeSample
        s = sample_setup
        survey_id = s['survey'].id
        tree_id = s['tree'].id
        n_trees_before = Tree.objects.count()

        resp = self._post(writer_client,
                          f'/api/campionamenti/survey/delete/{survey_id}/',
                          {VERSION: s['survey'].version})
        assert resp.status_code == 200
        assert not Survey.objects.filter(id=survey_id).exists()
        assert not Sample.objects.filter(survey_id=survey_id).exists()
        assert not TreeSample.objects.filter(
            sample__survey_id=survey_id,
        ).exists()
        # Tree row survives (it's the cross-sample identity carrier).
        assert Tree.objects.filter(id=tree_id).exists()
        assert Tree.objects.count() == n_trees_before

    def test_delete_empty_survey(self, writer_client, sample_setup):
        empty = Survey.objects.create(
            name='Empty survey 1', sample_grid=sample_setup['grid'],
        )
        resp = self._post(writer_client,
                          f'/api/campionamenti/survey/delete/{empty.id}/',
                          {VERSION: empty.version})
        assert resp.status_code == 200
        assert not Survey.objects.filter(id=empty.id).exists()

    def test_delete_saves_nonce(self, writer_client, sample_setup):
        empty = Survey.objects.create(
            name='Empty survey 1', sample_grid=sample_setup['grid'],
        )
        resp = self._post(
            writer_client, f'/api/campionamenti/survey/delete/{empty.id}/',
            {VERSION: empty.version, FIELD_NONCE: 'survey-delete-nonce'},
        )
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='survey-delete-nonce').exists()

    def test_delete_stale_version_conflicts(self, writer_client, sample_setup):
        empty = Survey.objects.create(
            name='Empty survey 1', sample_grid=sample_setup['grid'],
        )
        resp = self._post(
            writer_client, f'/api/campionamenti/survey/delete/{empty.id}/',
            {VERSION: empty.version + 1},
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert Survey.objects.filter(id=empty.id).exists()

    def test_reader_forbidden(self, reader_client, sample_setup):
        s = sample_setup
        resp = self._post(reader_client,
                          f'/api/campionamenti/survey/edit/{s["survey"].id}/',
                          {FIELD_NAME: 'X'})
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
        html = resp.json()[HTML]
        # Single creation path (CSV import moved to pencil modal).
        assert 'campionamenti-survey-form-empty' in html
        # Grid pulldown contains the fixture's grid and the unstructured option.
        assert sample_setup['grid'].name in html
        assert S.NO_SAMPLE_GRID in html
        assert SAMPLE_GRID_UNSTRUCTURED in html

    def test_reader_form_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/campionamenti/survey/form/')
        assert resp.status_code == 403

    def test_create_empty_survey(self, writer_client, sample_setup):
        from apps.base.models import Survey
        resp = self._post(writer_client, {
            FIELD_NAME: 'Rilevamento di prova',
            FIELD_SAMPLE_GRID_ID: str(sample_setup['grid'].id),
            FIELD_DESCRIPTION: 'desc',
        })
        assert resp.status_code == 200
        data = resp.json()
        sv = Survey.objects.get(id=data[ROW_ID])
        assert sv.name == 'Rilevamento di prova'
        assert sv.sample_grid_id == sample_setup['grid'].id

    def test_create_unstructured_survey(self, writer_client, db):
        from apps.base.models import Survey
        resp = self._post(writer_client, {
            FIELD_NAME: 'Rilevamento non strutturato',
            FIELD_SAMPLE_GRID_ID: SAMPLE_GRID_UNSTRUCTURED,
            FIELD_DESCRIPTION: 'desc',
        })
        assert resp.status_code == 200
        data = resp.json()
        sv = Survey.objects.get(id=data[ROW_ID])
        assert sv.name == 'Rilevamento non strutturato'
        assert sv.sample_grid_id is None
        survey_patch = next(
            p for p in data[PATCHES]
            if p[DATA_ID] == 'surveys' and p[ROW_ID] == sv.id
        )
        assert survey_patch[RECORD][4] is None
        assert not any(p[DATA_ID] == 'grids' for p in data[PATCHES])

    def test_name_required(self, writer_client, sample_setup):
        resp = self._post(writer_client, {
            FIELD_NAME: '', FIELD_SAMPLE_GRID_ID: str(sample_setup['grid'].id),
        })
        assert resp.status_code == 400

    def test_grid_required(self, writer_client, db):
        resp = self._post(writer_client, {FIELD_NAME: 'X', FIELD_SAMPLE_GRID_ID: ''})
        assert resp.status_code == 400

    def test_grid_must_exist(self, writer_client, db):
        resp = self._post(writer_client, {FIELD_NAME: 'X', FIELD_SAMPLE_GRID_ID: '9999'})
        assert resp.status_code == 400

    def test_name_duplicate_rejected(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, {
            FIELD_NAME: s['survey'].name,
            FIELD_SAMPLE_GRID_ID: str(s['grid'].id),
        })
        assert resp.status_code == 400

    def test_reader_forbidden(self, reader_client, sample_setup):
        s = sample_setup
        resp = self._post(reader_client, {
            FIELD_NAME: 'X', FIELD_SAMPLE_GRID_ID: str(s['grid'].id),
        })
        assert resp.status_code == 403
