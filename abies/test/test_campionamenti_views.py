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
from config import strings as S
from config.constants import (
    AREA_RECORDS, COLUMNS, FIELD_DATE, FIELD_DEFAULT_DATE, FIELD_DESCRIPTION,
    FIELD_D_CM, FIELD_ERRORS, FIELD_FUSTAIA, FIELD_H_M, FIELD_LAT, FIELD_LON,
    FIELD_MASS_Q, FIELD_NAME, FIELD_NOTE, FIELD_NUMBER, FIELD_PARCEL_ID,
    FIELD_PRESERVED, FIELD_R_M, FIELD_SAMPLE_AREA_ID, FIELD_SAMPLE_GRID_ID,
    FIELD_SHOOT, FIELD_SPECIES_ID, FIELD_STANDARD, FIELD_SURVEY_ID,
    GRID_RECORD, HTML, MESSAGE, RECORD, RECORDS, ROWS, ROW_ID, SAMPLE_RECORD,
    SURVEY_RECORD, SURVEY_RECORDS,
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
        assert 'Nome' in d[COLUMNS]
        assert any(r[d[COLUMNS].index(S.COL_NAME)] == 'Test grid' for r in d[ROWS])

    def test_surveys_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/surveys/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert d[ROWS][0][d[COLUMNS].index(S.COL_N_AREAS_VISITED)] == 1
        assert d[ROWS][0][d[COLUMNS].index(S.COL_N_AREAS_TOTAL)] == 1

    def test_sample_areas_data(self, writer_client, sample_setup, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/sample-areas/data/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert len(d[ROWS]) == 1
        assert d[ROWS][0][d[COLUMNS].index(S.COL_RAGGIO)] == 12

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
        assert row[d[COLUMNS].index(S.COL_PRODUCT)] == 'fustaia'
        assert row[d[COLUMNS].index(S.COL_D_CM)] == 30

    def test_trees_data_unknown_survey(self, writer_client, sample_setup,
                                       tmp_path, settings):
        """Requesting a non-existent survey id returns an empty digest."""
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/campionamenti/trees/9999/')
        assert resp.status_code == 200
        d = _read_gzip_json(resp)
        assert d[ROWS] == []

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
        html = resp.json()[HTML]
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
            r'<input[^>]*id="id_species"[^>]*>', html,
        )
        assert match is not None
        tag = match.group(0)
        assert 'type="hidden"' in tag
        assert 'data-density="' in tag
        assert 'data-name="' in tag


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
            FIELD_D_CM: '30', FIELD_H_M: '20.5', 'l10_mm': '12',
            'volume_m3': '0.7022', FIELD_MASS_Q: '6.32',
            FIELD_FUSTAIA: 'true',
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
            lat=0.0, lon=0.0, r_m=12,
        )
        s = sample_setup
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(other_area.id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '1', FIELD_D_CM: '30', FIELD_H_M: '20', 'l10_mm': '0',
            'volume_m3': '0.5', FIELD_MASS_Q: '4.7',
            FIELD_FUSTAIA: 'true',
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
            'volume_m3': '0', FIELD_MASS_Q: '0', FIELD_FUSTAIA: 'true',
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
            FIELD_FUSTAIA: 'true',
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
            FIELD_FUSTAIA: 'true',
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
            'tree_pick': str(s['tree'].id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '1',         # propagated from the existing tree
            FIELD_D_CM: '35', FIELD_H_M: '21', 'l10_mm': '0',
            'volume_m3': '0.9', FIELD_MASS_Q: '7.1',
            FIELD_FUSTAIA: 'true',
            FIELD_LAT: str(s['tree'].lat or 0.0),
            FIELD_LON: str(s['tree'].lon or 0.0),
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
            sample=other_sample, tree=other_tree, shoot=0, standard=False,
            number=7, d_cm=50, h_m=Decimal('30.00'), l10_mm=0,
            volume_m3=Decimal('1.0'), mass_q=Decimal('9.0'),
        )

        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),    # ours, not other_area
            'tree_pick': str(other_tree.id),
            FIELD_SPECIES_ID: str(s['tree'].species_id),
            FIELD_NUMBER: '7', FIELD_D_CM: '40', FIELD_H_M: '20', 'l10_mm': '0',
            'volume_m3': '0.5', FIELD_MASS_Q: '4.0',
            FIELD_FUSTAIA: 'true',
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
            FIELD_FUSTAIA: 'true',
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
        payload['sample_area_id'] = str(new_area.id)
        resp = self._post(writer_client, payload)
        assert resp.status_code == 200, resp.content
        new_sample = Sample.objects.get(
            survey=s['survey'], sample_area=new_area,
        )
        assert new_sample.date.isoformat() == '2025-03-10'

    def test_create_updates_sample_date_when_different(
        self, writer_client, sample_setup,
    ):
        """Adding a tree to an existing sample with a new date bumps the
        sample's date (same semantics as the legacy inline selector)."""
        s = sample_setup
        assert s['sample'].date.isoformat() == '2024-09-15'
        resp = self._post(writer_client, self._save_payload(s, 99, '2025-04-01'))
        assert resp.status_code == 200, resp.content
        s['sample'].refresh_from_db()
        assert s['sample'].date.isoformat() == '2025-04-01'

    def test_edit_updates_sample_date(self, writer_client, sample_setup):
        from apps.base.models import TreeSample
        s = sample_setup
        ts = TreeSample.objects.get(sample=s['sample'], number=1)
        payload = self._save_payload(s, 1, '2025-05-20')
        payload[ROW_ID] = str(ts.id)
        resp = self._post(writer_client, payload)
        assert resp.status_code == 200, resp.content
        s['sample'].refresh_from_db()
        assert s['sample'].date.isoformat() == '2025-05-20'

    def test_rejects_invalid_date(self, writer_client, sample_setup):
        payload = self._save_payload(sample_setup, 1, 'not-a-date')
        resp = self._post(writer_client, payload)
        assert resp.status_code == 400
        assert 'Data' in resp.json()[MESSAGE]

    def test_response_sample_record_reflects_new_date(
        self, writer_client, sample_setup,
    ):
        """The optimistic-update contract: when the write changes
        sample.date, the response's sample_record must carry it so the
        client cache patches without another fetch."""
        from apps.base.digests import build_sample_record
        s = sample_setup
        resp = self._post(writer_client, self._save_payload(s, 50, '2025-06-15'))
        assert resp.status_code == 200, resp.content
        s['sample'].refresh_from_db()
        assert resp.json()[SAMPLE_RECORD] == build_sample_record(s['sample'])


class TestTreeFormPriorTrees:
    """Form GET reflects the prior-trees pulldown contents."""

    def test_lists_prior_trees(self, writer_client, sample_setup):
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}'
            f'&area={s["area"].id}'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        # The existing fixture has tree number=1 in this area.
        assert 'id="id_tree_pick"' in html
        assert '+ nuovo albero' in html
        assert 'n.1' in html
        # next_number = max(existing)+1 = 2
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
            area_ha=Decimal('1.0'),
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
        idx = html.find('id="id_ceduo"')
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
        idx = html.find('id="id_ceduo"')
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
                sample=s['sample'], tree=coppice_tree, shoot=sh,
                standard=(sh == 2), number=7,
                d_cm=5 + sh, h_m=Decimal('8.00'), l10_mm=0,
            )
        resp = writer_client.get(
            f'/api/campionamenti/tree/form/?survey={s["survey"].id}'
            f'&area={s["area"].id}'
        )
        html = resp.json()[HTML]
        assert 'data-next-shoot="3"' in html
        assert 'ceduo' in html
        assert 'n.7' in html


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
            area_ha=Decimal('1.0'),
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
            FIELD_NUMBER: '1', FIELD_FUSTAIA: 'false',
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

    def test_coppice_requires_at_least_one_shoot(
        self, writer_client, sample_setup, regions, eclasses,
    ):
        coppice_eclass = next(e for e in eclasses if e.coppice)
        parcel = Parcel.objects.create(
            name='c2', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'),
        )
        area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=parcel, number='1',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(sample_setup['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(area.id),
            FIELD_SPECIES_ID: str(sample_setup['tree'].species_id),
            FIELD_NUMBER: '1', FIELD_FUSTAIA: 'false',
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
            sample=s['sample'], tree=existing, shoot=1,
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
            'tree_pick': str(existing.id),
            FIELD_SPECIES_ID: str(existing.species_id),
            FIELD_NUMBER: '42', FIELD_FUSTAIA: 'false',
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
            sample=s['sample'], tree=existing, shoot=1, standard=False,
            number=99, d_cm=5, h_m=Decimal('8.00'), l10_mm=0,
        )
        resp = self._post(writer_client, {
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            'tree_pick': str(existing.id),
            FIELD_SPECIES_ID: str(existing.species_id),
            FIELD_NUMBER: '99', FIELD_FUSTAIA: 'false',
            'shoots': json.dumps([
                {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 6,
                 FIELD_H_M: '8.5', 'l10_mm': 0},     # collides
            ]),
        })
        assert resp.status_code == 400
        msg = resp.json()[MESSAGE].lower()
        assert 'pollone' in msg

    def test_edit_coppice_single_shoot(self, writer_client, sample_setup,
                                       species):
        """Editing a single coppice TreeSample updates only that row."""
        s = sample_setup
        tree = Tree.objects.create(
            species=species[1], parcel=s['area'].parcel,
            preserved=False, coppice=True,
        )
        ts1 = TreeSample.objects.create(
            sample=s['sample'], tree=tree, shoot=1, standard=False,
            number=15, d_cm=5, h_m=Decimal('8.00'), l10_mm=0,
        )
        ts2 = TreeSample.objects.create(
            sample=s['sample'], tree=tree, shoot=2, standard=True,
            number=15, d_cm=6, h_m=Decimal('8.50'), l10_mm=12,
        )
        n_ts_before = TreeSample.objects.count()
        resp = self._post(writer_client, {
            ROW_ID: str(ts1.id),
            FIELD_SURVEY_ID: str(s['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(s['area'].id),
            'tree_pick': str(tree.id),
            FIELD_SPECIES_ID: str(species[0].id),         # changed
            FIELD_NUMBER: '15', FIELD_FUSTAIA: 'false',
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
        # Modal carries all three creation paths per campionamenti.md §1.
        assert 'data-path="empty"' in html
        assert 'data-path="auto"' in html
        assert 'data-path="csv"' in html
        # Default-active body is the empty-grid create form.
        assert 'campionamenti-grid-form-empty' in html

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

    def test_form_add_renders(self, writer_client, sample_setup):
        s = sample_setup
        resp = writer_client.get(
            f'/api/campionamenti/area/form/?grid={s["grid"].id}'
            f'&lat=38.5&lon=16.1'
        )
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert '<form' in html
        assert 'value="38.5"' in html or 'value="38.50' in html

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
            FIELD_LAT: '38.6', FIELD_LON: '16.2',
            'altitude_m': '500',
            FIELD_R_M: '15', FIELD_NOTE: 'test',
        })
        assert resp.status_code == 200, resp.content
        assert SampleArea.objects.filter(sample_grid=s['grid']).count() == n_before + 1

    def test_update_area(self, writer_client, sample_setup):
        s = sample_setup
        resp = self._post(writer_client, '/api/campionamenti/area/save/', {
            ROW_ID: s['area'].id,
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
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert not SampleArea.objects.filter(id=unused.id).exists()

    def test_delete_in_use_refused(self, writer_client, sample_setup):
        """An area referenced by any Sample is protected from delete."""
        s = sample_setup
        resp = writer_client.post(
            f'/api/campionamenti/area/delete/{s["area"].id}/',
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
            FIELD_FUSTAIA: 'true',
        })
        assert resp.status_code == 200, resp.content
        payload = resp.json()
        # Single-shoot fustaia create → records=[<one row>], no `record`.
        assert payload.get(RECORDS), payload
        assert len(payload[RECORDS]) == 1
        record = payload[RECORDS][0]
        # Match against the canonical row built from the freshly-saved ts.
        ts = TreeSample.objects.select_related(
            'sample__survey', 'sample__sample_area__parcel__region',
            'tree__species', 'tree__parcel',
        ).get(id=payload[ROW_ID])
        assert record == build_tree_sample_record(ts)
        assert len(record) == len(SAMPLED_TREE_COLUMNS)

    def test_tree_save_coppice_records_match_digest(
        self, writer_client, sample_setup, species, regions, eclasses,
    ):
        """Coppice multi-shoot creates return one record per shoot."""
        from apps.base.digests import build_tree_sample_record
        from apps.base.models import Parcel, SampleArea, TreeSample
        coppice_eclass = next(e for e in eclasses if e.coppice)
        parcel = Parcel.objects.create(
            name='cs', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('1.0'),
        )
        area = SampleArea.objects.create(
            sample_grid=sample_setup['grid'], parcel=parcel, number='1',
            lat=0.0, lon=0.0, r_m=12,
        )
        resp = self._post(writer_client, '/api/campionamenti/tree/save/', {
            FIELD_SURVEY_ID: str(sample_setup['survey'].id),
            FIELD_SAMPLE_AREA_ID: str(area.id),
            FIELD_SPECIES_ID: str(species[1].id),
            FIELD_NUMBER: '1', FIELD_FUSTAIA: 'false',
            'shoots': json.dumps([
                {FIELD_SHOOT: 1, FIELD_STANDARD: False, FIELD_D_CM: 5, FIELD_H_M: '8.0'},
                {FIELD_SHOOT: 2, FIELD_STANDARD: True,  FIELD_D_CM: 7, FIELD_H_M: '9.0'},
            ]),
            FIELD_LAT: '0', FIELD_LON: '0',
        })
        assert resp.status_code == 200, resp.content
        payload = resp.json()
        assert len(payload[RECORDS]) == 2
        ids = [r[0] for r in payload[RECORDS]]
        canonical = {
            ts.id: build_tree_sample_record(ts)
            for ts in TreeSample.objects.filter(id__in=ids).select_related(
                'sample__sample_area__parcel__region',
                'tree__species', 'tree__parcel',
            )
        }
        for record in payload[RECORDS]:
            assert record == canonical[record[0]]

    def test_tree_save_includes_sample_and_survey_records(
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
            FIELD_FUSTAIA: 'true',
        })
        from apps.base.digests import (
            SAMPLE_COLUMNS, SURVEY_COLUMNS, build_sample_record,
            build_survey_record,
        )
        payload = resp.json()
        assert payload[SAMPLE_RECORD][0] == s['sample'].id
        assert len(payload[SAMPLE_RECORD]) == len(SAMPLE_COLUMNS)
        s['sample'].refresh_from_db()
        assert payload[SAMPLE_RECORD] == build_sample_record(s['sample'])
        s['survey'].refresh_from_db()
        assert payload[SURVEY_RECORD] == build_survey_record(s['survey'])
        assert len(payload[SURVEY_RECORD]) == len(SURVEY_COLUMNS)

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
        assert payload[RECORD] == build_sample_area_record(area)
        s['grid'].refresh_from_db()
        assert payload[GRID_RECORD] == build_grid_record(s['grid'])
        # One survey in the fixture; assert it's in survey_records.
        s['survey'].refresh_from_db()
        assert any(
            r == build_survey_record(s['survey'])
            for r in payload[SURVEY_RECORDS]
        )

    def test_grid_save_returns_record(self, writer_client, db):
        from apps.base.digests import build_grid_record
        from apps.base.models import SampleGrid
        resp = self._post(writer_client, '/api/campionamenti/grid/save/', {
            FIELD_NAME: 'Griglia X', FIELD_DESCRIPTION: 'd',
        })
        payload = resp.json()
        grid = SampleGrid.objects.get(id=payload[ROW_ID])
        assert payload[RECORD] == build_grid_record(grid)

    def test_grid_edit_returns_record(self, writer_client, sample_setup):
        from apps.base.digests import build_grid_record
        s = sample_setup
        resp = self._post(
            writer_client,
            f'/api/campionamenti/grid/edit/{s["grid"].id}/',
            {FIELD_NAME: 'Renamed', FIELD_DESCRIPTION: ''},
        )
        payload = resp.json()
        s['grid'].refresh_from_db()
        assert payload[RECORD] == build_grid_record(s['grid'])

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
        assert payload[RECORD] == build_survey_record(survey)
        s['grid'].refresh_from_db()
        assert payload[GRID_RECORD] == build_grid_record(s['grid'])

    def test_survey_edit_returns_record(self, writer_client, sample_setup):
        from apps.base.digests import build_survey_record
        s = sample_setup
        resp = self._post(
            writer_client,
            f'/api/campionamenti/survey/edit/{s["survey"].id}/',
            {FIELD_NAME: 'Renamed', FIELD_DESCRIPTION: ''},
        )
        payload = resp.json()
        s['survey'].refresh_from_db()
        assert payload[RECORD] == build_survey_record(s['survey'])

    def test_tree_delete_returns_sample_and_survey(
        self, writer_client, sample_setup,
    ):
        from apps.base.digests import build_sample_record, build_survey_record
        from apps.base.models import TreeSample
        s = sample_setup
        ts_id = TreeSample.objects.first().id
        resp = writer_client.post(
            f'/api/campionamenti/tree/delete/{ts_id}/',
            content_type='application/json',
        )
        payload = resp.json()
        s['sample'].refresh_from_db()
        s['survey'].refresh_from_db()
        assert payload[SAMPLE_RECORD] == build_sample_record(s['sample'])
        assert payload[SURVEY_RECORD] == build_survey_record(s['survey'])


class TestTreeDelete:
    @staticmethod
    def _post(client, ts_id):
        return client.post(
            f'/api/campionamenti/tree/delete/{ts_id}/',
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
            'points': [
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

    def test_unknown_compresa_aborts(self, writer_client, sample_setup):
        n_before = SampleGrid.objects.count()
        resp = self._post(writer_client, {
            FIELD_NAME: 'Bad grid', FIELD_R_M: 12,
            'points': [
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
            'points': [
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
            'points': [
                {'compresa': sample_setup['area'].parcel.region.name,
                 'particella': sample_setup['area'].parcel.name,
                 FIELD_LAT: 38.5, FIELD_LON: 16.1},
            ],
        })
        assert resp.status_code == 400

    def test_empty_points_rejected(self, writer_client, db):
        resp = self._post(writer_client, {
            FIELD_NAME: 'Empty', FIELD_R_M: 12, 'points': [],
        })
        assert resp.status_code == 400

    def test_reader_forbidden(self, reader_client, sample_setup):
        resp = self._post(reader_client, {
            FIELD_NAME: 'X', FIELD_R_M: 12,
            'points': [
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
            content_type='application/json',
        )
        n_after = self._surveys_n_totali(writer_client, s['survey'].id,
                                         tmp_path, settings)
        assert n_after == n_before - 1


class TestGridCsvImport:
    URL = '/api/campionamenti/grid/import-csv/'

    @staticmethod
    def _post(client, grid_id, csv_text):
        from django.core.files.uploadedfile import SimpleUploadedFile
        return client.post(TestGridCsvImport.URL, {
            FIELD_SAMPLE_GRID_ID: str(grid_id),
            'file': SimpleUploadedFile(
                'grid.csv', csv_text.encode('utf-8-sig'),
                content_type='text/csv',
            ),
        })

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
        # Response shape mirrors area_save_view: record + area_records.
        assert RECORD in data and AREA_RECORDS in data
        assert len(data[AREA_RECORDS]) == 2

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

    def test_missing_grid_id(self, writer_client, db):
        from django.core.files.uploadedfile import SimpleUploadedFile
        resp = writer_client.post(self.URL, {
            'file': SimpleUploadedFile('x.csv', b'Compresa\n'),
        })
        assert resp.status_code == 400

    def test_grid_not_found(self, writer_client, db):
        from django.core.files.uploadedfile import SimpleUploadedFile
        resp = writer_client.post(self.URL, {
            FIELD_SAMPLE_GRID_ID: '999999',
            'file': SimpleUploadedFile('x.csv', b'Compresa\n'),
        })
        assert resp.status_code == 404

    def test_missing_required_column(self, writer_client, sample_setup):
        grid = SampleGrid.objects.create(name='Bad cols')
        csv_text = (
            'Compresa,Particella,Area saggio,Lon,Lat\n'    # missing Quota,Raggio
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
        from django.core.files.uploadedfile import SimpleUploadedFile
        grid = sample_setup['grid']
        resp = reader_client.post(self.URL, {
            FIELD_SAMPLE_GRID_ID: str(grid.id),
            'file': SimpleUploadedFile('x.csv', b'a,b\n1,2'),
        })
        assert resp.status_code == 403


class TestTreeCsvImport:
    URL = '/api/campionamenti/survey/import-csv/'

    @staticmethod
    def _post(client, survey_id, csv_text, default_date=''):
        from django.core.files.uploadedfile import SimpleUploadedFile
        return client.post(TestTreeCsvImport.URL, {
            FIELD_SURVEY_ID: str(survey_id),
            FIELD_DEFAULT_DATE: default_date,
            'file': SimpleUploadedFile(
                'trees.csv', csv_text.encode('utf-8-sig'),
                content_type='text/csv',
            ),
        })

    def test_happy_path(self, writer_client, sample_setup):
        """Importing rows into an empty survey creates samples + trees +
        tree_samples."""
        from apps.base.models import Survey, Tree, TreeSample
        s = sample_setup
        # Use a fresh empty survey so we're not measuring fixture trees.
        empty_survey = Survey.objects.create(
            name='CSV import target', sample_grid=s['grid'],
        )
        compresa = s['area'].parcel.region.name
        particella = s['area'].parcel.name
        adc = s['area'].number
        n_trees_before = Tree.objects.count()
        n_ts_before = TreeSample.objects.count()
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Genere,Fustaia,Data\n'
            f'{compresa},{particella},{adc},10,0,false,30,20.5,10,Abete,true,'
            '2024-09-15\n'
            f'{compresa},{particella},{adc},11,0,false,32,22.5,11,Abete,true,'
            '2024-09-15\n'
        )
        resp = self._post(writer_client, empty_survey.id, csv_text)
        assert resp.status_code == 200, resp.content
        data = resp.json()
        assert data['n_samples'] == 1
        assert data['n_trees'] == 2
        assert Tree.objects.count() == n_trees_before + 2
        assert TreeSample.objects.count() == n_ts_before + 2

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
            'D_cm,H_m,L10_mm,Genere,Fustaia\n'
            f'{s["area"].parcel.region.name},{s["area"].parcel.name},'
            f'{s["area"].number},10,0,false,30,20.5,10,Abete,true\n'
        )
        resp = self._post(writer_client, s['survey'].id, csv_text)
        assert resp.status_code == 400

    def test_default_date_used_when_no_data_column(self, writer_client,
                                                  sample_setup):
        from apps.base.models import Sample
        s = sample_setup
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Genere,Fustaia\n'
            f'{s["area"].parcel.region.name},{s["area"].parcel.name},'
            f'{s["area"].number},99,0,false,30,20.5,10,Abete,true\n'
        )
        resp = self._post(writer_client, s['survey'].id, csv_text,
                          default_date='2025-06-01')
        assert resp.status_code == 200, resp.content

    def test_empty_survey_id_returns_clean_400(self, writer_client, sample_setup):
        """User-reported bug: leaving the target-survey pulldown on
        '— Seleziona —' used to silently fail (HTML5 `required` blocked the
        submit and our JS handler never ran).  The form is now `novalidate`
        and we rely on the server to return a friendly 400 — make sure
        that path renders the error message clients can show."""
        from django.core.files.uploadedfile import SimpleUploadedFile
        resp = writer_client.post(self.URL, {
            FIELD_SURVEY_ID: '',      # empty: user didn't pick a survey
            FIELD_DEFAULT_DATE: '',
            'file': SimpleUploadedFile('x.csv', b'a,b\n1,2'),
        })
        assert resp.status_code == 400
        body = resp.json()
        assert body[MESSAGE]    # non-empty user-facing message

    def test_unknown_area_reports_error(self, writer_client, sample_setup):
        s = sample_setup
        csv_text = (
            'Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
            'D_cm,H_m,L10_mm,Genere,Fustaia,Data\n'
            f'{s["area"].parcel.region.name},{s["area"].parcel.name},'
            'ZZZ,1,0,false,30,20,0,Abete,true,2024-09-15\n'
        )
        resp = self._post(writer_client, s['survey'].id, csv_text)
        assert resp.status_code == 400
        body = resp.json()
        assert any('ZZZ' in e for e in body[FIELD_ERRORS])

    def test_reader_forbidden(self, reader_client, sample_setup):
        from django.core.files.uploadedfile import SimpleUploadedFile
        resp = reader_client.post(self.URL, {
            FIELD_SURVEY_ID: str(sample_setup['survey'].id),
            'file': SimpleUploadedFile('x.csv', b'a,b\n1,2'),
        })
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
                          {FIELD_NAME: 'Rinominata', FIELD_DESCRIPTION: 'desc'})
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
                          {FIELD_NAME: 'Other grid X'})
        assert resp.status_code == 400

    def test_delete_grid_in_use_refused(self, writer_client, sample_setup):
        """Grid with surveys is protected (Survey.sample_grid on_delete=PROTECT
        plus the explicit pre-check)."""
        s = sample_setup
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/delete/{s["grid"].id}/')
        assert resp.status_code == 400
        assert SampleGrid.objects.filter(id=s['grid'].id).exists()

    def test_delete_unused_grid(self, writer_client, db):
        """An empty grid (no surveys, no areas) deletes cleanly."""
        g = SampleGrid.objects.create(name='Empty grid')
        resp = self._post(writer_client,
                          f'/api/campionamenti/grid/delete/{g.id}/')
        assert resp.status_code == 200
        assert not SampleGrid.objects.filter(id=g.id).exists()

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
                          f'/api/campionamenti/grid/delete/{g.id}/')
        assert resp.status_code == 200
        assert not SampleArea.objects.filter(sample_grid=g).exists()

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
                          {FIELD_NAME: 'Sv rinominato', FIELD_DESCRIPTION: 'desc'})
        assert resp.status_code == 200
        s['survey'].refresh_from_db()
        assert s['survey'].name == 'Sv rinominato'

    def test_edit_survey_duplicate_name(self, writer_client, sample_setup):
        s = sample_setup
        Survey.objects.create(name='Other survey X', sample_grid=s['grid'])
        resp = self._post(writer_client,
                          f'/api/campionamenti/survey/edit/{s["survey"].id}/',
                          {FIELD_NAME: 'Other survey X'})
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
                          f'/api/campionamenti/survey/delete/{survey_id}/')
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
                          f'/api/campionamenti/survey/delete/{empty.id}/')
        assert resp.status_code == 200
        assert not Survey.objects.filter(id=empty.id).exists()

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
        # Modal carries both creation paths per campionamenti.md §2.
        assert 'data-path="empty"' in html
        assert 'data-path="csv"' in html
        assert 'campionamenti-survey-form-empty' in html
        # Grid pulldown contains the fixture's grid.
        assert sample_setup['grid'].name in html

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
