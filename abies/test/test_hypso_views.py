"""Tests for the hypsometric-parameters settings endpoints."""

import base64
from datetime import date
from decimal import Decimal
import gzip
import json

import pytest
from django.test import Client
from django.urls import reverse

from apps.base import hypsometry
from apps.base.digests import build_hypso_param_record, generate_hypso_params
from apps.base.models import (
    DigestStatus, HYPSO_FUNC_LN, HypsoParam, HypsoParamSet,
    HypsoParamSource, Sample, Survey, Tree, TreeSample, UsedNonce,
)
from config import strings as S
from config.constants import (
    COLUMNS, DIGEST_HYPSO_PARAMS, DIGEST_PARCEL_DENDROMETRY_POINTS,
    FIELD_FILE, FIELD_ID, FIELD_MIN_N, FIELD_NAME, FIELD_NONCE, FIELD_SOURCE,
    FIELD_SURVEY_IDS, FIELD_SURVEYS, FIELD_TREES, FIELD_USE_FOR_HEIGHT_PLOTS,
    ROWS,
)

URL_DATA = reverse('impostazioni-hypso-data')
URL_ACTIVE = reverse('impostazioni-hypso-active-set')
URL_SURVEYS = reverse('impostazioni-hypso-surveys')
URL_COMPUTE = reverse('impostazioni-hypso-compute')
URL_ACCEPT = reverse('impostazioni-hypso-accept')
URL_IMPORT = reverse('impostazioni-hypso-import')
URL_EXPORT = reverse('impostazioni-hypso-export')
URL_CLEAR = reverse('impostazioni-hypso-clear')

CSV_HEADER = ','.join([
    S.CSV_COL_REGION.lower(), S.CSV_COL_SPECIES.lower(), S.CSV_COL_FUNCTION,
    S.CSV_COL_A, S.CSV_COL_B, S.CSV_COL_R2, S.CSV_COL_N_REGRESSION,
])


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


def _post_json(client, url, body):
    return client.post(url, data=json.dumps(body),
                       content_type='application/json')


def _compute_body(hypso_samples, min_n=5, use_for_height_plots=None):
    body = {FIELD_MIN_N: min_n, FIELD_SURVEY_IDS: [hypso_samples['survey'].id]}
    if use_for_height_plots is not None:
        body[FIELD_USE_FOR_HEIGHT_PLOTS] = use_for_height_plots
    return body


def _persist(
        hypso_samples, min_n=5, survey_ids=None, use_for_height_plots=False,
):
    rows = hypsometry.compute_params([hypso_samples['survey'].id], min_n)
    return hypsometry.replace_active_set(
        rows, source=HypsoParamSource.COMPUTED, min_n=min_n,
        survey_ids=survey_ids or [],
        use_for_height_plots=use_for_height_plots,
    )


def _read_gzip_json(resp):
    return json.loads(gzip.decompress(resp.getvalue()))


def _csv_b64(content):
    raw = content if isinstance(content, bytes) else content.encode('utf-8')
    return base64.b64encode(raw).decode('ascii')


def _make_unstructured_hypso_survey(parcel, species, *, name='Unstructured hypso',
                                    h_measured=True, count=5):
    survey = Survey.objects.create(name=name)
    sample = Sample.objects.create(
        survey=survey, sample_area=None, date=date(2026, 7, 17),
    )
    heights = [Decimal('10.00'), Decimal('12.00'), Decimal('13.00'),
               Decimal('15.00'), Decimal('16.00'), Decimal('18.00')]
    for i in range(count):
        tree = Tree.objects.create(species=species, parcel=parcel, coppice=False)
        TreeSample.objects.create(
            sample=sample, tree=tree, parcel=parcel, number=i + 1,
            shoot=0, standard=False,
            d_cm=10 + (i * 5), h_m=heights[i], h_measured=h_measured,
        )
    return survey


class TestSourceSurveys:
    def test_lists_measured_non_coppice_counts_for_eligible_surveys(
            self, writer_client, hypso_samples, parcels, species,
    ):
        unstructured = _make_unstructured_hypso_survey(parcels[0], species[0])
        hidden = _make_unstructured_hypso_survey(
            parcels[0], species[0], name='Derived heights only', h_measured=False,
        )

        resp = writer_client.get(URL_SURVEYS)

        assert resp.status_code == 200
        rows = {row[FIELD_NAME]: row for row in resp.json()[FIELD_SURVEYS]}
        assert (
            rows[hypso_samples['survey'].name][FIELD_ID]
            == hypso_samples['survey'].id
        )
        assert rows[hypso_samples['survey'].name][FIELD_TREES] == 15
        assert rows[unstructured.name][FIELD_ID] == unstructured.id
        assert rows[unstructured.name][FIELD_TREES] == 5
        assert hidden.name not in rows


class TestCompute:
    def test_returns_candidate_without_persisting(self, writer_client, hypso_samples):
        resp = _post_json(writer_client, URL_COMPUTE, _compute_body(hypso_samples))
        assert resp.status_code == 200
        assert len(resp.json()[ROWS]) == 1
        assert HypsoParamSet.objects.count() == 0

    def test_uses_only_measured_height_rows(self, writer_client, hypso_samples):
        ts = (TreeSample.objects
              .filter(
                  sample__survey=hypso_samples['survey'],
                  tree__species=hypso_samples['species'],
                  tree__coppice=False,
              )
              .order_by('id')
              .first())
        ts.h_measured = False
        ts.save()

        resp = _post_json(
            writer_client, URL_COMPUTE, _compute_body(hypso_samples, min_n=12),
        )

        assert resp.status_code == 200
        assert resp.json()[ROWS] == []

    def test_unstructured_measured_rows_contribute(
            self, writer_client, parcels, species,
    ):
        survey = _make_unstructured_hypso_survey(parcels[0], species[0])
        resp = _post_json(
            writer_client, URL_COMPUTE,
            {FIELD_MIN_N: 5, FIELD_SURVEY_IDS: [survey.id]},
        )

        assert resp.status_code == 200
        payload = resp.json()
        assert len(payload[ROWS]) == 1
        row = payload[ROWS][0]
        assert row[payload[COLUMNS].index(S.COL_REGION)] == parcels[0].region.name
        assert row[payload[COLUMNS].index(S.COL_N_REGRESSION)] == 5

    def test_requires_surveys(self, writer_client, hypso_samples):
        resp = _post_json(writer_client, URL_COMPUTE,
                          {FIELD_MIN_N: 5, FIELD_SURVEY_IDS: []})
        assert resp.status_code == 400

    def test_rejects_non_positive_min_n(self, writer_client, hypso_samples):
        resp = _post_json(writer_client, URL_COMPUTE,
                          _compute_body(hypso_samples, min_n=0))
        assert resp.status_code == 400


class TestAccept:
    def test_persists_active_set(
        self, writer_client, hypso_samples, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        resp = _post_json(writer_client, URL_ACCEPT, _compute_body(hypso_samples))
        assert resp.status_code == 200
        active = hypsometry.active_set()
        assert active is not None
        assert active.source == HypsoParamSource.COMPUTED
        assert active.use_for_height_plots is False
        assert HypsoParam.objects.filter(param_set=active).count() == 1

    def test_persists_height_plot_choice(
            self, writer_client, hypso_samples, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        resp = _post_json(
            writer_client, URL_ACCEPT,
            _compute_body(hypso_samples, use_for_height_plots=True),
        )
        assert resp.status_code == 200
        assert hypsometry.active_set().use_for_height_plots is True

    def test_archives_prior_set(self, writer_client, hypso_samples, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        _post_json(writer_client, URL_ACCEPT, _compute_body(hypso_samples))
        _post_json(writer_client, URL_ACCEPT, _compute_body(hypso_samples))
        assert HypsoParamSet.objects.count() == 2
        assert HypsoParamSet.objects.filter(superseded_at__isnull=True).count() == 1


class TestImportExportClear:
    def _csv_upload(self, regions, species):
        content = (
            CSV_HEADER + '\n'
            + f'{regions[0].name},{species[0].common_name},'
              f'{HYPSO_FUNC_LN},7.0,-4.0,0.6,30\n'
        )
        return _csv_b64(content)

    def test_import_replaces_active_set(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        resp = _post_json(
            writer_client, URL_IMPORT,
            {FIELD_FILE: self._csv_upload(regions, species)},
        )
        assert resp.status_code == 200
        active = hypsometry.active_set()
        assert active is not None
        assert active.source == HypsoParamSource.IMPORTED
        assert HypsoParam.objects.filter(param_set=active).count() == 1

    def test_import_aborts_on_unknown_region(
        self, writer_client, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        content = (CSV_HEADER + '\n'
                   + f'Nessuna,{species[0].common_name},{HYPSO_FUNC_LN},7,-4,0.5,20\n')
        resp = _post_json(writer_client, URL_IMPORT, {FIELD_FILE: _csv_b64(content)})
        assert resp.status_code == 400
        assert HypsoParamSet.objects.count() == 0

    def test_import_rejects_non_utf8(self, writer_client, db, tmp_path, settings):
        """A non-UTF-8 upload is reported cleanly (not a 500) — it flows
        through parse_param_csv → csv_io decode."""
        settings.DIGEST_DIR = tmp_path
        resp = _post_json(
            writer_client, URL_IMPORT,
            {FIELD_FILE: _csv_b64(b'\xff\xfe not utf-8 bytes')},
        )
        assert resp.status_code == 400
        assert HypsoParamSet.objects.count() == 0

    def test_import_requires_file(self, writer_client, db):
        assert _post_json(writer_client, URL_IMPORT, {}).status_code == 400

    def test_import_saves_nonce(self, writer_client, regions, species, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = _post_json(writer_client, URL_IMPORT, {
            FIELD_FILE: self._csv_upload(regions, species),
            FIELD_NONCE: 'hypso-import-nonce',
        })
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='hypso-import-nonce').exists()

    def _upload_csv(self, writer_client, *data_rows):
        content = CSV_HEADER + '\n' + ''.join(r + '\n' for r in data_rows)
        return _post_json(
            writer_client, URL_IMPORT, {FIELD_FILE: _csv_b64(content)},
        )

    def test_import_rejects_out_of_range_r2(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        resp = self._upload_csv(
            writer_client,
            f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},7,-4,5.0,20',
        )
        assert resp.status_code == 400
        assert HypsoParamSet.objects.count() == 0

    def test_import_rejects_duplicate_pair(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        line = f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},7,-4,0.6,20'
        resp = self._upload_csv(writer_client, line, line)
        assert resp.status_code == 400
        assert HypsoParamSet.objects.count() == 0

    def test_import_is_all_or_nothing(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        resp = self._upload_csv(
            writer_client,
            f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},7,-4,0.6,20',
            f'Nessuna,{species[0].common_name},{HYPSO_FUNC_LN},7,-4,0.6,20',
        )
        assert resp.status_code == 400
        assert HypsoParamSet.objects.count() == 0

    def test_import_rejects_header_only(self, writer_client, db, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        assert _post_json(
            writer_client, URL_IMPORT, {FIELD_FILE: _csv_b64(CSV_HEADER + '\n')},
        ).status_code == 400

    def test_import_accepts_bom_encoded_file(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        content = (
            CSV_HEADER + '\n'
            + f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},7,-4,0.6,20\n'
        )
        resp = _post_json(
            writer_client, URL_IMPORT,
            {FIELD_FILE: _csv_b64(content.encode('utf-8-sig'))},
        )
        assert resp.status_code == 200
        assert hypsometry.active_set() is not None

    def test_export_with_no_active_set_is_header_only(self, writer_client, db):
        resp = writer_client.get(URL_EXPORT)
        assert resp.status_code == 200
        lines = resp.content.decode('utf-8').strip().splitlines()
        assert len(lines) == 1

    def test_export_streams_active_params(self, writer_client, hypso_samples):
        from apps.base import csv_io
        _persist(hypso_samples)
        resp = writer_client.get(URL_EXPORT)
        assert resp.status_code == 200
        body = resp.content.decode('utf-8')
        assert hypso_samples['species'].common_name in body
        # Column order matches the settings table: ..., a, b, n, r2.  Field
        # separator follows the install locale.
        delimiter, _ = csv_io.export_format()
        assert body.splitlines()[0].split(delimiter) == [
            S.CSV_COL_REGION.lower(), S.CSV_COL_SPECIES.lower(),
            S.CSV_COL_FUNCTION, S.CSV_COL_A, S.CSV_COL_B,
            S.CSV_COL_N_REGRESSION, S.CSV_COL_R2,
        ]

    def test_clear_archives_active(self, writer_client, hypso_samples, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        _persist(hypso_samples)
        assert _post_json(writer_client, URL_CLEAR, {}).status_code == 200
        assert hypsometry.active_set() is None


class TestActiveSet:
    def test_metadata_for_computed_set(self, writer_client, hypso_samples):
        _persist(
            hypso_samples, survey_ids=[hypso_samples['survey'].id],
            use_for_height_plots=True,
        )
        d = writer_client.get(URL_ACTIVE).json()
        assert d[FIELD_SOURCE] == HypsoParamSource.COMPUTED
        assert d[FIELD_MIN_N] == 5
        assert d[FIELD_SURVEYS] == [hypso_samples['survey'].name]
        assert d[FIELD_USE_FOR_HEIGHT_PLOTS] is True

    def test_no_active_set(self, writer_client, db):
        assert writer_client.get(URL_ACTIVE).json()[FIELD_SOURCE] is None


class TestPermissions:
    def test_reader_cannot_compute(self, reader_client, hypso_samples):
        resp = _post_json(reader_client, URL_COMPUTE, _compute_body(hypso_samples))
        assert resp.status_code == 403

    def test_reader_cannot_clear(self, reader_client, db):
        assert _post_json(reader_client, URL_CLEAR, {}).status_code == 403

    def test_anonymous_blocked(self, client, db):
        assert client.get(URL_DATA).status_code in (302, 403)


class TestDigestInvalidation:
    """Each write to the active parameter set must refresh the served
    `hypso_params` digest — the mark form reads a/b from it. Every test
    performs a write via the public endpoint, re-reads the digest endpoint,
    and asserts the change is reflected."""

    def test_accept_refreshes_digest(
        self, writer_client, hypso_samples, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        assert _read_gzip_json(writer_client.get(URL_DATA))[ROWS] == []
        _post_json(
            writer_client, URL_ACCEPT,
            _compute_body(hypso_samples, use_for_height_plots=True),
        )
        assert len(_read_gzip_json(writer_client.get(URL_DATA))[ROWS]) == 1
        assert DigestStatus.objects.get(
            name=DIGEST_PARCEL_DENDROMETRY_POINTS,
        ).stale is True

    def test_imported_values_reach_the_served_digest(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        # The mark form reads a/b from this digest; lock that an imported
        # coefficient lands intact (the auto-fill h = a*ln(D)+b is JS-side).
        settings.DIGEST_DIR = tmp_path
        content = (
            CSV_HEADER + '\n'
            + f'{regions[0].name},{species[0].common_name},'
              f'{HYPSO_FUNC_LN},10,-10,0.6,20\n'
        )
        resp = _post_json(
            writer_client, URL_IMPORT, {FIELD_FILE: _csv_b64(content)},
        )
        assert resp.status_code == 200
        digest = _read_gzip_json(writer_client.get(URL_DATA))
        cols = digest[COLUMNS]
        row = next(
            r for r in digest[ROWS]
            if r[cols.index(S.COL_REGION)] == regions[0].name
            and r[cols.index(S.COL_SPECIES)] == species[0].common_name
        )
        assert row[cols.index(S.COL_FUNCTION)] == HYPSO_FUNC_LN
        assert row[cols.index(S.COL_A)] == 10.0
        assert row[cols.index(S.COL_B)] == -10.0
        assert DigestStatus.objects.get(
            name=DIGEST_PARCEL_DENDROMETRY_POINTS,
        ).stale is True

    def test_clear_empties_the_served_digest(
        self, writer_client, hypso_samples, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        _post_json(writer_client, URL_ACCEPT, _compute_body(hypso_samples))
        assert len(_read_gzip_json(writer_client.get(URL_DATA))[ROWS]) == 1
        assert _post_json(writer_client, URL_CLEAR, {}).status_code == 200
        assert _read_gzip_json(writer_client.get(URL_DATA))[ROWS] == []
        assert DigestStatus.objects.get(
            name=DIGEST_PARCEL_DENDROMETRY_POINTS,
        ).stale is True

    def test_build_record_matches_generator(self, hypso_samples, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        s = _persist(hypso_samples)
        generate_hypso_params()
        path = settings.DIGEST_DIR / f'{DIGEST_HYPSO_PARAMS}.json.gz'
        digest = json.loads(gzip.decompress(path.read_bytes()))
        p = HypsoParam.objects.get(param_set=s)
        assert digest[ROWS][0] == build_hypso_param_record(p)
