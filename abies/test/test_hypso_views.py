"""Tests for the hypsometric-parameters settings endpoints."""

import gzip
import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse

from apps.base import hypsometry
from apps.base.digests import build_hypso_param_record, generate_hypso_params
from apps.base.models import (
    HYPSO_FUNC_LN, HypsoParam, HypsoParamSet, HypsoParamSource,
)
from config import strings as S
from config.constants import (
    COLUMNS, DIGEST_HYPSO_PARAMS, FIELD_FILE, FIELD_MIN_N, FIELD_SOURCE,
    FIELD_SURVEY_IDS, FIELD_SURVEYS, ROWS,
)

URL_DATA = reverse('impostazioni-hypso-data')
URL_ACTIVE = reverse('impostazioni-hypso-active-set')
URL_COMPUTE = reverse('impostazioni-hypso-compute')
URL_ACCEPT = reverse('impostazioni-hypso-accept')
URL_IMPORT = reverse('impostazioni-hypso-import')
URL_EXPORT = reverse('impostazioni-hypso-export')
URL_CLEAR = reverse('impostazioni-hypso-clear')

CSV_HEADER = ','.join([
    S.CSV_COL_COMPRESA.lower(), S.CSV_COL_GENERE.lower(), S.CSV_COL_FUNZIONE,
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


def _compute_body(hypso_samples, min_n=5):
    return {FIELD_MIN_N: min_n, FIELD_SURVEY_IDS: [hypso_samples['survey'].id]}


def _persist(hypso_samples, min_n=5, survey_ids=None):
    rows = hypsometry.compute_params([hypso_samples['survey'].id], min_n)
    return hypsometry.replace_active_set(
        rows, source=HypsoParamSource.COMPUTED, min_n=min_n,
        survey_ids=survey_ids or [],
    )


def _read_gzip_json(resp):
    return json.loads(gzip.decompress(resp.getvalue()))


class TestCompute:
    def test_returns_candidate_without_persisting(self, writer_client, hypso_samples):
        resp = _post_json(writer_client, URL_COMPUTE, _compute_body(hypso_samples))
        assert resp.status_code == 200
        assert len(resp.json()[ROWS]) == 1
        assert HypsoParamSet.objects.count() == 0

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
        assert HypsoParam.objects.filter(param_set=active).count() == 1

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
        return SimpleUploadedFile('e.csv', content.encode('utf-8'),
                                  content_type='text/csv')

    def test_import_replaces_active_set(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.post(
            URL_IMPORT, {FIELD_FILE: self._csv_upload(regions, species)})
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
        f = SimpleUploadedFile('e.csv', content.encode('utf-8'))
        resp = writer_client.post(URL_IMPORT, {FIELD_FILE: f})
        assert resp.status_code == 400
        assert HypsoParamSet.objects.count() == 0

    def test_import_rejects_non_utf8(self, writer_client, db, tmp_path, settings):
        """A non-UTF-8 upload is reported cleanly (not a 500) — it flows
        through parse_param_csv → csv_io decode."""
        settings.DIGEST_DIR = tmp_path
        f = SimpleUploadedFile('e.csv', b'\xff\xfe not utf-8 bytes')
        resp = writer_client.post(URL_IMPORT, {FIELD_FILE: f})
        assert resp.status_code == 400
        assert HypsoParamSet.objects.count() == 0

    def test_import_requires_file(self, writer_client, db):
        assert writer_client.post(URL_IMPORT).status_code == 400

    def _upload_csv(self, writer_client, *data_rows):
        content = CSV_HEADER + '\n' + ''.join(r + '\n' for r in data_rows)
        return writer_client.post(
            URL_IMPORT,
            {FIELD_FILE: SimpleUploadedFile('e.csv', content.encode('utf-8'))},
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
        f = SimpleUploadedFile('e.csv', (CSV_HEADER + '\n').encode('utf-8'))
        assert writer_client.post(URL_IMPORT, {FIELD_FILE: f}).status_code == 400

    def test_import_accepts_bom_encoded_file(
        self, writer_client, regions, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        content = (
            CSV_HEADER + '\n'
            + f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},7,-4,0.6,20\n'
        )
        f = SimpleUploadedFile('e.csv', content.encode('utf-8-sig'))
        resp = writer_client.post(URL_IMPORT, {FIELD_FILE: f})
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
            S.CSV_COL_COMPRESA.lower(), S.CSV_COL_GENERE.lower(),
            S.CSV_COL_FUNZIONE, S.CSV_COL_A, S.CSV_COL_B,
            S.CSV_COL_N_REGRESSION, S.CSV_COL_R2,
        ]

    def test_clear_archives_active(self, writer_client, hypso_samples, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        _persist(hypso_samples)
        assert writer_client.post(URL_CLEAR).status_code == 200
        assert hypsometry.active_set() is None


class TestActiveSet:
    def test_metadata_for_computed_set(self, writer_client, hypso_samples):
        _persist(hypso_samples, survey_ids=[hypso_samples['survey'].id])
        d = writer_client.get(URL_ACTIVE).json()
        assert d[FIELD_SOURCE] == HypsoParamSource.COMPUTED
        assert d[FIELD_MIN_N] == 5
        assert d[FIELD_SURVEYS] == [hypso_samples['survey'].name]

    def test_no_active_set(self, writer_client, db):
        assert writer_client.get(URL_ACTIVE).json()[FIELD_SOURCE] is None


class TestPermissions:
    def test_reader_cannot_compute(self, reader_client, hypso_samples):
        resp = _post_json(reader_client, URL_COMPUTE, _compute_body(hypso_samples))
        assert resp.status_code == 403

    def test_reader_cannot_clear(self, reader_client, db):
        assert reader_client.post(URL_CLEAR).status_code == 403

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
        _post_json(writer_client, URL_ACCEPT, _compute_body(hypso_samples))
        assert len(_read_gzip_json(writer_client.get(URL_DATA))[ROWS]) == 1

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
        resp = writer_client.post(
            URL_IMPORT,
            {FIELD_FILE: SimpleUploadedFile('e.csv', content.encode('utf-8'))},
        )
        assert resp.status_code == 200
        digest = _read_gzip_json(writer_client.get(URL_DATA))
        cols = digest[COLUMNS]
        row = next(
            r for r in digest[ROWS]
            if r[cols.index(S.COL_COMPRESA)] == regions[0].name
            and r[cols.index(S.COL_SPECIES)] == species[0].common_name
        )
        assert row[cols.index(S.COL_FUNCTION)] == HYPSO_FUNC_LN
        assert row[cols.index(S.COL_A)] == 10.0
        assert row[cols.index(S.COL_B)] == -10.0

    def test_clear_empties_the_served_digest(
        self, writer_client, hypso_samples, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        _post_json(writer_client, URL_ACCEPT, _compute_body(hypso_samples))
        assert len(_read_gzip_json(writer_client.get(URL_DATA))[ROWS]) == 1
        assert writer_client.post(URL_CLEAR).status_code == 200
        assert _read_gzip_json(writer_client.get(URL_DATA))[ROWS] == []

    def test_build_record_matches_generator(self, hypso_samples, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        s = _persist(hypso_samples)
        generate_hypso_params()
        path = settings.DIGEST_DIR / f'{DIGEST_HYPSO_PARAMS}.json.gz'
        digest = json.loads(gzip.decompress(path.read_bytes()))
        p = HypsoParam.objects.get(param_set=s)
        assert digest[ROWS][0] == build_hypso_param_record(p)
