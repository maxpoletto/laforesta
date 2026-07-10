"""Stage converted mark CSVs as Ipso inbox uploads."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import IntegrityError, transaction

from apps.base.numparse import coord_float
from apps.base.models import Parcel, Species
from apps.piano_di_taglio import csv_plan
from apps.piano_di_taglio.mark_import import MARK_CSV_SPECIES_HEADERS
from apps.ipso.models import IpsoUpload
from apps.ipso import staging
from config import strings as S
from config.constants import (
    FIELD_ACC_M, FIELD_CLIENT_RECORD_ID, FIELD_COMPLETED_AT, FIELD_DAMAGED,
    FIELD_DATE, FIELD_D_CM, FIELD_H_M, FIELD_H_MEASURED, FIELD_HYPSO_PARAM_SET_ID,
    FIELD_LAT, FIELD_LON, FIELD_MODE, FIELD_NUMBER, FIELD_OPERATOR,
    FIELD_PARCEL_ID, FIELD_REFERENCE_VERSION, FIELD_REGION_ID,
    FIELD_SCHEMA_VERSION, FIELD_SESSION_ID, FIELD_SPECIES_ID,
    FIELD_WORK_PACKAGE_ID, IPSO_MODE_MARTELLATE,
    IPSO_REFERENCE_LEGACY_CONVERTED, RECORDS, SESSION,
    TREE_H_QUANTUM, parse_bool,
)

SESSION_NAMESPACE = uuid.UUID('6bde56d0-9184-4962-9b61-7df0e16e3d2d')

REQUIRED = {
    'date': [S.CSV_COL_DATA],
    'compresa': [S.CSV_COL_REGION],
    'particella': [S.CSV_COL_PARCEL],
    'species': MARK_CSV_SPECIES_HEADERS,
    'd_cm': [S.CSV_COL_D_CM],
    'h_m': [S.CSV_COL_H_M],
}
OPTIONAL = {
    'damaged': [S.CSV_COL_DAMAGED],
    'number': [S.CSV_COL_NUMBER],
    'h_measured': [S.CSV_COL_H_MEASURED],
    'lat': [S.CSV_COL_LAT],
    'lon': [S.CSV_COL_LON],
    'acc_m': [S.CSV_COL_ACC_M],
    'operator': [S.CSV_COL_OPERATOR],
}


class Command(BaseCommand):
    help = 'Stage converted mark CSVs as received Ipso uploads.'

    def add_arguments(self, parser):
        parser.add_argument(
            'source_dir', nargs='?', default='data/canonical/marks',
            help='Directory containing converted mark CSV files.',
        )

    def handle(self, *args, **options):
        source_dir = Path(options['source_dir'])
        Path(settings.IPSO_INBOX_DIR).mkdir(parents=True, exist_ok=True)
        if not source_dir.is_dir():
            self.stdout.write(f'No mark upload directory found: {source_dir}')
            return

        parcels = _parcel_map()
        species = _species_map()
        staged = 0
        skipped = 0
        for path in sorted(source_dir.glob('*.csv')):
            payload, csv_text = _payload_from_csv(path, parcels, species)
            checksum = staging.payload_checksum(payload)
            session_id = payload[SESSION][FIELD_SESSION_ID]
            existing = IpsoUpload.objects.filter(session_id=session_id).first()
            if existing is not None:
                if existing.checksum == checksum:
                    skipped += 1
                    continue
                raise CommandError(
                    f'{path}: session {session_id} already exists with different content'
                )

            inbox_path = staging.upload_inbox_path(session_id)
            try:
                with transaction.atomic():
                    IpsoUpload.objects.create(
                        **staging.upload_model_fields(payload, checksum, inbox_path)
                    )
                    staging.write_upload_files(inbox_path, payload, checksum, csv_text)
            except IntegrityError as exc:
                raise CommandError(f'{path}: could not stage upload') from exc
            staged += 1

        self.stdout.write(f'Staged {staged} mark upload(s); skipped {skipped}.')


def _payload_from_csv(path: Path, parcels: dict, species: dict) -> tuple[dict, str]:
    csv_text = path.read_text(encoding='utf-8-sig')
    result = csv_plan.read_optional(path.read_bytes(), required=REQUIRED, optional=OPTIONAL)
    if isinstance(result, csv_plan.CsvError):
        raise CommandError(f'{path}: {result.message}')
    if result is None or not result.rows:
        raise CommandError(f'{path}: empty mark upload')

    reader = result.reader
    records = []
    damaged = False
    operators = set()
    region_ids = set()
    for i, row in enumerate(result.rows, start=1):
        record, record_damaged = _record_from_row(path, i, reader, row, parcels, species)
        records.append(record)
        damaged = damaged or record_damaged
        region_ids.add(record[FIELD_REGION_ID])
        operator = record.get(FIELD_OPERATOR, '')
        if operator:
            operators.add(operator)

    session_id = _session_id(path, csv_text)
    session = {
        FIELD_SESSION_ID: session_id,
        FIELD_MODE: IPSO_MODE_MARTELLATE,
        FIELD_SCHEMA_VERSION: 1,
        FIELD_REFERENCE_VERSION: IPSO_REFERENCE_LEGACY_CONVERTED,
        FIELD_WORK_PACKAGE_ID: '',
        FIELD_OPERATOR: next(iter(operators)) if len(operators) == 1 else '',
        FIELD_COMPLETED_AT: '',
        FIELD_DAMAGED: damaged,
    }
    if len(region_ids) == 1:
        session[FIELD_REGION_ID] = next(iter(region_ids))
    return {SESSION: session, RECORDS: records}, csv_text


def _record_from_row(path: Path, index: int, reader, row: dict, parcels: dict, species: dict):
    compresa = row['compresa'].strip()
    particella = row['particella'].strip()
    species_name = row['species'].strip()
    parcel = parcels.get((compresa.lower(), particella.lower()))
    if parcel is None:
        raise CommandError(f'{path}: row {index}: unknown parcel {compresa}/{particella}')
    sp = species.get(species_name.lower())
    if sp is None:
        raise CommandError(f'{path}: row {index}: unknown species {species_name}')

    d_cm = reader.integer(row.get('d_cm'))
    h_m = reader.decimal(row.get('h_m'))
    if d_cm is None or d_cm <= 0:
        raise CommandError(f'{path}: row {index}: invalid {S.CSV_COL_D_CM}')
    if h_m is None or h_m <= 0:
        raise CommandError(f'{path}: row {index}: invalid {S.CSV_COL_H_M}')

    record = {
        FIELD_CLIENT_RECORD_ID: str(index),
        FIELD_DATE: _parse_date(path, index, row['date']).isoformat(),
        FIELD_REGION_ID: parcel.region_id,
        FIELD_PARCEL_ID: parcel.id,
        FIELD_SPECIES_ID: sp.id,
        FIELD_NUMBER: _optional_int(path, index, reader, row.get('number'), S.CSV_COL_NUMBER),
        FIELD_D_CM: d_cm,
        FIELD_H_M: format(h_m.quantize(TREE_H_QUANTUM), 'f'),
        FIELD_H_MEASURED: _optional_bool(path, index, row.get('h_measured'), S.CSV_COL_H_MEASURED),
        FIELD_HYPSO_PARAM_SET_ID: None,
        FIELD_LAT: _optional_coord(path, index, reader, row.get('lat'), S.CSV_COL_LAT),
        FIELD_LON: _optional_coord(path, index, reader, row.get('lon'), S.CSV_COL_LON),
        FIELD_ACC_M: _optional_int(path, index, reader, row.get('acc_m'), S.CSV_COL_ACC_M),
        FIELD_OPERATOR: (row.get('operator') or '').strip(),
    }
    damaged = _optional_bool(path, index, row.get('damaged'), S.CSV_COL_DAMAGED)
    return record, damaged


def _parse_date(path: Path, index: int, value: str):
    raw = (value or '').strip()
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y'):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    raise CommandError(f'{path}: row {index}: invalid {S.CSV_COL_DATA}')


def _optional_int(path: Path, index: int, reader, value, label: str) -> int | None:
    parsed, ok = reader.opt_int(value or '')
    if not ok:
        raise CommandError(f'{path}: row {index}: invalid {label}')
    if parsed is not None and parsed <= 0:
        raise CommandError(f'{path}: row {index}: invalid {label}')
    return parsed


def _optional_coord(path: Path, index: int, reader, value, label: str) -> float | None:
    parsed, ok = reader.opt_decimal(value or '')
    if not ok:
        raise CommandError(f'{path}: row {index}: invalid {label}')
    return coord_float(parsed)


def _optional_bool(path: Path, index: int, value, label: str) -> bool:
    raw = (value or '').strip()
    if not raw:
        return False
    parsed = parse_bool(raw)
    if parsed is None:
        raise CommandError(f'{path}: row {index}: invalid {label}')
    return parsed


def _session_id(path: Path, csv_text: str) -> str:
    try:
        return str(uuid.UUID(path.stem))
    except ValueError:
        digest = hashlib.sha256(csv_text.encode('utf-8')).hexdigest()
        return str(uuid.uuid5(SESSION_NAMESPACE, f'{path.name}:{digest}'))


def _parcel_map() -> dict:
    return {
        (p.region.name.lower(), p.name.lower()): p
        for p in Parcel.objects.select_related('region')
    }


def _species_map() -> dict:
    return {sp.common_name.lower(): sp for sp in Species.objects.all()}
