"""Read-only review, export receipts and recovery CSVs for staged tree imports."""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
import hashlib
import json

from django.core import signing
from django.http import HttpResponse, JsonResponse

from apps.base import csv_io
from apps.base.models import Parcel, Sample, SampleArea, Species, TreeMark
from apps.base.numparse import int_or_none
from apps.base.validation import InputValidationError, decimal_field_value
from apps.campionamenti.csv_trees import FREE_TREE_CSV_REQUIRED, TREE_CSV_REQUIRED
from apps.ipso.import_validation import record_id
from apps.piano_di_taglio.mark_import import mark_volume_and_mass
from config import strings as S
from config.constants import (
    FIELD_H_MEASURED, FIELD_LAT, FIELD_LON, FIELD_PARCEL, PRESSLER_DEFAULT, RECORDS, SESSION,
)

# Public purpose namespaces: Django signs both token kinds with SECRET_KEY.
REVIEW_SALT = 'ipso.tree-import.review.v1'
EXPORT_SALT = 'ipso.tree-import.export.v1'
# Workflow timeout for both token kinds; staged/recovery data does not expire.
REVIEW_MAX_AGE = 30 * 60
EXPORT_HEADER = 'X-Import-Export-Token'


@dataclass
class TreeImportReview:
    upload: object
    payload: dict
    target: object
    rows: list
    issues: dict[int, list[str]]
    warnings: list[str] = field(default_factory=list)

    @property
    def records(self):
        return self.payload[RECORDS]

    @property
    def accepted_indices(self):
        return [i for i in range(1, len(self.records) + 1) if i not in self.issues]

    def reject(self, index, message):
        self.issues.setdefault(index, []).append(message)

    def validate_groups_and_storage(self):
        """Reject ambiguous groups as a whole, and catch late decimal overflows."""
        source_rows = list(zip(self.accepted_indices, self.rows, strict=True))
        numbered = defaultdict(list)
        dates = defaultdict(set)
        free_dates = set()
        for index, record in enumerate(self.records, 1):
            number = int_or_none(record.get('number'))
            if number is not None and number > 0:
                if self.upload.mode == 'samples':
                    key = (record.get('sample_area_id'), number, record.get('shoot') or 0)
                elif self.upload.mode == 'free_survey':
                    key = (record.get('parcel_id'), number) if record.get('preserved') else number
                else:
                    key = number
                numbered[key].append(index)
            if self.upload.mode in {'samples', 'free_survey'}:
                try:
                    parsed_date = date.fromisoformat(str(record.get('date')))
                    dates[record.get('sample_area_id')].add(parsed_date)
                    free_dates.add(parsed_date)
                except (ValueError, TypeError):
                    pass
        for indices in numbered.values():
            if len(indices) > 1:
                for index in indices:
                    record = self.records[index - 1]
                    message = (S.IPSO_ERR_IMPORT_ROW.format(record_id(record, index),
                               S.ERR_MARK_NUMBER_DUPLICATE.format(record.get('number')))
                               if self.upload.mode == 'martellate' else
                               S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(record_id(record, index)))
                    self.reject(index, message)
        if self.upload.mode == 'samples':
            existing_areas = set(Sample.objects.filter(survey=self.target).values_list('sample_area_id', flat=True))
            for index, record in enumerate(self.records, 1):
                area = record.get('sample_area_id')
                if area not in existing_areas and len(dates[area]) > 1:
                    self.reject(index, S.IPSO_ERR_IMPORT_RECORD_DATE_AMBIGUOUS.format(record_id(record, index)))
        if self.upload.mode == 'free_survey' and len(free_dates) > 1:
            for index, record in enumerate(self.records, 1):
                self.reject(index, S.IPSO_ERR_IMPORT_RECORD_FREE_DATE_AMBIGUOUS.format(record_id(record, index)))
        existing_marks = {}
        if self.upload.mode == 'martellate':
            existing_marks = dict(TreeMark.objects.filter(harvest_plan_item=self.target)
                                  .exclude(number=None).values_list('number', 'import_fingerprint'))
        for index, row in source_rows:
            label = record_id(self.records[index - 1], index)
            if self.upload.mode == 'martellate':
                if row.number in existing_marks and existing_marks[row.number] != row.fingerprint:
                    self.reject(index, S.IPSO_ERR_IMPORT_ROW.format(label, S.ERR_MARK_NUMBER_DUPLICATE.format(row.number)))
                try:
                    volume, mass = mark_volume_and_mass(row.d_cm, row.h_m, row.species)
                except InputValidationError:
                    self.reject(index, S.IPSO_ERR_IMPORT_RECORD_DERIVED_OVERFLOW.format(label))
                    continue
                model = 'base.TreeMark'
            else:
                volume, mass = row['volume_m3'], row['mass_q']
                model = 'base.TreeSample'
            if any(value is not None and decimal_field_value(value, model, name) is None
                   for name, value in (('volume_m3', volume), ('mass_q', mass))):
                self.reject(index, S.IPSO_ERR_IMPORT_RECORD_DERIVED_OVERFLOW.format(label))
        self.rows = [row for index, row in source_rows if index not in self.issues]

    def snapshot(self, user):
        data = {
            'upload': self.upload.pk, 'checksum': self.upload.checksum, 'mode': self.upload.mode,
            'target': self.target.pk, 'target_version': self.target.version, 'user': user.pk,
            'accepted': self.accepted_indices, 'issues': self.issues, 'warnings': self.warnings,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return {'upload': self.upload.pk, 'target': self.target.pk, 'user': user.pk,
                'digest': hashlib.sha256(canonical.encode()).hexdigest()}

    def token(self, user):
        return signing.dumps(self.snapshot(user), salt=REVIEW_SALT, compress=True)

    def token_matches(self, token, user, *, salt=REVIEW_SALT):
        if not isinstance(token, str):
            return False
        try:
            return signing.loads(token, salt=salt, max_age=REVIEW_MAX_AGE) == self.snapshot(user)
        except (signing.BadSignature, TypeError, ValueError):
            return False

    def response(self, user, message=''):
        return JsonResponse({
            'status': 'import_review', 'review_token': self.token(user),
            'valid_count': len(self.rows), 'excluded_count': len(self.issues),
            'errors': ['; '.join(dict.fromkeys(messages)) for _, messages in sorted(self.issues.items())],
            'warnings': self.warnings, 'message': message,
        }, status=409)

    def outcome(self, imported, skipped=0):
        return {
            'imported': imported, 'skipped_duplicates': skipped, 'excluded_count': len(self.issues),
            'accepted_record_ids': [record_id(self.records[i - 1], i) for i in self.accepted_indices],
            'excluded': [{'index': index, 'record_id': record_id(self.records[index - 1], index),
                          'errors': list(dict.fromkeys(messages))}
                         for index, messages in sorted(self.issues.items())],
        }

    def export(self, user):
        response = recovery_csv(self.upload, self.payload, self.outcome(0)['excluded'])
        response[EXPORT_HEADER] = signing.dumps(self.snapshot(user), salt=EXPORT_SALT, compress=True)
        return response


def read_review_token(token, user, upload_id):
    if not isinstance(token, str):
        return None
    try:
        data = signing.loads(token, salt=REVIEW_SALT, max_age=REVIEW_MAX_AGE)
        if data['user'] == user.pk and data['upload'] == upload_id:
            return data
    except (signing.BadSignature, TypeError, ValueError, KeyError):
        pass
    return None


def recovery_csv(upload, payload, excluded):
    """Use normal domain import headers, with provenance and lossless raw data."""
    records = payload[RECORDS]
    parcels = {p.pk: p for p in Parcel.objects.select_related('region')}
    species = dict(Species.objects.values_list('pk', 'common_name'))
    areas = dict(SampleArea.objects.values_list('pk', 'number'))
    delimiter, decimal_sep = csv_io.export_format()
    buffer, writer = csv_io.csv_buffer(delimiter)
    if upload.mode == 'martellate':
        columns = [S.CSV_COL_DATA, S.CSV_COL_REGION, S.CSV_COL_PARCEL, S.CSV_COL_DAMAGED,
                   S.CSV_COL_NUMBER, S.CSV_COL_SPECIES, S.CSV_COL_D_CM, S.CSV_COL_H_M]
    else:
        columns = list(TREE_CSV_REQUIRED if upload.mode == 'samples' else FREE_TREE_CSV_REQUIRED)
        columns.append(S.CSV_COL_DATA)
    columns += [S.CSV_COL_PRESERVED, S.CSV_COL_H_MEASURED, S.CSV_COL_LAT, S.CSV_COL_LON,
                S.CSV_COL_ACC_M, S.CSV_COL_OPERATOR, S.CSV_COL_NOTE,
                S.CSV_COL_IPSO_ROW, S.CSV_COL_IMPORT_ERRORS, S.CSV_COL_IPSO_SOURCE]
    writer.writerow(columns)
    for rejected in excluded:
        raw = records[rejected['index'] - 1]
        parcel = parcels.get(raw.get('parcel_id'))
        values = {
            S.CSV_COL_DATA: raw.get('date'),
            S.CSV_COL_REGION: parcel.region.name if parcel else raw.get('region_id'),
            S.CSV_COL_PARCEL: parcel.name if parcel else raw.get('parcel_id'),
            S.CSV_COL_SAMPLE_AREA: areas.get(raw.get('sample_area_id'), raw.get('sample_area_id')),
            S.CSV_COL_TREE: raw.get('number'), S.CSV_COL_NUMBER: raw.get('number'),
            S.CSV_COL_SPECIES: species.get(raw.get('species_id'), raw.get('species_id')),
            S.CSV_COL_D_CM: raw.get('d_cm'),
            S.CSV_COL_H_M: csv_io.format_decimal(raw.get('h_m'), decimal_sep),
            S.CSV_COL_COPPICE_SHOOT: raw.get('shoot') or 0,
            S.CSV_COL_COPPICE_STD: int(bool(raw.get('standard'))),
            S.CSV_COL_HIGHFOREST: int(not bool(raw.get('coppice'))),
            S.CSV_COL_L10_MM: raw.get('l10_mm') or 0,
            S.CSV_COL_PRESSLER: csv_io.format_decimal(raw.get('pressler_coeff') or PRESSLER_DEFAULT, decimal_sep),
            S.CSV_COL_PRESERVED: int(bool(raw.get('preserved'))),
            S.CSV_COL_H_MEASURED: int(bool(raw.get('h_measured'))),
            S.CSV_COL_LAT: csv_io.format_decimal(raw.get('lat'), decimal_sep),
            S.CSV_COL_LON: csv_io.format_decimal(raw.get('lon'), decimal_sep),
            S.CSV_COL_ACC_M: raw.get('acc_m'),
            S.CSV_COL_OPERATOR: raw.get('operator') or payload.get(SESSION, {}).get('operator', ''),
            S.CSV_COL_NOTE: raw.get('note', ''),
            S.CSV_COL_DAMAGED: int(bool(payload.get(SESSION, {}).get('damaged'))),
            S.CSV_COL_IPSO_ROW: rejected['record_id'],
            S.CSV_COL_IMPORT_ERRORS: '; '.join(rejected['errors']),
            S.CSV_COL_IPSO_SOURCE: json.dumps(raw, ensure_ascii=False, sort_keys=True),
        }
        writer.writerow([values.get(column, '') for column in columns])
    response = HttpResponse(buffer.getvalue(), content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="{S.IPSO_REJECTED_CSV.format(upload.pk)}"'
    response['Cache-Control'] = 'no-store'
    return response
