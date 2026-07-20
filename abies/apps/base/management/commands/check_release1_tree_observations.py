# Preflight/postflight checks for Release 1 tree-observation migration.

from __future__ import annotations

from datetime import date as date_type

from django.core.management.base import BaseCommand, CommandError
from django.db import connection


PRESERVED_HISTORY_SURVEY_NAME = 'Alberi da preservare - storico'
PRESERVED_LEGACY_UNKNOWN_DATE = date_type(1970, 1, 1)
MAX_ROWS_PER_FAILURE = 20


class Command(BaseCommand):
    help = (
        'Check Release 1 tree-observation data invariants before or after the '
        'preserved-tree migration.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--phase', choices=('pre', 'post'), required=True,
            help=(
                'pre: run before applying Release 1 migrations; '
                'post: run after applying Release 1 migrations.'
            ),
        )

    def handle(self, *args, phase: str, **opts):
        checker = _Release1TreeObservationChecker()
        if phase == 'pre':
            failures = checker.check_preflight()
        else:
            failures = checker.check_postflight()
        if failures:
            raise CommandError(_format_failures(failures))
        label = 'preflight' if phase == 'pre' else 'postflight'
        self.stdout.write(self.style.SUCCESS(
            f'Release 1 tree-observation {label} OK.'
        ))


class _Release1TreeObservationChecker:
    def __init__(self):
        with connection.cursor() as cursor:
            self.tables = set(connection.introspection.table_names(cursor))
        self._columns_cache: dict[str, set[str]] = {}

    def check_preflight(self) -> list[tuple[str, list[dict[str, object]]]]:
        failures = []
        failures.extend(self._require_tables([
            'base_treepreserved', 'base_sample', 'base_survey',
        ]))
        if failures:
            return failures

        self._add_if_rows(
            failures,
            'Legacy TreePreserved rows missing diameter required by migration',
            '''
            SELECT id, date, d_cm, h_m
            FROM base_treepreserved
            WHERE d_cm IS NULL
            ORDER BY id
            ''',
        )
        self._add_if_rows(
            failures,
            'Existing free Sample rows need ad-hoc handling before Release 1',
            '''
            SELECT s.id, s.survey_id, sv.name AS survey, s.date
            FROM base_sample s
            JOIN base_survey sv ON sv.id = s.survey_id
            WHERE s.sample_area_id IS NULL
            ORDER BY s.id
            ''',
        )
        self._add_if_rows(
            failures,
            'Reserved preserved-tree history survey name is already structured',
            '''
            SELECT id, name, sample_grid_id, active
            FROM base_survey
            WHERE name = %s AND sample_grid_id IS NOT NULL
            ORDER BY id
            ''',
            [PRESERVED_HISTORY_SURVEY_NAME],
        )

        if self._has_columns('base_treesample', {'parcel_id'}):
            self._add_structured_parcel_mismatches(failures)
        if self._has_columns('base_treesample', {'parcel_id', 'preserved_number'}):
            self._add_preserved_identity_failures(failures)
        return failures

    def check_postflight(self) -> list[tuple[str, list[dict[str, object]]]]:
        failures = []
        failures.extend(self._require_tables([
            'base_treepreserved', 'base_treesample', 'base_sample',
            'base_survey', 'base_samplearea', 'base_tree',
        ]))
        required_columns = {
            'base_treesample': {
                'sample_id', 'tree_id', 'parcel_id', 'number',
                'preserved_number', 'shoot', 'd_cm', 'h_m', 'h_measured',
                'lat', 'lon', 'acc_m', 'operator', 'note',
            },
            'base_sample': {'id', 'survey_id', 'sample_area_id', 'date'},
            'base_survey': {'id', 'name', 'sample_grid_id', 'active'},
            'base_tree': {'id', 'preserved'},
        }
        failures.extend(self._require_columns(required_columns))
        if failures:
            return failures

        self._add_if_rows(
            failures,
            'Legacy TreePreserved rows without migrated TreeSample rows',
            '''
            SELECT tp.id, tp.parcel_id, tp.number, tp.tree_id, tp.date
            FROM base_treepreserved tp
            LEFT JOIN base_survey sv
              ON sv.name = %s
            LEFT JOIN base_sample s
              ON s.survey_id = sv.id
             AND s.sample_area_id IS NULL
             AND s.date = COALESCE(tp.date, %s)
            LEFT JOIN base_treesample ts
              ON ts.sample_id = s.id
             AND ts.tree_id = tp.tree_id
             AND ts.parcel_id = tp.parcel_id
             AND ts.preserved_number = tp.number
             AND ts.number = tp.number
             AND ts.shoot = 0
            WHERE ts.id IS NULL
            ORDER BY tp.id
            ''',
            [PRESERVED_HISTORY_SURVEY_NAME, PRESERVED_LEGACY_UNKNOWN_DATE],
        )
        self._add_if_rows(
            failures,
            'Migrated TreeSample rows differ from legacy TreePreserved rows',
            '''
            SELECT
                tp.id AS tree_preserved_id,
                ts.id AS tree_sample_id,
                tp.d_cm AS legacy_d_cm,
                ts.d_cm AS migrated_d_cm,
                tp.h_m AS legacy_h_m,
                ts.h_m AS migrated_h_m,
                tp.h_measured AS legacy_h_measured,
                ts.h_measured AS migrated_h_measured,
                tp.lat AS legacy_lat,
                ts.lat AS migrated_lat,
                tp.lon AS legacy_lon,
                ts.lon AS migrated_lon,
                tp.acc_m AS legacy_acc_m,
                ts.acc_m AS migrated_acc_m,
                tp.operator AS legacy_operator,
                ts.operator AS migrated_operator,
                tp.note AS legacy_note,
                ts.note AS migrated_note
            FROM base_treepreserved tp
            JOIN base_survey sv
              ON sv.name = %s
            JOIN base_sample s
              ON s.survey_id = sv.id
             AND s.sample_area_id IS NULL
             AND s.date = COALESCE(tp.date, %s)
            JOIN base_treesample ts
              ON ts.sample_id = s.id
             AND ts.tree_id = tp.tree_id
             AND ts.parcel_id = tp.parcel_id
             AND ts.preserved_number = tp.number
             AND ts.number = tp.number
             AND ts.shoot = 0
            WHERE tp.d_cm <> ts.d_cm
               OR (
                    tp.h_m <> ts.h_m
                    OR (tp.h_m IS NULL AND ts.h_m IS NOT NULL)
                    OR (tp.h_m IS NOT NULL AND ts.h_m IS NULL)
               )
               OR (
                    CASE WHEN tp.h_m IS NULL THEN 0 ELSE tp.h_measured END
               ) <> ts.h_measured
               OR tp.lat <> ts.lat
               OR tp.lon <> ts.lon
               OR (
                    tp.acc_m <> ts.acc_m
                    OR (tp.acc_m IS NULL AND ts.acc_m IS NOT NULL)
                    OR (tp.acc_m IS NOT NULL AND ts.acc_m IS NULL)
               )
               OR COALESCE(tp.operator, '') <> COALESCE(ts.operator, '')
               OR COALESCE(tp.note, '') <> COALESCE(ts.note, '')
            ORDER BY tp.id
            ''',
            [PRESERVED_HISTORY_SURVEY_NAME, PRESERVED_LEGACY_UNKNOWN_DATE],
        )
        self._add_structured_parcel_mismatches(failures)
        self._add_preserved_identity_failures(failures)
        self._add_if_rows(
            failures,
            'Preserved TreeSample rows point to Tree rows not marked preserved',
            '''
            SELECT ts.id, ts.tree_id, ts.parcel_id, ts.preserved_number
            FROM base_treesample ts
            JOIN base_tree t ON t.id = ts.tree_id
            WHERE ts.preserved_number IS NOT NULL AND NOT t.preserved
            ORDER BY ts.id
            ''',
        )
        return failures

    def _add_structured_parcel_mismatches(self, failures):
        self._add_if_rows(
            failures,
            'Structured TreeSample rows have parcel different from SampleArea parcel',
            '''
            SELECT
                ts.id,
                ts.sample_id,
                ts.parcel_id AS tree_sample_parcel_id,
                sa.parcel_id AS sample_area_parcel_id
            FROM base_treesample ts
            JOIN base_sample s ON s.id = ts.sample_id
            JOIN base_samplearea sa ON sa.id = s.sample_area_id
            WHERE s.sample_area_id IS NOT NULL
              AND ts.parcel_id <> sa.parcel_id
            ORDER BY ts.id
            ''',
        )

    def _add_preserved_identity_failures(self, failures):
        self._add_if_rows(
            failures,
            'Preserved TreeSample identity maps to more than one Tree',
            '''
            SELECT parcel_id, preserved_number, COUNT(DISTINCT tree_id) AS tree_count
            FROM base_treesample
            WHERE preserved_number IS NOT NULL
            GROUP BY parcel_id, preserved_number
            HAVING COUNT(DISTINCT tree_id) > 1
            ORDER BY parcel_id, preserved_number
            ''',
        )
        self._add_if_rows(
            failures,
            'Duplicate preserved TreeSample rows within one sample and parcel',
            '''
            SELECT sample_id, parcel_id, preserved_number, COUNT(*) AS row_count
            FROM base_treesample
            WHERE preserved_number IS NOT NULL
            GROUP BY sample_id, parcel_id, preserved_number
            HAVING COUNT(*) > 1
            ORDER BY sample_id, parcel_id, preserved_number
            ''',
        )
        self._add_if_rows(
            failures,
            'Preserved TreeSample rows use non-positive preserved numbers',
            '''
            SELECT id, sample_id, parcel_id, preserved_number
            FROM base_treesample
            WHERE preserved_number IS NOT NULL AND preserved_number <= 0
            ORDER BY id
            ''',
        )

    def _require_tables(self, tables: list[str]):
        rows = [{'table': table} for table in tables if table not in self.tables]
        if rows:
            return [('Required database tables are missing', rows)]
        return []

    def _require_columns(self, required: dict[str, set[str]]):
        rows = []
        for table, columns in required.items():
            if table not in self.tables:
                rows.append({'table': table, 'column': '<table missing>'})
                continue
            existing = self._columns(table)
            for column in sorted(columns - existing):
                rows.append({'table': table, 'column': column})
        if rows:
            return [('Required Release 1 database columns are missing', rows)]
        return []

    def _has_columns(self, table: str, columns: set[str]) -> bool:
        return table in self.tables and columns <= self._columns(table)

    def _columns(self, table: str) -> set[str]:
        if table not in self._columns_cache:
            with connection.cursor() as cursor:
                description = connection.introspection.get_table_description(
                    cursor, table,
                )
            self._columns_cache[table] = {field.name for field in description}
        return self._columns_cache[table]

    def _add_if_rows(self, failures, label, sql, params=None):
        rows = self._select(sql, params or [])
        if rows:
            failures.append((label, rows))

    def _select(self, sql: str, params: list[object]):
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _format_failures(failures: list[tuple[str, list[dict[str, object]]]]) -> str:
    lines = ['Release 1 tree-observation check failed:']
    for label, rows in failures:
        lines.append(f'- {label} ({len(rows)} row(s))')
        for row in rows[:MAX_ROWS_PER_FAILURE]:
            lines.append(f'  - {_format_row(row)}')
        if len(rows) > MAX_ROWS_PER_FAILURE:
            remaining = len(rows) - MAX_ROWS_PER_FAILURE
            lines.append(f'  - ... {remaining} more row(s) omitted')
    return '\n'.join(lines)


def _format_row(row: dict[str, object]) -> str:
    return ', '.join(f'{key}={value!r}' for key, value in row.items())
