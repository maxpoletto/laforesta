"""Reference-table CSV import cores: regions, eclasses, crews, species, products.

These are root tables (no foreign keys), seeded into an empty instance by
``bootstrap`` and re-importable standalone for dev.  Each table is described
declaratively by a ``RefTable`` (its model, unique key column, and typed
columns); the generic ``resolve_columns`` -> ``validate_rows`` -> ``apply``
functions drive every table identically, mirroring the three-phase contract of
the domain cores (see ``apps/campionamenti/csv_grid``).

``validate_rows`` is pure (no DB writes); ``apply`` upserts by the key column
and is idempotent, so a standalone re-import is safe.
"""

from dataclasses import dataclass

from django.db import transaction

from apps.base.digests import mark_all_stale
from apps.base.models import Crew, Eclass, Product, Region, Species, Tractor
from config import strings as S
from config.constants import (
    FIELD_ACTIVE, FIELD_COMMON_NAME, FIELD_COPPICE, FIELD_DENSITY,
    FIELD_LATIN_NAME, FIELD_MANUFACTURER, FIELD_MINOR, FIELD_MIN_HARVEST_VOLUME,
    FIELD_MODEL, FIELD_NAME, FIELD_NOTES, FIELD_PRESSLER_DEFAULT, FIELD_SORT_ORDER,
    FIELD_YEAR,
)

# How a raw CSV cell maps to a Python value.
KIND_STR = 'str'
KIND_INT = 'int'
KIND_DECIMAL = 'decimal'
KIND_BOOL = 'bool'


@dataclass(frozen=True)
class RefColumn:
    """One column of a reference table's canonical CSV."""
    header: str             # canonical CSV header (a localized S.CSV_COL_* const)
    field: str              # model field name (a FIELD_* const)
    kind: str               # one of KIND_*
    required: bool = True
    default: object = None  # value for an optional column that is absent or
                            # blank; None means "omit so the model default applies"


@dataclass(frozen=True)
class RefTable:
    """A root reference table loadable from one canonical CSV file.

    ``name`` doubles as the data-dir filename stem (``<name>.csv``).
    ``columns[0]`` is the unique key used for upsert and duplicate detection.
    """
    name: str
    model: type
    columns: tuple

    @property
    def key(self) -> RefColumn:
        return self.columns[0]


REGIONS = RefTable('regions', Region, (
    RefColumn(S.CSV_COL_REGION, FIELD_NAME, KIND_STR),
))

ECLASSES = RefTable('eclasses', Eclass, (
    RefColumn(S.CSV_COL_CLASS, FIELD_NAME, KIND_STR),
    RefColumn(S.CSV_COL_COPPICE, FIELD_COPPICE, KIND_BOOL),
    RefColumn(S.CSV_COL_MIN_VOLUME, FIELD_MIN_HARVEST_VOLUME, KIND_INT,
              required=False, default=0),
))

CREWS = RefTable('crews', Crew, (
    RefColumn(S.CSV_COL_CREW, FIELD_NAME, KIND_STR),
    RefColumn(S.CSV_COL_NOTE, FIELD_NOTES, KIND_STR, required=False, default=''),
    RefColumn(S.CSV_COL_ACTIVE, FIELD_ACTIVE, KIND_BOOL, required=False,
              default=True),
))

SPECIES = RefTable('species', Species, (
    RefColumn(S.CSV_COL_SPECIES, FIELD_COMMON_NAME, KIND_STR),
    RefColumn(S.CSV_COL_LATIN, FIELD_LATIN_NAME, KIND_STR, required=False,
              default=''),
    RefColumn(S.CSV_COL_DENSITY, FIELD_DENSITY, KIND_DECIMAL, required=False),
    RefColumn(S.CSV_COL_PRESSLER, FIELD_PRESSLER_DEFAULT, KIND_DECIMAL, required=False),
    RefColumn(S.CSV_COL_MINOR, FIELD_MINOR, KIND_BOOL, required=False,
              default=False),
    RefColumn(S.CSV_COL_ACTIVE, FIELD_ACTIVE, KIND_BOOL, required=False,
              default=True),
    RefColumn(S.CSV_COL_SORT_ORDER, FIELD_SORT_ORDER, KIND_INT, required=False,
              default=0),
))

PRODUCTS = RefTable('products', Product, (
    RefColumn(S.CSV_COL_PRODUCT, FIELD_NAME, KIND_STR),
))

TRACTORS = RefTable('tractors', Tractor, (
    RefColumn(S.CSV_COL_TRACTOR_NAME, FIELD_NAME, KIND_STR),
    RefColumn(S.CSV_COL_MANUFACTURER, FIELD_MANUFACTURER, KIND_STR,
              required=False, default=''),
    RefColumn(S.CSV_COL_MODEL, FIELD_MODEL, KIND_STR, required=False, default=''),
    RefColumn(S.CSV_COL_YEAR, FIELD_YEAR, KIND_INT, required=False),
))

# Iterated by the bootstrap orchestrator (plan 06) in declaration order.
ALL_TABLES = (REGIONS, ECLASSES, CREWS, TRACTORS, SPECIES, PRODUCTS)


def resolve_columns(table: RefTable, fieldnames):
    """Map model field -> actual header for every present column.

    Returns ``(found, missing)``; ``missing`` lists the canonical headers of
    absent *required* columns.  No aliasing — these are new canonical files.
    """
    found, missing = {}, []
    available = set(fieldnames)
    for col in table.columns:
        if col.header in available:
            found[col.field] = col.header
        elif col.required:
            missing.append(col.header)
    return found, missing


def _coerce(reader, raw, kind):
    """Coerce one raw cell.  Returns ``(value, ok)``: ``value`` is ``None`` for a
    blank cell; ``ok`` is ``False`` only when a non-blank cell fails to parse.
    Numeric/boolean handling delegates to the shared ``CsvReader`` helpers so the
    blank-vs-invalid logic lives in exactly one place."""
    if kind == KIND_STR:
        return ((raw or '').strip() or None), True
    if kind == KIND_INT:
        return reader.opt_int(raw)
    if kind == KIND_DECIMAL:
        return reader.opt_decimal(raw)
    return reader.opt_bool(raw)   # KIND_BOOL


def validate_rows(table: RefTable, reader, cols):
    """Validate parsed rows for ``table``.  Pure: no DB writes.

    Returns ``(parsed, errors)``.  Each parsed row is a dict of model field ->
    value ready for ``apply``; absent optional columns and blank optional cells
    fall back to the column default (or are omitted so the model default
    applies).  A blank *required* value, an unparseable numeric, or a duplicate
    key is an error naming the 1-based data row (header is row 1) and column.
    """
    errors, parsed, seen = [], [], set()
    for i, row in enumerate(reader, 2):
        values, ok = {}, True
        for col in table.columns:
            header = cols.get(col.field)
            if header is None:           # optional column absent -> model default
                continue
            value, parsed_ok = _coerce(reader, row.get(header), col.kind)
            if not parsed_ok:
                errors.append(S.ERR_CSV_VALUE_PARSE.format(
                    i, col.header, (row.get(header) or '').strip()))
                ok = False
                break
            if value is None:            # blank cell
                if col.required:
                    errors.append(
                        S.ERR_CSV_VALUE_REQUIRED.format(i, col.header))
                    ok = False
                    break
                if col.default is not None:
                    values[col.field] = col.default
            else:
                values[col.field] = value
        if not ok:
            continue
        key_val = values[table.key.field]
        if key_val in seen:
            errors.append(S.ERR_CSV_DUPLICATE_KEY.format(
                i, table.key.header, key_val))
            continue
        seen.add(key_val)
        parsed.append(values)
    return parsed, errors


def apply(table: RefTable, parsed):
    """Upsert validated rows by the key column.  Idempotent.

    Returns ``(n_created, n_updated)``.  Wrapped in a transaction so a mid-batch
    failure leaves no partial rows; marks every digest stale.
    """
    created = updated = 0
    key_field = table.key.field
    with transaction.atomic():
        for values in parsed:
            defaults = {f: v for f, v in values.items() if f != key_field}
            obj, was_created = table.model.objects.get_or_create(
                defaults=defaults, **{key_field: values[key_field]})
            if was_created:
                created += 1
                continue
            changed = [f for f, v in defaults.items() if getattr(obj, f) != v]
            if changed:
                for f in changed:
                    setattr(obj, f, defaults[f])
                fields = list(changed)
                if hasattr(obj, 'version'):
                    obj.version += 1
                    fields += ['version', 'modified_at']
                obj.save(update_fields=fields)
                updated += 1
        mark_all_stale()
    return created, updated
