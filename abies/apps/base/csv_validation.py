"""Typed CSV cells with field-specific, one-message-per-row diagnostics.

CsvReader owns decoding and locale-tolerant coercion. CsvRow adds required,
range and storage checks without writing data or changing import policies.
"""
from datetime import date, datetime

from apps.base.numparse import coord_float
from apps.base.validation import decimal_field_value
from config import strings as S


class CsvRow:
    def __init__(self, reader, values, number, *, labels=None):
        self.reader = reader
        self.values = values
        self.number = number
        self.labels = labels or {}
        self.errors = []

    def raw(self, column):
        return (self.values.get(column) or '').strip()

    def problem(self, column, rule):
        label = self.labels.get(column, column)
        label = {
            S.CSV_COL_D_CM: S.CSV_LABEL_DIAMETER,
            S.CSV_COL_H_M: S.CSV_LABEL_HEIGHT,
        }.get(label, label)
        self.errors.append(S.ERR_CSV_FIELD.format(label, rule))

    def validate(self, errors):
        if self.errors:
            errors.append(S.ERR_CSV_ROW_INVALID.format(self.number, '; '.join(self.errors)))
            return False
        return True

    def text(self, column, *, required=True, default=''):
        value = self.raw(column)
        if not value and required:
            self.problem(column, S.CSV_RULE_REQUIRED)
        return value or default

    def _number(self, column, parser, rule, *, required, default, positive,
                minimum, maximum, storage=None):
        raw = self.raw(column)
        if not raw:
            if required:
                self.problem(column, S.CSV_RULE_REQUIRED)
            return default
        value = parser(raw)
        if value is None:
            self.problem(column, rule)
            return None
        if storage and decimal_field_value(value, *storage) is None:
            self.problem(column, S.CSV_RULE_STORAGE)
            return None
        if positive and value <= 0:
            self.problem(column, rule)
            return None
        if ((minimum is not None and value < minimum)
                or (maximum is not None and value > maximum)):
            if minimum is not None and maximum is not None:
                rule = S.CSV_RULE_RANGE.format(minimum, maximum)
            elif minimum is not None:
                rule = S.CSV_RULE_MINIMUM.format(minimum)
            else:
                rule = S.CSV_RULE_MAXIMUM.format(maximum)
            self.problem(column, rule)
            return None
        return value

    def integer(self, column, *, required=True, default=None, positive=False,
                minimum=None, maximum=None):
        return self._number(column, self.reader.integer,
            S.CSV_RULE_POSITIVE_INTEGER if positive else S.CSV_RULE_INTEGER,
            required=required, default=default, positive=positive,
            minimum=minimum, maximum=maximum)

    def decimal(self, column, *, required=True, default=None, positive=False,
                minimum=None, maximum=None, storage=None):
        return self._number(column, self.reader.decimal,
            S.CSV_RULE_POSITIVE_NUMBER if positive else S.CSV_RULE_NUMBER,
            required=required, default=default, positive=positive,
            minimum=minimum, maximum=maximum, storage=storage)

    def boolean(self, column, *, required=False, default=False):
        raw = self.raw(column)
        if not raw:
            if required:
                self.problem(column, S.CSV_RULE_REQUIRED)
            return default
        value, ok = self.reader.opt_bool(raw)
        if not ok:
            self.problem(column, S.CSV_RULE_BOOLEAN)
        return value

    def date(self, column, *, required=True, default=None, formats=()):
        raw = self.raw(column)
        if not raw:
            if required and default is None:
                self.problem(column, S.CSV_RULE_REQUIRED)
            return default
        try:
            return date.fromisoformat(raw)
        except ValueError:
            for pattern in formats:
                try:
                    return datetime.strptime(raw, pattern).date()
                except ValueError:
                    pass
        self.problem(column, S.CSV_RULE_DATE)
        return None

    def coordinates(self, latitude, longitude, *, required=False):
        previous_errors = len(self.errors)
        lat = self.decimal(latitude, required=required, minimum=-90, maximum=90)
        lon = self.decimal(longitude, required=required, minimum=-180, maximum=180)
        if len(self.errors) == previous_errors and (lat is None) != (lon is None):
            self.problem(latitude if lat is None else longitude, S.CSV_RULE_COORDINATE_PAIR)
        return coord_float(lat), coord_float(lon)
