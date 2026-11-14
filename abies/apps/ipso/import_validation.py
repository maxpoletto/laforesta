"""Source-row identity and diagnostics shared by staged tree importers."""
from math import isfinite

from apps.base.numparse import int_or_none, to_decimal
from apps.base.validation import decimal_field_value
from config import strings as S
from config.constants import FIELD_CLIENT_RECORD_ID


def record_id(record, index):
    """Match the original Ipso ID shown in the preview, even after deletions."""
    if isinstance(record, dict):
        value = record.get(FIELD_CLIENT_RECORD_ID)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(index)


class RowErrors:
    def __init__(self, issues=None):
        self.messages = []
        self.issues = issues if issues is not None else {}

    def add(self, index, message):
        self.messages.append(message)
        self.issues.setdefault(index, []).append(message)


def position_problems(record):
    """Check the same position/accuracy invariants as the domain schema."""
    lat, lon = record.get('lat'), record.get('lon')
    problems = []
    if (lat is None) != (lon is None) or any(
        value is not None and (type(value) not in {int, float} or not isfinite(value) or abs(value) > limit)
        for value, limit in ((lat, 90), (lon, 180))
    ):
        problems.append(S.IPSO_MEASUREMENT_POSITION_INVALID)
    accuracy = record.get('acc_m')
    if accuracy is not None and ((parsed := int_or_none(accuracy)) is None or parsed < 0):
        problems.append(S.IPSO_MEASUREMENT_ACCURACY_INVALID)
    return problems


def sample_field_problems(record):
    problems = []
    for name, message in (('shoot', S.IPSO_MEASUREMENT_SHOOT_INVALID),
                          ('l10_mm', S.IPSO_MEASUREMENT_L10_INVALID)):
        value = record.get(name)
        if name == 'l10_mm' and isinstance(value, str) and not value.strip():
            continue
        if value is not None and ((parsed := int_or_none(value)) is None or parsed < 0):
            problems.append(message)
    raw = record.get('pressler_coeff')
    if raw is not None:
        value = to_decimal(raw, '.')
        if value is not None:
            value = decimal_field_value(value, 'base.TreeSample', 'pressler_coeff')
        if value is None or value <= 0:
            problems.append(S.IPSO_MEASUREMENT_PRESSLER_INVALID)
    return problems
