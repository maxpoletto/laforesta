from datetime import timedelta

import pytest
from django.utils import timezone

from apps.base import hypsometry
from apps.base.models import HypsoParam

pytestmark = pytest.mark.django_db


def test_archival_advances_set_clock_without_rewriting_equations(regions, species, monkeypatch):
    snapshot = hypsometry.merge_active_set([
        hypsometry.ParamRow(regions[0], species[0], 7, -4, .6, 20),
    ], source='imported')
    original_rows = list(HypsoParam.objects.filter(param_set=snapshot).values())
    before = (snapshot.created_at, snapshot.modified_at, snapshot.history.count())
    later = before[1] + timedelta(minutes=1)
    monkeypatch.setattr(timezone, 'now', lambda: later)
    assert hypsometry.clear_active_set()
    snapshot.refresh_from_db()
    assert snapshot.modified_at == snapshot.superseded_at == later
    assert snapshot.created_at == before[0] and snapshot.version == 2
    assert snapshot.history.count() == before[2] + 1
    assert list(HypsoParam.objects.filter(param_set=snapshot).values()) == original_rows
    assert not hypsometry.clear_active_set()
    assert snapshot.history.count() == before[2] + 1
