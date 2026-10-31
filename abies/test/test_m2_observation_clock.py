from datetime import timedelta

import pytest
from django.test import Client
from django.utils import timezone

from apps.base.models import Observation, ObservationCategory

pytestmark = pytest.mark.django_db


def test_observation_edit_owns_one_clock_and_token_for_metadata_and_categories(regions, writer_user, monkeypatch):
    observation = Observation.objects.create(date='2026-10-01', region=regions[0],
        text='Before', lat=38, lon=16, operator='Operator', source='bosco')
    category = ObservationCategory.objects.create(name='Clock category')
    client = Client()
    client.force_login(writer_user)
    before = (observation.created_at, observation.modified_at, observation.history.count())
    later = before[1] + timedelta(minutes=1)
    monkeypatch.setattr(timezone, 'now', lambda: later)
    response = client.post('/api/bosco/observations/save/', dict(
        row_id=observation.pk, version=observation.version, date='2026-10-02',
        region_id=regions[0].pk, text='After', lat='38', lon='16',
        category_ids=[str(category.pk)]))
    assert response.status_code == 200, response.content
    observation.refresh_from_db()
    assert observation.modified_at == later and observation.created_at == before[0]
    assert observation.version == 2 and observation.history.count() == before[2] + 1
    assert list(observation.categories.all()) == [category]
