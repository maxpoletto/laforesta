"""Harvest deletion history retains the species masses lost with child rows."""

import gzip
import json
from decimal import Decimal

import pytest
from django.db import transaction
from django.test import Client
from django.utils import translation

from apps.base.digests import generate_audit
from apps.prelievi.models import Harvest, HarvestSpecies
from config import strings as S
from config.constants import ROW_ID, ROWS, VERSION


@pytest.fixture
def harvest_factory(parcels, crews, products, species):
    def create(mass='200', percentages=(50, 25, 25, 0)):
        harvest = Harvest.objects.create(
            date='2026-11-15', parcel=parcels[0], crew=crews[0],
            product=products[0], mass_q=Decimal(mass),
        )
        HarvestSpecies.objects.bulk_create([
            HarvestSpecies(harvest=harvest, species=sp, percent=percent)
            for sp, percent in zip(species, percentages)
        ])
        return harvest
    return create


def deletion_events(settings):
    generate_audit()
    with gzip.open(settings.DIGEST_DIR / 'audit.json.gz', 'rt') as source:
        data = json.load(source)
    return [row for row in data[ROWS]
            if row[3] == S.TABLE_HARVEST and row[4] == S.AUDIT_DELETE]


@pytest.mark.parametrize('batch', [False, True])
def test_api_deletion_lists_nonzero_species_masses(
        harvest_factory, writer_user, settings, batch):
    harvests = [harvest_factory()]
    if batch:
        harvests.append(harvest_factory(mass='400'))
    client = Client()
    client.force_login(writer_user)
    selected = [{ROW_ID: obj.pk, VERSION: obj.version} for obj in harvests]
    body = {ROWS: selected} if batch else selected[0]

    response = client.post(
        '/api/prelievi/delete/', json.dumps(body), content_type='application/json',
    )

    assert response.status_code == 200
    events = deletion_events(settings)
    assert len(events) == len(harvests)
    for event in events:
        assert 'Specie (q): Abete: ' in event[5]
        assert 'Castagno: ' in event[5]
        # Minor species keep their real name rather than becoming a catch-all.
        assert 'Acero: ' in event[5]
        assert 'Altro:' not in event[5]
        assert 'Altra latifoglia:' not in event[5]
        assert event[2] == writer_user.username
        assert event[6] == ''
    texts = [event[5] for event in events]
    assert any('Abete: 100, Castagno: 50, Acero: 50' in text for text in texts)
    if batch:
        assert any('Abete: 200, Castagno: 100, Acero: 100' in text for text in texts)
    assert not HarvestSpecies.objects.exists()


def test_queryset_deletion_freezes_names_and_individual_breakdowns(
        harvest_factory, species, settings):
    first, second = harvest_factory(), harvest_factory(mass='40', percentages=(0, 100))
    ids = [first.pk, second.pk]
    Harvest.objects.filter(pk__in=ids).delete()
    species[0].common_name = 'Abete bianco'
    species[0].save()

    events = deletion_events(settings)

    assert len(events) == 2
    texts = [event[5] for event in events]
    assert any('Abete: 100, Castagno: 50, Acero: 50' in text for text in texts)
    assert any('Specie (q): Castagno: 40' in text for text in texts)
    assert all('Abete bianco:' not in text for text in texts)
    # Regenerating the digest must not change history or its captured names.
    assert deletion_events(settings) == events
    assert Harvest.history.filter(id__in=ids, history_type='-').count() == 2


@pytest.mark.parametrize(('language', 'expected'), [
    ('it', 'Abete: 0,0101, Castagno: 0,9999'),
    ('en', 'Abete: 0.0101, Castagno: 0.9999'),
])
def test_species_mass_is_exact_and_localized(
        harvest_factory, settings, language, expected):
    harvest = harvest_factory(mass='1.01', percentages=(1, 99))
    harvest.delete()
    with translation.override(language):
        events = deletion_events(settings)
    assert expected in events[0][5]


@pytest.mark.parametrize(('mass', 'percentages'), [('0', (50, 50)), ('50', ())])
def test_zero_mass_or_missing_species_has_no_empty_breakdown(
        harvest_factory, settings, mass, percentages):
    harvest = harvest_factory(mass=mass, percentages=percentages)
    harvest.delete()
    assert 'Specie (q)' not in deletion_events(settings)[0][5]
    assert Harvest.history.get(history_type='-').species_masses == []


def test_history_keeps_canonical_snapshot_only_for_deletion(harvest_factory, species):
    harvest = harvest_factory(mass='1.01', percentages=(1, 99))
    harvest.note = 'Corrected note'
    harvest.save()
    assert all(event.species_masses is None for event in harvest.history.all())

    harvest.delete()

    event = Harvest.history.get(history_type='-')
    assert event.species_masses == [
        {'species_id': species[0].pk, 'species_name': 'Abete', 'mass_q': '0.0101'},
        {'species_id': species[1].pk, 'species_name': 'Castagno', 'mass_q': '0.9999'},
    ]


def test_rollback_leaves_no_deletion_history_or_lost_children(harvest_factory):
    harvest = harvest_factory()
    pk = harvest.pk
    with transaction.atomic():
        harvest.delete()
        assert Harvest.history.get(history_type='-').species_masses
        transaction.set_rollback(True)

    assert Harvest.objects.filter(pk=pk).exists()
    assert HarvestSpecies.objects.filter(harvest_id=pk).count() == 4
    assert not Harvest.history.filter(history_type='-').exists()


def test_history_without_a_species_snapshot_still_renders(harvest_factory, settings):
    harvest = harvest_factory()
    harvest.delete()
    # NULL represents a history entry for which no breakdown was captured.
    Harvest.history.filter(history_type='-').update(species_masses=None)
    event, = deletion_events(settings)
    assert 'Q.li: 200' in event[5]
    assert 'Specie (q)' not in event[5]
