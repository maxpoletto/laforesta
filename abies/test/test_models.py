"""Tests for base and prelievi models."""

import pytest
from decimal import Decimal

from django.db import IntegrityError

from apps.base.models import (
    Crew, Eclass, Parcel, Region, Species, Tractor, User, Role,
    DigestStatus, UsedNonce,
)
from apps.prelievi.models import HarvestOp, HarvestSpecies, HarvestTractor


# ---------------------------------------------------------------------------
# TimestampedModel / version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_crew_starts_at_version_1(self, crews):
        assert crews[0].version == 1

    def test_version_not_auto_incremented_on_save(self, crews):
        """Version is managed by application code, not auto-incremented."""
        c = crews[0]
        c.notes = 'updated'
        c.save()
        c.refresh_from_db()
        assert c.version == 1  # unchanged — app code must bump it

    def test_timestamps_populated(self, crews):
        c = crews[0]
        assert c.created_at is not None
        assert c.modified_at is not None


# ---------------------------------------------------------------------------
# Unique constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_region_name_unique(self, regions):
        with pytest.raises(IntegrityError):
            Region.objects.create(name='Capistrano')

    def test_species_common_name_unique(self, species):
        with pytest.raises(IntegrityError):
            Species.objects.create(common_name='Abete')

    def test_parcel_unique_together(self, parcels, eclasses):
        with pytest.raises(IntegrityError):
            Parcel.objects.create(
                name='1', region=parcels[0].region, eclass=eclasses[0],
                area_ha=Decimal('1'),
            )

    def test_parcel_same_name_different_region_ok(self, parcels):
        """Parcel '1' in Capistrano and Fabrizia are distinct."""
        names = [p.name for p in parcels if p.name == '1']
        assert len(names) == 2


# ---------------------------------------------------------------------------
# FK protection
# ---------------------------------------------------------------------------

class TestFKProtection:
    def test_delete_region_with_parcel_blocked(self, parcels):
        with pytest.raises(Exception):  # PROTECT
            parcels[0].region.delete()

    def test_delete_eclass_with_parcel_blocked(self, parcels):
        with pytest.raises(Exception):
            parcels[0].eclass.delete()

    def test_delete_parcel_ok(self, parcels):
        pk = parcels[0].pk
        parcels[0].delete()
        assert not Parcel.objects.filter(pk=pk).exists()


# ---------------------------------------------------------------------------
# Species ordering
# ---------------------------------------------------------------------------

class TestSpeciesOrdering:
    def test_default_ordering_by_sort_order(self, species):
        names = list(Species.objects.values_list('common_name', flat=True))
        assert names == ['Abete', 'Castagno', 'Altro']

    def test_altro_sorts_last(self, species):
        last = Species.objects.last()
        assert last.common_name == 'Altro'
        assert last.sort_order == 999


# ---------------------------------------------------------------------------
# HarvestOp + cascade
# ---------------------------------------------------------------------------

class TestHarvestOp:
    @pytest.fixture
    def harvest_op(self, parcels, crews, optypes):
        return HarvestOp.objects.create(
            date='2024-03-15', optype=optypes[0], parcel=parcels[0],
            crew=crews[0], quintals=Decimal('100.50'),
        )

    def test_create(self, harvest_op):
        assert harvest_op.pk is not None
        assert harvest_op.quintals == Decimal('100.50')

    def test_nullable_record_fields(self, harvest_op):
        assert harvest_op.record1 is None
        assert harvest_op.record2 is None

    def test_cascade_to_species(self, harvest_op, species):
        HarvestSpecies.objects.create(
            harvest_op=harvest_op, species=species[0], percent=100,
        )
        assert HarvestSpecies.objects.count() == 1
        harvest_op.delete()
        assert HarvestSpecies.objects.count() == 0

    def test_cascade_to_tractor(self, harvest_op, tractors):
        HarvestTractor.objects.create(
            harvest_op=harvest_op, tractor=tractors[0], percent=100,
        )
        assert HarvestTractor.objects.count() == 1
        harvest_op.delete()
        assert HarvestTractor.objects.count() == 0

    def test_species_protect(self, harvest_op, species):
        HarvestSpecies.objects.create(
            harvest_op=harvest_op, species=species[0], percent=100,
        )
        with pytest.raises(Exception):
            species[0].delete()

    def test_crew_protect(self, harvest_op):
        with pytest.raises(Exception):
            harvest_op.crew.delete()


# ---------------------------------------------------------------------------
# DigestStatus
# ---------------------------------------------------------------------------

class TestDigestStatus:
    def test_create_and_defaults(self, db):
        ds = DigestStatus.objects.create(name='test')
        assert ds.stale is False

    def test_pk_is_name(self, db):
        ds = DigestStatus.objects.create(name='test')
        assert ds.pk == 'test'


# ---------------------------------------------------------------------------
# UsedNonce
# ---------------------------------------------------------------------------

class TestUsedNonce:
    def test_nonce_unique(self, admin_user):
        UsedNonce.objects.create(nonce='abc', user=admin_user, response_json='{}')
        with pytest.raises(IntegrityError):
            UsedNonce.objects.create(nonce='abc', user=admin_user, response_json='{}')


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

class TestUser:
    def test_default_role_is_reader(self, db):
        u = User.objects.create_user(username='x', password='pass1234!')
        assert u.role == Role.READER

    def test_role_choices(self):
        assert set(Role.values) == {'admin', 'writer', 'reader'}

    def test_history_tracking(self, admin_user):
        admin_user.role = Role.WRITER
        admin_user.save()
        assert admin_user.history.count() == 2  # create + update
