"""Tests for base and prelievi models."""

import pytest
from decimal import Decimal

from django.db import IntegrityError, transaction

from apps.base.models import (
    Crew, Eclass, HarvestDetail, HarvestPlan, HarvestPlanItem, Product, Parcel,
    Region, Sample, SampleArea, SampleGrid, Species, Survey, Tractor, Tree,
    TreeMark, TreeSample, User, Role, SiteSettings, DigestStatus, UsedNonce,
)
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor


# ---------------------------------------------------------------------------
# TimestampedModel / version
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# __str__ methods
# ---------------------------------------------------------------------------

class TestStr:
    def test_region(self, regions):
        assert str(regions[0]) == 'Capistrano'

    def test_eclass(self, eclasses):
        assert str(eclasses[0]) == 'A'

    def test_crew(self, crews):
        assert str(crews[0]) == 'Alfa'

    def test_tractor(self, tractors):
        assert str(tractors[0]) == 'Fiat 110-90'

    def test_tractor_name_preferred(self, tractors):
        tractors[0].name = 'T1'
        assert tractors[0].display_name == 'T1'
        assert str(tractors[0]) == 'T1'

    def test_species(self, species):
        assert str(species[0]) == 'Abete'

    def test_product(self, products):
        assert str(products[0]) == 'Tronchi'

    def test_parcel(self, parcels):
        assert str(parcels[0]) == 'Capistrano-1'

    def test_harvest_plan(self, db):
        hp = HarvestPlan.objects.create(
            name='Piano 2020-2030', year_start=2020, year_end=2030,
        )
        assert str(hp) == 'Piano 2020-2030'

    def test_harvest_detail(self, db):
        hd = HarvestDetail.objects.create(description='Cut firs 20-40cm')
        assert str(hd) == 'Cut firs 20-40cm'


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
    @staticmethod
    def _plan_item(parcel):
        plan = HarvestPlan.objects.create(
            name=f'Constraint plan {parcel.id}', year_start=2026, year_end=2036,
        )
        return HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcel, year_planned=2026,
        )

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

    def test_coppice_parcel_requires_interval_and_standards(self, regions, eclasses):
        with pytest.raises(IntegrityError), transaction.atomic():
            Parcel.objects.create(
                name='C1', region=regions[0], eclass=eclasses[2],
                area_ha=Decimal('1.00'),
            )

    def test_highforest_parcel_rejects_coppice_metadata(self, regions, eclasses):
        with pytest.raises(IntegrityError), transaction.atomic():
            Parcel.objects.create(
                name='H1', region=regions[0], eclass=eclasses[0],
                area_ha=Decimal('1.00'), intervention_interval=18,
                standards_per_ha=75,
            )

    def test_coppice_parcel_accepts_interval_and_standards(self, regions, eclasses):
        parcel = Parcel.objects.create(
            name='C1', region=regions[0], eclass=eclasses[2],
            area_ha=Decimal('1.00'), intervention_interval=18,
            standards_per_ha=75,
        )

        assert parcel.intervention_interval == 18
        assert parcel.standards_per_ha == 75

    def test_coppice_parcel_update_requires_interval_and_standards(
            self, regions, eclasses):
        parcel = Parcel.objects.create(
            name='C1', region=regions[0], eclass=eclasses[2],
            area_ha=Decimal('1.00'), intervention_interval=18,
            standards_per_ha=75,
        )
        parcel.intervention_interval = None
        with pytest.raises(IntegrityError), transaction.atomic():
            parcel.save(update_fields=['intervention_interval'])

    def test_eclass_change_to_coppice_requires_parcel_metadata(
            self, regions, eclasses):
        Parcel.objects.create(
            name='H1', region=regions[0], eclass=eclasses[0],
            area_ha=Decimal('1.00'),
        )
        eclasses[0].coppice = True
        with pytest.raises(IntegrityError), transaction.atomic():
            eclasses[0].save(update_fields=['coppice'])

    def test_eclass_change_to_highforest_rejects_coppice_metadata(
            self, regions, eclasses):
        Parcel.objects.create(
            name='C1', region=regions[0], eclass=eclasses[2],
            area_ha=Decimal('1.00'), intervention_interval=18,
            standards_per_ha=75,
        )
        eclasses[2].coppice = False
        with pytest.raises(IntegrityError), transaction.atomic():
            eclasses[2].save(update_fields=['coppice'])

    def test_tree_mark_number_unique_within_item(self, species, parcels):
        planned_item = self._plan_item(parcels[0])
        tree1 = Tree.objects.create(species=species[0])
        tree2 = Tree.objects.create(species=species[0])
        TreeMark.objects.create(
            harvest_plan_item=planned_item, tree=tree1, parcel=parcels[0], number=7,
            date='2026-07-05', d_cm=30, h_m=Decimal('20.0'), operator='Mario',
        )
        with pytest.raises(IntegrityError), transaction.atomic():
            TreeMark.objects.create(
                harvest_plan_item=planned_item, tree=tree2, parcel=parcels[0], number=7,
                date='2026-07-05', d_cm=31, h_m=Decimal('21.0'), operator='Mario',
            )

    def test_tree_mark_null_numbers_are_not_unique(self, species, parcels):
        planned_item = self._plan_item(parcels[0])
        tree1 = Tree.objects.create(species=species[0])
        tree2 = Tree.objects.create(species=species[0])
        TreeMark.objects.create(
            harvest_plan_item=planned_item, tree=tree1, parcel=parcels[0], number=None,
            date='2026-07-05', d_cm=30, h_m=Decimal('20.0'), operator='Mario',
        )
        TreeMark.objects.create(
            harvest_plan_item=planned_item, tree=tree2, parcel=parcels[0], number=None,
            date='2026-07-05', d_cm=31, h_m=Decimal('21.0'), operator='Mario',
        )
        assert TreeMark.objects.filter(harvest_plan_item=planned_item).count() == 2

    def test_tree_sample_number_shoot_unique_within_sample(self, parcels, species):
        grid = SampleGrid.objects.create(name='Unique tree sample grid')
        area = SampleArea.objects.create(
            sample_grid=grid, parcel=parcels[0], number='1',
            lat=38.5, lon=16.3,
        )
        survey = Survey.objects.create(name='Unique tree sample survey', sample_grid=grid)
        sample = Sample.objects.create(sample_area=area, survey=survey, date='2026-07-05')
        tree1 = Tree.objects.create(species=species[0])
        tree2 = Tree.objects.create(species=species[0])
        ts = TreeSample.objects.create(
            sample=sample, tree=tree1, parcel=parcels[0], number=4, shoot=0,
            d_cm=30, h_m=Decimal('20.0'),
        )
        assert ts.h_measured is False
        with pytest.raises(IntegrityError), transaction.atomic():
            TreeSample.objects.create(
                sample=sample, tree=tree2, parcel=parcels[0], number=4, shoot=0,
                d_cm=31, h_m=Decimal('21.0'),
            )

    def test_tree_sample_same_number_different_shoot_ok(self, parcels, species):
        grid = SampleGrid.objects.create(name='Coppice shoot unique grid')
        area = SampleArea.objects.create(
            sample_grid=grid, parcel=parcels[0], number='1',
            lat=38.5, lon=16.3,
        )
        survey = Survey.objects.create(name='Coppice shoot unique survey', sample_grid=grid)
        sample = Sample.objects.create(sample_area=area, survey=survey, date='2026-07-05')
        tree = Tree.objects.create(species=species[0])
        TreeSample.objects.create(
            sample=sample, tree=tree, parcel=parcels[0], number=4, shoot=1,
            d_cm=30, h_m=Decimal('20.0'),
        )
        TreeSample.objects.create(
            sample=sample, tree=tree, parcel=parcels[0], number=4, shoot=2,
            d_cm=31, h_m=Decimal('21.0'),
        )
        assert TreeSample.objects.filter(sample=sample, number=4).count() == 2

    def test_tree_sample_preserved_number_unique_within_sample_parcel(
            self, parcels, species,
    ):
        survey = Survey.objects.create(name='PAI sample uniqueness survey')
        sample = Sample.objects.create(
            sample_area=None, survey=survey, date='2026-07-05',
        )
        tree1 = Tree.objects.create(species=species[0])
        tree2 = Tree.objects.create(species=species[0])
        TreeSample.objects.create(
            sample=sample, tree=tree1, parcel=parcels[0],
            number=1, preserved_number=7, d_cm=30, h_m=Decimal('20.0'),
        )
        with pytest.raises(IntegrityError), transaction.atomic():
            TreeSample.objects.create(
                sample=sample, tree=tree2, parcel=parcels[0],
                number=2, preserved_number=7, d_cm=31, h_m=Decimal('21.0'),
            )

    def test_tree_sample_preserved_number_must_be_positive(self, parcels, species):
        survey = Survey.objects.create(name='PAI positive survey')
        sample = Sample.objects.create(
            sample_area=None, survey=survey, date='2026-07-05',
        )
        tree = Tree.objects.create(species=species[0])
        with pytest.raises(IntegrityError), transaction.atomic():
            TreeSample.objects.create(
                sample=sample, tree=tree, parcel=parcels[0],
                number=1, preserved_number=0, d_cm=30, h_m=Decimal('20.0'),
            )

    def test_tree_sample_height_required_unless_preserved(self, parcels, species):
        survey = Survey.objects.create(name='Height required survey')
        sample = Sample.objects.create(
            sample_area=None, survey=survey, date='2026-07-05',
        )
        tree = Tree.objects.create(species=species[0])

        with pytest.raises(IntegrityError), transaction.atomic():
            TreeSample.objects.create(
                sample=sample, tree=tree, parcel=parcels[0],
                number=1, d_cm=30, h_m=None, h_measured=False,
            )

    def test_preserved_tree_sample_may_have_unknown_height(self, parcels, species):
        survey = Survey.objects.create(name='PAI unknown height survey')
        sample = Sample.objects.create(
            sample_area=None, survey=survey, date='2026-07-05',
        )
        tree = Tree.objects.create(species=species[0])

        ts = TreeSample.objects.create(
            sample=sample, tree=tree, parcel=parcels[0], number=1,
            preserved_number=7, d_cm=30, h_m=None, h_measured=False,
        )

        assert ts.h_m is None
        assert ts.h_measured is False

    def test_unknown_height_cannot_be_marked_measured(self, parcels, species):
        survey = Survey.objects.create(name='Unknown measured height survey')
        sample = Sample.objects.create(
            sample_area=None, survey=survey, date='2026-07-05',
        )
        tree = Tree.objects.create(species=species[0])

        with pytest.raises(IntegrityError), transaction.atomic():
            TreeSample.objects.create(
                sample=sample, tree=tree, parcel=parcels[0], number=1,
                preserved_number=7, d_cm=30, h_m=None, h_measured=True,
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
        assert names == ['Abete', 'Castagno', 'Acero', 'Altro']

    def test_other_sorts_last(self, species):
        last = Species.objects.last()
        assert last.common_name == 'Altro'
        assert last.sort_order == 999


# ---------------------------------------------------------------------------
# Harvest + cascade
# ---------------------------------------------------------------------------

class TestHarvest:
    @pytest.fixture
    def harvest(self, parcels, crews, products):
        return Harvest.objects.create(
            date='2024-03-15', product=products[0], parcel=parcels[0],
            crew=crews[0], mass_q=Decimal('100.50'),
        )

    def test_create(self, harvest):
        assert harvest.pk is not None
        assert harvest.mass_q == Decimal('100.50')

    def test_nullable_record_fields(self, harvest):
        assert harvest.record1 is None
        assert harvest.record2 is None

    def test_cascade_to_species(self, harvest, species):
        HarvestSpecies.objects.create(
            harvest=harvest, species=species[0], percent=100,
        )
        assert HarvestSpecies.objects.count() == 1
        harvest.delete()
        assert HarvestSpecies.objects.count() == 0

    def test_cascade_to_tractor(self, harvest, tractors):
        HarvestTractor.objects.create(
            harvest=harvest, tractor=tractors[0], percent=100,
        )
        assert HarvestTractor.objects.count() == 1
        harvest.delete()
        assert HarvestTractor.objects.count() == 0

    def test_species_protect(self, harvest, species):
        HarvestSpecies.objects.create(
            harvest=harvest, species=species[0], percent=100,
        )
        with pytest.raises(Exception):
            species[0].delete()

    def test_crew_protect(self, harvest):
        with pytest.raises(Exception):
            harvest.crew.delete()


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
    def test_nonce_unique_per_user(self, admin_user, writer_user):
        UsedNonce.objects.create(nonce='abc', user=admin_user, response_json='{}')
        UsedNonce.objects.create(nonce='abc', user=writer_user, response_json='{}')
        with pytest.raises(IntegrityError):
            with transaction.atomic():
                UsedNonce.objects.create(
                    nonce='abc', user=admin_user, response_json='{}',
                )


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

class TestUser:
    def test_default_role_is_reader(self, db):
        u = User.objects.create_user(username='x', password='pass1234!')
        assert u.role == Role.READER

    def test_default_landing_page_is_blank(self, db):
        u = User.objects.create_user(username='x2', password='pass1234!')
        assert u.landing_page == ''

    def test_role_choices(self):
        assert set(Role.values) == {'admin', 'writer', 'reader'}

    def test_history_tracking(self, admin_user):
        admin_user.role = Role.WRITER
        admin_user.save()
        assert admin_user.history.count() == 2  # create + update


# ---------------------------------------------------------------------------
# Site settings
# ---------------------------------------------------------------------------

class TestSiteSettings:
    def test_load_returns_singleton(self, db):
        first = SiteSettings.load()
        second = SiteSettings.load()

        assert first.pk == 1
        assert second.pk == 1
        assert SiteSettings.objects.count() == 1
