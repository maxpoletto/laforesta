"""Base models — shared across all domains."""

from django.contrib.auth.models import AbstractUser
from django.db import models
from simple_history.models import HistoricalRecords

from config import strings as S


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TimestampedModel(models.Model):
    """Optimistic-locking version + timestamps on every mutable table."""
    version = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class Role(models.TextChoices):
    ADMIN = 'admin', 'Admin'
    WRITER = 'writer', 'Writer'
    READER = 'reader', 'Reader'


class LoginMethod(models.TextChoices):
    PASSWORD = 'password', 'Password'
    OAUTH = 'oauth', 'OAuth'


class User(AbstractUser):
    role = models.CharField(max_length=10, choices=Role.choices, default=Role.READER)
    login_method = models.CharField(
        max_length=10, choices=LoginMethod.choices, default=LoginMethod.PASSWORD,
    )
    history = HistoricalRecords()

    class Meta(AbstractUser.Meta):
        verbose_name = S.USER
        verbose_name_plural = S.USERS


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

class Region(models.Model):
    """Forest region (compresa)."""
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        verbose_name = S.REGION
        verbose_name_plural = S.REGIONS

    def __str__(self):
        return self.name


class Eclass(models.Model):
    """Parcel economic class (comparto)."""
    name = models.CharField(max_length=10, unique=True)
    coppice = models.BooleanField(default=False)
    min_harvest_volume = models.IntegerField(default=0)

    class Meta:
        verbose_name = S.ECLASS
        verbose_name_plural = S.ECLASSES

    def __str__(self):
        return self.name


class Crew(TimestampedModel):
    """Team of lumberjacks (squadra)."""
    name = models.CharField(max_length=100, unique=True)
    notes = models.TextField(blank=True)
    active = models.BooleanField(default=True)

    class Meta:
        verbose_name = S.CREW
        verbose_name_plural = S.CREWS

    history = HistoricalRecords()

    def __str__(self):
        return self.name


class Tractor(TimestampedModel):
    """Tractor used in harvest operations."""
    manufacturer = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    year = models.IntegerField(null=True, blank=True)
    active = models.BooleanField(default=True)

    history = HistoricalRecords()

    class Meta:
        verbose_name = S.TRACTOR
        verbose_name_plural = S.TRACTORS

    def __str__(self):
        return f'{self.manufacturer} {self.model}'


class Species(TimestampedModel):
    """Tree species."""
    common_name = models.CharField(max_length=100, unique=True)
    latin_name = models.CharField(max_length=100, blank=True)
    sort_order = models.IntegerField(default=0)
    active = models.BooleanField(default=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.SPECIES
        verbose_name_plural = S.SPECIES_PLURAL
        ordering = ['sort_order']

    def __str__(self):
        return self.common_name


class Optype(models.Model):
    """Harvest operation type (extensible enum)."""
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        verbose_name = S.OPTYPE
        verbose_name_plural = S.OPTYPES

    def __str__(self):
        return self.name


class Note(models.Model):
    """Harvest note category (extensible enum)."""
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        verbose_name = S.NOTE
        verbose_name_plural = S.NOTES

    def __str__(self):
        return self.name


# ---------------------------------------------------------------------------
# Forest structure
# ---------------------------------------------------------------------------

class HarvestPlan(models.Model):
    """Multi-year harvest plan."""
    year_start = models.IntegerField()
    year_end = models.IntegerField()
    description = models.TextField(blank=True)

    class Meta:
        verbose_name = S.HARVEST_PLAN
        verbose_name_plural = S.HARVEST_PLANS

    def __str__(self):
        return f'{self.year_start}–{self.year_end}'


class HarvestDetail(models.Model):
    """Reusable harvest instruction for a parcel."""
    description = models.TextField()
    interval = models.IntegerField(null=True, blank=True)

    class Meta:
        verbose_name = S.HARVEST_DETAIL
        verbose_name_plural = S.HARVEST_DETAILS

    def __str__(self):
        return self.description[:80]


class Parcel(models.Model):
    """Forest parcel (particella)."""
    name = models.CharField(max_length=20)
    region = models.ForeignKey(Region, on_delete=models.PROTECT)
    eclass = models.ForeignKey(Eclass, on_delete=models.PROTECT)
    area_ha = models.DecimalField(max_digits=7, decimal_places=2)
    ave_age = models.IntegerField(null=True, blank=True)
    location_name = models.CharField(max_length=200, blank=True)
    altitude_min_m = models.IntegerField(null=True, blank=True)
    altitude_max_m = models.IntegerField(null=True, blank=True)
    aspect = models.CharField(max_length=20, blank=True)
    grade_pct = models.IntegerField(null=True, blank=True)
    desc_veg = models.TextField(blank=True)
    desc_geo = models.TextField(blank=True)
    harvest_plan = models.ForeignKey(
        HarvestPlan, on_delete=models.SET_NULL, null=True, blank=True,
    )

    class Meta:
        verbose_name = S.PARCEL
        verbose_name_plural = S.PARCELS
        unique_together = [('name', 'region')]

    def __str__(self):
        return f'{self.region.name}-{self.name}'


class HarvestPlanItem(models.Model):
    """Scheduled harvest of a parcel within a plan."""
    harvest_plan = models.ForeignKey(HarvestPlan, on_delete=models.CASCADE)
    parcel = models.ForeignKey(Parcel, on_delete=models.CASCADE)
    year = models.IntegerField()
    quintals = models.IntegerField(default=0)

    class Meta:
        verbose_name = S.HARVEST_PLAN_ITEM
        verbose_name_plural = S.HARVEST_PLAN_ITEMS
        unique_together = [('harvest_plan', 'parcel', 'year')]


class ParcelPlanDetail(models.Model):
    """Maps a harvest detail to a parcel within a plan."""
    harvest_plan = models.ForeignKey(HarvestPlan, on_delete=models.CASCADE)
    parcel = models.ForeignKey(Parcel, on_delete=models.CASCADE)
    harvest_detail = models.ForeignKey(HarvestDetail, on_delete=models.CASCADE)

    class Meta:
        verbose_name = S.PARCEL_PLAN_DETAIL
        verbose_name_plural = S.PARCEL_PLAN_DETAILS
        unique_together = [('harvest_plan', 'parcel')]


class SampleArea(models.Model):
    """Geo-referenced sample plot (area di saggio)."""
    number = models.IntegerField()
    parcel = models.ForeignKey(Parcel, on_delete=models.CASCADE)
    lat = models.FloatField()
    lng = models.FloatField()
    altitude_m = models.IntegerField(null=True, blank=True)
    plan_year = models.IntegerField()

    class Meta:
        verbose_name = S.SAMPLE_AREA
        verbose_name_plural = S.SAMPLE_AREAS


class PreservedTree(models.Model):
    """Tree marked for preservation (pianta ad accrescimento indefinito)."""
    species = models.ForeignKey(Species, on_delete=models.PROTECT)
    region = models.ForeignKey(Region, on_delete=models.PROTECT)
    parcel = models.ForeignKey(Parcel, on_delete=models.CASCADE)
    lat = models.FloatField()
    lng = models.FloatField()
    note = models.TextField(blank=True)

    class Meta:
        verbose_name = S.PRESERVED_TREE
        verbose_name_plural = S.PRESERVED_TREES


# ---------------------------------------------------------------------------
# Digest staleness tracking
# ---------------------------------------------------------------------------

class DigestStatus(models.Model):
    """Tracks whether a pre-computed JSON digest needs regeneration."""
    name = models.CharField(max_length=100, primary_key=True)
    stale = models.BooleanField(default=False)

    class Meta:
        verbose_name = S.DIGEST_STATUS
        verbose_name_plural = S.DIGEST_STATUSES


# ---------------------------------------------------------------------------
# Idempotency nonces
# ---------------------------------------------------------------------------

class UsedNonce(models.Model):
    """Prevents duplicate form submissions."""
    nonce = models.CharField(max_length=64, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    response_json = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = S.USED_NONCE
        verbose_name_plural = S.USED_NONCES
