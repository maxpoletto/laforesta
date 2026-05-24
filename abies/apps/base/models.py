"""Base models — shared across all domains."""

import re
from decimal import Decimal

from django.contrib.auth.models import AbstractUser
from django.db import models
from simple_history.models import HistoricalRecords

from config import strings as S

_NATSORT_RE = re.compile(r'(\d+)')


def _natsort_key(s: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in _NATSORT_RE.split(s)]


def parcel_sort_key(p) -> tuple:
    """Sort key for natural ordering: region name, then parcel name."""
    return (p.region.name, _natsort_key(p.name))


def render_flag_note(damaged: bool, unhealthy: bool, psr: bool) -> str:
    """Comma-joined Italian string for the damaged/unhealthy/psr flags.

    Used by the prelievi digest, the harvest_plan_items digest, and any
    UI surface that renders the Note column.  At most two of the three
    co-occur in practice.  Empty string when no flag is set.
    """
    parts = []
    if damaged:
        parts.append(S.FLAG_DAMAGED)
    if unhealthy:
        parts.append(S.FLAG_UNHEALTHY)
    if psr:
        parts.append(S.FLAG_PSR)
    return ', '.join(parts)


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
    density = models.DecimalField(
        max_digits=5, decimal_places=2, default=Decimal('5.00'),
        help_text='Wood density in q/m³ (typical range 4–8).',
    )
    active = models.BooleanField(default=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.SPECIES
        verbose_name_plural = S.SPECIES_PLURAL
        ordering = ['sort_order']

    def __str__(self):
        return self.common_name


class Product(models.Model):
    """Harvested product type (extensible enum)."""
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        verbose_name = S.PRODUCT
        verbose_name_plural = S.PRODUCTS

    def __str__(self):
        return self.name


# ---------------------------------------------------------------------------
# Forest structure
# ---------------------------------------------------------------------------

class HarvestPlan(TimestampedModel):
    """Multi-year harvest plan.

    Writer-editable from the plan-selector pencil affordance; tracked
    by HistoricalRecords for audit and carries `version` for optimistic
    locking, like every other mutable domain table.
    """
    name = models.CharField(max_length=100, unique=True)
    year_start = models.IntegerField()
    year_end = models.IntegerField()
    description = models.TextField(blank=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.HARVEST_PLAN
        verbose_name_plural = S.HARVEST_PLANS

    def __str__(self):
        return self.name


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

    class Meta:
        verbose_name = S.PARCEL
        verbose_name_plural = S.PARCELS
        unique_together = [('name', 'region')]

    def __str__(self):
        return f'{self.region.name}-{self.name}'


class HarvestPlanItemState(models.IntegerChoices):
    """State machine for HarvestPlanItem.

    Transitions are monotonic — state only advances. Auto-transitions:
    `planned → marked` on first linked TreeMark; `open|marked → harvesting`
    on first linked Harvest. Manual transitions via HarvestTransition:
    `planned|marked → open` (Apri cantiere) and `open|harvesting → closed`
    (Chiudi cantiere). Coppice items skip `marked`.
    """
    PLANNED    = 0, S.STATE_PLANNED
    MARKED     = 1, S.STATE_MARKED
    OPEN       = 2, S.STATE_OPEN
    HARVESTING = 3, S.STATE_HARVESTING
    CLOSED     = 4, S.STATE_CLOSED


class HarvestPlanItem(TimestampedModel):
    """Scheduled harvest of a parcel (or region-wide intervento) within a
    plan.

    Either `region` or `parcel` is set, never both (enforced by SQLite
    trigger; see migration 0002_triggers.py). Volumes are materialized
    aggregates over the linked tree_marks and harvests; see
    `docs/database.md` for the invalidation chain.
    """
    harvest_plan = models.ForeignKey(HarvestPlan, on_delete=models.CASCADE)
    # Region XOR parcel.  Region-wide items exist for damaged / unhealthy
    # operations approved across an entire region.
    region = models.ForeignKey(
        Region, on_delete=models.PROTECT, null=True, blank=True,
        related_name='harvest_plan_items',
    )
    parcel = models.ForeignKey(
        Parcel, on_delete=models.PROTECT, null=True, blank=True,
        related_name='harvest_plan_items',
    )
    state = models.IntegerField(
        choices=HarvestPlanItemState.choices,
        default=HarvestPlanItemState.PLANNED,
    )
    year_planned = models.IntegerField()
    # Date of the implicit planned → marked transition (= date of the
    # earliest linked TreeMark). For coppice items: date of the first
    # linked Harvest. NULL while state is still `planned`.
    date_actual = models.DateField(null=True, blank=True)
    # Materialized volume aggregates. NULL for coppice items where it
    # does not apply.
    volume_planned_m3 = models.DecimalField(
        max_digits=10, decimal_places=3, null=True, blank=True,
        help_text='Planned cut volume from CSV import; NULL for coppice.',
    )
    volume_marked_m3 = models.DecimalField(
        max_digits=10, decimal_places=3, null=True, blank=True,
        help_text='SUM(tree_mark.volume_m3); NULL for coppice items.',
    )
    volume_actual_m3 = models.DecimalField(
        max_digits=10, decimal_places=3, default=0,
        help_text='SUM(harvest.volume_m3) over linked harvests.',
    )
    intervention_area_ha = models.DecimalField(
        max_digits=7, decimal_places=2, null=True, blank=True,
        help_text='Coppice staged-cut area; NULL for whole-parcel cuts.',
    )
    damaged = models.BooleanField(default=False)
    unhealthy = models.BooleanField(default=False)
    psr = models.BooleanField(default=False)
    note = models.CharField(max_length=255, blank=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.HARVEST_PLAN_ITEM
        verbose_name_plural = S.HARVEST_PLAN_ITEMS

    def clean(self):
        """App-side mirror of the SQLite triggers.

        The DB enforces region XOR parcel and the state-monotonic
        invariant via triggers; this method gives early, localised
        feedback during form save.
        """
        from django.core.exceptions import ValidationError
        if (self.region_id is None) == (self.parcel_id is None):
            raise ValidationError(
                'Exactly one of region or parcel must be set.'
            )
        if self.pk is not None:
            old_state = (
                type(self).objects.filter(pk=self.pk).values_list('state', flat=True).first()
            )
            if old_state is not None and self.state < old_state:
                raise ValidationError(
                    f'State cannot regress: {old_state} → {self.state}'
                )

    def __str__(self):
        scope = self.parcel or self.region or '?'
        return f'{self.harvest_plan.name} {self.year_planned} {scope}'


class HarvestTransition(models.Model):
    """Open / close event on a HarvestPlanItem cantiere.

    Each item has at most one open row and at most one close row. The
    item's `state` is advanced server-side when a transition row is
    written.
    """
    harvest_plan_item = models.ForeignKey(
        HarvestPlanItem, on_delete=models.PROTECT,
        related_name='transitions',
    )
    open = models.BooleanField(help_text='True = Apri cantiere, False = Chiudi cantiere.')
    date = models.DateField()
    note = models.CharField(
        max_length=255, blank=True,
        help_text='Typically the regional permit number.',
    )

    class Meta:
        verbose_name = S.HARVEST_TRANSITION
        verbose_name_plural = S.HARVEST_TRANSITIONS
        unique_together = [('harvest_plan_item', 'open')]


class TreeHeightRegression(models.Model):
    """Per-(plan, region, species) ipsometric regression coefficients.

    Used by the Nuovo albero martellato form to auto-populate `h_m`
    from `d_cm` via `h = a × ln(D) + b` (when function='ln').
    """
    harvest_plan = models.ForeignKey(HarvestPlan, on_delete=models.CASCADE)
    region = models.ForeignKey(Region, on_delete=models.PROTECT)
    species = models.ForeignKey(Species, on_delete=models.PROTECT)
    function = models.CharField(max_length=10, default='ln')
    a = models.DecimalField(max_digits=10, decimal_places=4)
    b = models.DecimalField(max_digits=10, decimal_places=4)
    r2 = models.DecimalField(max_digits=6, decimal_places=4)
    n = models.IntegerField()

    class Meta:
        verbose_name = S.TREE_HEIGHT_REGRESSION
        verbose_name_plural = S.TREE_HEIGHT_REGRESSIONS
        unique_together = [('harvest_plan', 'region', 'species')]


class ParcelPlanDetail(models.Model):
    """Maps a harvest detail to a parcel within a plan."""
    harvest_plan = models.ForeignKey(HarvestPlan, on_delete=models.CASCADE)
    parcel = models.ForeignKey(Parcel, on_delete=models.CASCADE)
    harvest_detail = models.ForeignKey(HarvestDetail, on_delete=models.CASCADE)

    class Meta:
        verbose_name = S.PARCEL_PLAN_DETAIL
        verbose_name_plural = S.PARCEL_PLAN_DETAILS
        unique_together = [('harvest_plan', 'parcel')]


class SampleGrid(TimestampedModel):
    """A named layout of sample areas across one or more regions.

    Multiple surveys can reference the same grid over time.  Each
    `sample_area` row belongs to exactly one grid via FK.
    """
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.SAMPLE_GRID
        verbose_name_plural = S.SAMPLE_GRIDS

    def __str__(self):
        return self.name


class SampleArea(TimestampedModel):
    """Geo-referenced sample plot (area di saggio)."""
    sample_grid = models.ForeignKey(SampleGrid, on_delete=models.CASCADE)
    # `number` is a manually assigned identifier, usually a numeric string
    # like "27", occasionally with a suffix like "27 bis" for late
    # additions.  Unique within a parcel; need not be unique across.
    number = models.CharField(max_length=20)
    parcel = models.ForeignKey(Parcel, on_delete=models.CASCADE)
    lat = models.FloatField()
    lon = models.FloatField()
    altitude_m = models.IntegerField(null=True, blank=True)
    r_m = models.IntegerField(default=12)
    note = models.CharField(max_length=255, blank=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.SAMPLE_AREA
        verbose_name_plural = S.SAMPLE_AREAS
        unique_together = [('sample_grid', 'parcel', 'number')]


class Survey(TimestampedModel):
    """A high-level sampling operation against a specific grid.

    Optionally linked to a harvest plan; otherwise an ad-hoc research
    survey.  Completeness is computed: a survey is "complete" when
    every area in its grid has at least one Sample.
    """
    name = models.CharField(max_length=100, unique=True)
    harvest_plan = models.ForeignKey(
        HarvestPlan, on_delete=models.SET_NULL, null=True, blank=True,
    )
    sample_grid = models.ForeignKey(SampleGrid, on_delete=models.PROTECT)
    description = models.TextField(blank=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.SURVEY
        verbose_name_plural = S.SURVEYS

    def __str__(self):
        return self.name


class Sample(TimestampedModel):
    """A single visit to one sample area within a survey.

    Schema-level invariant (enforced by SQLite trigger in the migration):
    `sample_area.sample_grid_id == survey.sample_grid_id`.
    """
    sample_area = models.ForeignKey(SampleArea, on_delete=models.CASCADE)
    survey = models.ForeignKey(Survey, on_delete=models.CASCADE)
    date = models.DateField()
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.SAMPLE
        verbose_name_plural = S.SAMPLES
        unique_together = [('sample_area', 'survey')]


class Tree(TimestampedModel):
    """A physical tree, tracked over time.

    Sampled trees can be revisited across surveys (same `tree_id`);
    marked trees and PAI trees are static observations.  See
    `database.md` for the cross-sample-identity convention.
    """
    species = models.ForeignKey(Species, on_delete=models.PROTECT)
    year = models.IntegerField(null=True, blank=True)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    acc_m = models.IntegerField(
        null=True, blank=True,
        help_text='Reported GPS accuracy in meters (e.g., from ipso PWA).',
    )
    parcel = models.ForeignKey(Parcel, on_delete=models.PROTECT)
    preserved = models.BooleanField(default=False)
    coppice = models.BooleanField(default=False)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.TREE
        verbose_name_plural = S.TREES


class TreeMark(TimestampedModel):
    """A (high-forest) tree marked for felling, under a HarvestPlanItem.

    Synthetic `id` PK plus UNIQUE(harvest_plan_item, tree) preserves the
    natural compound key. There is no separate `Mark` aggregate row: the
    set of TreeMark rows sharing a harvest_plan_item is "the mark" in
    user-facing language.
    """
    harvest_plan_item = models.ForeignKey(
        HarvestPlanItem, on_delete=models.PROTECT,
        related_name='tree_marks',
    )
    tree = models.ForeignKey(Tree, on_delete=models.CASCADE)
    date = models.DateField()
    d_cm = models.IntegerField()
    h_m = models.DecimalField(max_digits=5, decimal_places=2)
    h_measured = models.BooleanField(default=False)
    volume_m3 = models.DecimalField(max_digits=8, decimal_places=4)
    mass_q = models.DecimalField(max_digits=8, decimal_places=3)
    lat = models.FloatField()
    lon = models.FloatField()
    acc_m = models.IntegerField(null=True, blank=True)
    operator = models.CharField(max_length=100)
    # Row-content fingerprint used for idempotent CSV re-imports (see
    # `docs/pages/piano-di-taglio.md` "Importa CSV martellate").  Null
    # for manually entered rows.
    import_fingerprint = models.CharField(max_length=64, null=True, blank=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.TREE_MARK
        verbose_name_plural = S.TREE_MARKS
        unique_together = [('harvest_plan_item', 'tree')]


class TreeSample(TimestampedModel):
    """Measurements taken on a tree during a sample.

    Synthetic `id` PK + UNIQUE(sample, tree, shoot) preserves the
    natural compound key while keeping row_id semantics uniform with
    other Abies tables.
    """
    sample = models.ForeignKey(Sample, on_delete=models.CASCADE)
    tree = models.ForeignKey(Tree, on_delete=models.PROTECT)
    shoot = models.IntegerField(default=0)
    standard = models.BooleanField(default=False)
    number = models.IntegerField()
    d_cm = models.IntegerField()
    h_m = models.DecimalField(max_digits=5, decimal_places=2)
    l10_mm = models.IntegerField(default=0)
    volume_m3 = models.DecimalField(
        max_digits=8, decimal_places=4, null=True, blank=True,
        help_text='Tabacchi-derived volume; NULL for coppice rows.',
    )
    mass_q = models.DecimalField(
        max_digits=8, decimal_places=3, null=True, blank=True,
        help_text='volume_m3 × species.density; NULL for coppice rows.',
    )
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.TREE_SAMPLE
        verbose_name_plural = S.TREE_SAMPLES
        unique_together = [('sample', 'tree', 'shoot')]


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
