"""SQLite triggers enforcing cross-table invariants.

Documented in `docs/database.md`.  Triggers complement Python-level
validation in views and `Model.clean()`; the DB is the last line of
defence against bad writes.

Grouped here in one migration so the trigger set is easy to read; each
RunSQL block has explicit `reverse_sql` so `migrate <name> 0001` drops
the triggers cleanly.
"""

from django.db import migrations


# ---------------------------------------------------------------------------
# (1) sample.sample_grid == sample_area.sample_grid
# ---------------------------------------------------------------------------

SAMPLE_GRID_MATCH_INSERT = """
CREATE TRIGGER IF NOT EXISTS sample_grid_match_insert
BEFORE INSERT ON base_sample
FOR EACH ROW
WHEN (
    SELECT survey.sample_grid_id != sample_area.sample_grid_id
    FROM base_survey AS survey, base_samplearea AS sample_area
    WHERE survey.id = NEW.survey_id AND sample_area.id = NEW.sample_area_id
)
BEGIN
    SELECT RAISE(ABORT, 'sample_grid mismatch: survey.sample_grid != sample_area.sample_grid');
END;
"""

SAMPLE_GRID_MATCH_UPDATE = """
CREATE TRIGGER IF NOT EXISTS sample_grid_match_update
BEFORE UPDATE OF survey_id, sample_area_id ON base_sample
FOR EACH ROW
WHEN (
    SELECT survey.sample_grid_id != sample_area.sample_grid_id
    FROM base_survey AS survey, base_samplearea AS sample_area
    WHERE survey.id = NEW.survey_id AND sample_area.id = NEW.sample_area_id
)
BEGIN
    SELECT RAISE(ABORT, 'sample_grid mismatch: survey.sample_grid != sample_area.sample_grid');
END;
"""

SAMPLE_GRID_MATCH_DROP = """
DROP TRIGGER IF EXISTS sample_grid_match_insert;
DROP TRIGGER IF EXISTS sample_grid_match_update;
"""


# ---------------------------------------------------------------------------
# (2) harvest_plan_item: region XOR parcel (exactly one set)
# ---------------------------------------------------------------------------

HPI_REGION_XOR_PARCEL_INSERT = """
CREATE TRIGGER IF NOT EXISTS hpi_region_xor_parcel_insert
BEFORE INSERT ON base_harvestplanitem
FOR EACH ROW
WHEN (NEW.region_id IS NULL) = (NEW.parcel_id IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'harvest_plan_item: exactly one of region or parcel must be set');
END;
"""

HPI_REGION_XOR_PARCEL_UPDATE = """
CREATE TRIGGER IF NOT EXISTS hpi_region_xor_parcel_update
BEFORE UPDATE OF region_id, parcel_id ON base_harvestplanitem
FOR EACH ROW
WHEN (NEW.region_id IS NULL) = (NEW.parcel_id IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'harvest_plan_item: exactly one of region or parcel must be set');
END;
"""

HPI_REGION_XOR_PARCEL_DROP = """
DROP TRIGGER IF EXISTS hpi_region_xor_parcel_insert;
DROP TRIGGER IF EXISTS hpi_region_xor_parcel_update;
"""


# ---------------------------------------------------------------------------
# (3) harvest.harvest_plan_item parcel consistency
#
# When harvest.harvest_plan_item_id is non-null, either:
#   - the linked item has a parcel and it equals harvest.parcel_id, or
#   - the linked item has a region and it equals the region of harvest.parcel.
# ---------------------------------------------------------------------------

HARVEST_PARCEL_CONSISTENCY_INSERT = """
CREATE TRIGGER IF NOT EXISTS harvest_parcel_consistency_insert
BEFORE INSERT ON prelievi_harvest
FOR EACH ROW
WHEN NEW.harvest_plan_item_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM base_harvestplanitem AS hpi, base_parcel AS p
    WHERE hpi.id = NEW.harvest_plan_item_id
      AND p.id = NEW.parcel_id
      AND (
          hpi.parcel_id = NEW.parcel_id
          OR hpi.region_id = p.region_id
      )
)
BEGIN
    SELECT RAISE(ABORT, 'harvest.parcel does not match linked harvest_plan_item parcel or region');
END;
"""

HARVEST_PARCEL_CONSISTENCY_UPDATE = """
CREATE TRIGGER IF NOT EXISTS harvest_parcel_consistency_update
BEFORE UPDATE OF parcel_id, harvest_plan_item_id ON prelievi_harvest
FOR EACH ROW
WHEN NEW.harvest_plan_item_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM base_harvestplanitem AS hpi, base_parcel AS p
    WHERE hpi.id = NEW.harvest_plan_item_id
      AND p.id = NEW.parcel_id
      AND (
          hpi.parcel_id = NEW.parcel_id
          OR hpi.region_id = p.region_id
      )
)
BEGIN
    SELECT RAISE(ABORT, 'harvest.parcel does not match linked harvest_plan_item parcel or region');
END;
"""

HARVEST_PARCEL_CONSISTENCY_DROP = """
DROP TRIGGER IF EXISTS harvest_parcel_consistency_insert;
DROP TRIGGER IF EXISTS harvest_parcel_consistency_update;
"""


# ---------------------------------------------------------------------------
# (4) harvest.{damaged,unhealthy,psr} == harvest_plan_item.{...} when linked
# ---------------------------------------------------------------------------

HARVEST_FLAGS_CONSISTENCY_INSERT = """
CREATE TRIGGER IF NOT EXISTS harvest_flags_consistency_insert
BEFORE INSERT ON prelievi_harvest
FOR EACH ROW
WHEN NEW.harvest_plan_item_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM base_harvestplanitem AS hpi
    WHERE hpi.id = NEW.harvest_plan_item_id
      AND hpi.damaged   = NEW.damaged
      AND hpi.unhealthy = NEW.unhealthy
      AND hpi.psr       = NEW.psr
)
BEGIN
    SELECT RAISE(ABORT, 'harvest flags must match linked harvest_plan_item flags');
END;
"""

HARVEST_FLAGS_CONSISTENCY_UPDATE = """
CREATE TRIGGER IF NOT EXISTS harvest_flags_consistency_update
BEFORE UPDATE OF damaged, unhealthy, psr, harvest_plan_item_id ON prelievi_harvest
FOR EACH ROW
WHEN NEW.harvest_plan_item_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM base_harvestplanitem AS hpi
    WHERE hpi.id = NEW.harvest_plan_item_id
      AND hpi.damaged   = NEW.damaged
      AND hpi.unhealthy = NEW.unhealthy
      AND hpi.psr       = NEW.psr
)
BEGIN
    SELECT RAISE(ABORT, 'harvest flags must match linked harvest_plan_item flags');
END;
"""

HARVEST_FLAGS_CONSISTENCY_DROP = """
DROP TRIGGER IF EXISTS harvest_flags_consistency_insert;
DROP TRIGGER IF EXISTS harvest_flags_consistency_update;
"""


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0001_initial'),
        ('prelievi', '0001_initial'),
    ]

    operations = [
        migrations.RunSQL(SAMPLE_GRID_MATCH_INSERT, reverse_sql=SAMPLE_GRID_MATCH_DROP),
        migrations.RunSQL(SAMPLE_GRID_MATCH_UPDATE, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(HPI_REGION_XOR_PARCEL_INSERT, reverse_sql=HPI_REGION_XOR_PARCEL_DROP),
        migrations.RunSQL(HPI_REGION_XOR_PARCEL_UPDATE, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(HARVEST_PARCEL_CONSISTENCY_INSERT, reverse_sql=HARVEST_PARCEL_CONSISTENCY_DROP),
        migrations.RunSQL(HARVEST_PARCEL_CONSISTENCY_UPDATE, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(HARVEST_FLAGS_CONSISTENCY_INSERT, reverse_sql=HARVEST_FLAGS_CONSISTENCY_DROP),
        migrations.RunSQL(HARVEST_FLAGS_CONSISTENCY_UPDATE, reverse_sql=migrations.RunSQL.noop),
    ]
