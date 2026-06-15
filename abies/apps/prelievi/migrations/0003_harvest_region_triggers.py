"""SQLite triggers for the Harvest region XOR parcel invariant.

Adds three new (or restored) trigger families on prelievi_harvest.

Background: Django's SQLite backend must rebuild the prelievi_harvest table
to make parcel_id nullable (migration 0002 AlterField).  The rebuild drops
the old table — and all its triggers — then creates a fresh one.  Triggers
created in base/0002_triggers on prelievi_harvest therefore no longer exist
after 0002 runs.  This migration restores the missing ones and adds new ones:

1. harvest_region_xor_parcel_{insert,update}: NEW — enforce that exactly one
   of region_id and parcel_id is set on every Harvest row.

2. harvest_parcel_consistency_{insert,update}: RESTORED & UPDATED — the
   base/0002 versions only covered parcel-level harvests (parcel_id NOT NULL).
   The new versions also accept region-wide harvests (parcel_id IS NULL)
   linked to a region-wide HarvestPlanItem.

3. harvest_flags_consistency_{insert,update}: RESTORED — identical to the
   base/0002 versions, which were lost when the table was rebuilt in 0002.

Documented in `docs/database.md`.
"""

from django.db import migrations

# ---------------------------------------------------------------------------
# (1) harvest: region XOR parcel  [NEW]
# ---------------------------------------------------------------------------

HARVEST_XOR_INSERT = """
CREATE TRIGGER IF NOT EXISTS harvest_region_xor_parcel_insert
BEFORE INSERT ON prelievi_harvest
FOR EACH ROW
WHEN (NEW.region_id IS NULL) = (NEW.parcel_id IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'harvest: exactly one of region or parcel must be set');
END;
"""
HARVEST_XOR_UPDATE = HARVEST_XOR_INSERT.replace(
    'harvest_region_xor_parcel_insert', 'harvest_region_xor_parcel_update'
).replace('BEFORE INSERT ON prelievi_harvest',
          'BEFORE UPDATE OF region_id, parcel_id ON prelievi_harvest')
HARVEST_XOR_DROP = """
DROP TRIGGER IF EXISTS harvest_region_xor_parcel_insert;
DROP TRIGGER IF EXISTS harvest_region_xor_parcel_update;
"""

# ---------------------------------------------------------------------------
# (2) harvest.harvest_plan_item parcel/region consistency  [RESTORED + UPDATED]
#
# Replaces the base/0002 harvest_parcel_consistency_* triggers.
# New versions handle both parcel-level and region-wide harvests.
# ---------------------------------------------------------------------------

CONSISTENCY_INSERT = """
CREATE TRIGGER harvest_parcel_consistency_insert
BEFORE INSERT ON prelievi_harvest
FOR EACH ROW
WHEN NEW.harvest_plan_item_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM base_harvestplanitem AS hpi
    WHERE hpi.id = NEW.harvest_plan_item_id
      AND (
        (NEW.parcel_id IS NOT NULL AND (
            hpi.parcel_id = NEW.parcel_id
            OR hpi.region_id = (SELECT region_id FROM base_parcel WHERE id = NEW.parcel_id)))
        OR (NEW.parcel_id IS NULL AND hpi.region_id = NEW.region_id)
      )
)
BEGIN
    SELECT RAISE(ABORT, 'harvest.parcel/region does not match linked harvest_plan_item');
END;
"""
CONSISTENCY_UPDATE = CONSISTENCY_INSERT.replace(
    'harvest_parcel_consistency_insert', 'harvest_parcel_consistency_update'
).replace('BEFORE INSERT ON prelievi_harvest',
          'BEFORE UPDATE OF parcel_id, region_id, harvest_plan_item_id ON prelievi_harvest')

# Reverse: drop new versions and restore the original parcel-only triggers
# from base/0002_triggers.py so that migration reversal is faithful.
CONSISTENCY_REVERSE = """
DROP TRIGGER IF EXISTS harvest_parcel_consistency_insert;
DROP TRIGGER IF EXISTS harvest_parcel_consistency_update;
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

# ---------------------------------------------------------------------------
# (3) harvest.{damaged,unhealthy,psr} flags consistency  [RESTORED]
#
# Identical SQL to base/0002_triggers.py — restored here because the
# AlterField in prelievi/0002 rebuilds the prelievi_harvest table and
# drops all its triggers.
# ---------------------------------------------------------------------------

HARVEST_FLAGS_INSERT = """
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
HARVEST_FLAGS_UPDATE = """
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
HARVEST_FLAGS_DROP = """
DROP TRIGGER IF EXISTS harvest_flags_consistency_insert;
DROP TRIGGER IF EXISTS harvest_flags_consistency_update;
"""


class Migration(migrations.Migration):
    dependencies = [
        ('prelievi', '0002_harvest_region_historicalharvest_region_and_more'),
        ('base', '0002_triggers'),
    ]
    operations = [
        # (1) region XOR parcel — new invariant
        migrations.RunSQL(HARVEST_XOR_INSERT, reverse_sql=HARVEST_XOR_DROP),
        migrations.RunSQL(HARVEST_XOR_UPDATE, reverse_sql=migrations.RunSQL.noop),
        # (2) parcel/region consistency — restored and updated
        migrations.RunSQL(CONSISTENCY_INSERT, reverse_sql=CONSISTENCY_REVERSE),
        migrations.RunSQL(CONSISTENCY_UPDATE, reverse_sql=migrations.RunSQL.noop),
        # (3) flags consistency — restored (lost when table was rebuilt in 0002)
        migrations.RunSQL(HARVEST_FLAGS_INSERT, reverse_sql=HARVEST_FLAGS_DROP),
        migrations.RunSQL(HARVEST_FLAGS_UPDATE, reverse_sql=migrations.RunSQL.noop),
    ]
