from django.db import migrations


HARVEST_XOR_INSERT = """
CREATE TRIGGER IF NOT EXISTS harvest_region_xor_parcel_insert
BEFORE INSERT ON prelievi_harvest
FOR EACH ROW
WHEN (NEW.region_id IS NULL) = (NEW.parcel_id IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'harvest: exactly one of region or parcel must be set');
END;
"""
HARVEST_XOR_UPDATE = """
CREATE TRIGGER IF NOT EXISTS harvest_region_xor_parcel_update
BEFORE UPDATE OF region_id, parcel_id ON prelievi_harvest
FOR EACH ROW
WHEN (NEW.region_id IS NULL) = (NEW.parcel_id IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'harvest: exactly one of region or parcel must be set');
END;
"""

CONSISTENCY_INSERT = """
CREATE TRIGGER IF NOT EXISTS harvest_parcel_consistency_insert
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
CONSISTENCY_UPDATE = """
CREATE TRIGGER IF NOT EXISTS harvest_parcel_consistency_update
BEFORE UPDATE OF parcel_id, region_id, harvest_plan_item_id ON prelievi_harvest
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

DROP_TRIGGERS = """
DROP TRIGGER IF EXISTS harvest_region_xor_parcel_insert;
DROP TRIGGER IF EXISTS harvest_region_xor_parcel_update;
DROP TRIGGER IF EXISTS harvest_parcel_consistency_insert;
DROP TRIGGER IF EXISTS harvest_parcel_consistency_update;
DROP TRIGGER IF EXISTS harvest_flags_consistency_insert;
DROP TRIGGER IF EXISTS harvest_flags_consistency_update;
"""


class Migration(migrations.Migration):
    dependencies = [
        ('prelievi', '0002_harvest_import_fingerprint'),
    ]

    operations = [
        migrations.RunSQL(HARVEST_XOR_INSERT, reverse_sql=DROP_TRIGGERS),
        migrations.RunSQL(HARVEST_XOR_UPDATE, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(CONSISTENCY_INSERT, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(CONSISTENCY_UPDATE, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(HARVEST_FLAGS_INSERT, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(HARVEST_FLAGS_UPDATE, reverse_sql=migrations.RunSQL.noop),
    ]
