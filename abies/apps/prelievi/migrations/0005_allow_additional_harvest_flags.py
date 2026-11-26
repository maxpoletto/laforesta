from importlib import import_module

from django.db import migrations, models
from django.db.migrations.exceptions import IrreversibleError


previous = import_module('apps.prelievi.migrations.0003_harvest_consistency_triggers')
DROP_FLAGS = """
DROP TRIGGER IF EXISTS harvest_flags_consistency_insert;
DROP TRIGGER IF EXISTS harvest_flags_consistency_update;
"""


def require_worksite_flags(suffix, event):
    return f"""
CREATE TRIGGER harvest_flags_consistency_{suffix}
BEFORE {event} ON prelievi_harvest
FOR EACH ROW
WHEN NEW.harvest_plan_item_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM base_harvestplanitem AS item
    WHERE item.id = NEW.harvest_plan_item_id
      AND ((item.damaged AND NOT NEW.damaged)
        OR (item.unhealthy AND NOT NEW.unhealthy)
        OR (item.psr AND NOT NEW.psr))
)
BEGIN
    SELECT RAISE(ABORT, 'harvest must keep the flags of its linked worksite');
END;
"""


def check_rollback(apps, schema_editor):
    harvests = apps.get_model('prelievi', 'Harvest').objects.using(
        schema_editor.connection.alias,
    )
    if harvests.filter(harvest_plan_item__isnull=False).exclude(
        damaged=models.F('harvest_plan_item__damaged'),
        unhealthy=models.F('harvest_plan_item__unhealthy'),
        psr=models.F('harvest_plan_item__psr'),
    ).exists():
        raise IrreversibleError('Linked harvests have flags that differ from their worksite.')


class Migration(migrations.Migration):
    dependencies = [
        ('base', '0043_optional_parcel_area'),  # Last migration to rebuild these triggers.
        ('prelievi', '0004_historicalharvest_species_masses'),
    ]
    operations = [
        migrations.RunSQL(
            DROP_FLAGS, reverse_sql=previous.HARVEST_FLAGS_INSERT + previous.HARVEST_FLAGS_UPDATE,
        ),
        migrations.RunSQL(
            require_worksite_flags('insert', 'INSERT')
            + require_worksite_flags('update', 'UPDATE OF damaged, unhealthy, psr, harvest_plan_item_id'),
            reverse_sql=DROP_FLAGS,
        ),
        migrations.RunPython(migrations.RunPython.noop, check_rollback),
    ]
