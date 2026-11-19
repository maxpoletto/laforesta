from importlib import import_module

from django.db import migrations, models


# SQLite rebuilds the parcel table. Its own triggers and the harvest triggers
# that query it must be removed and restored around the rebuild.
triggers = import_module('apps.base.migrations.0017_parcel_harvest_mechanism')


class Migration(migrations.Migration):
    dependencies = [('base', '0042_whole_walk_drawings')]
    operations = [
        migrations.RunSQL(triggers.TRIGGER_DROP, reverse_sql=triggers.TRIGGER_CREATE),
        migrations.AlterField('parcel', 'area_ha',
                              models.DecimalField(max_digits=7, decimal_places=2, null=True, blank=True)),
        migrations.AlterField('historicalparcel', 'area_ha',
                              models.DecimalField(max_digits=7, decimal_places=2, null=True, blank=True)),
        migrations.RunSQL(triggers.TRIGGER_CREATE, reverse_sql=triggers.TRIGGER_DROP),
    ]
