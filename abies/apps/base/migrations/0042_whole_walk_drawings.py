from django.db import migrations, models
import django.db.models.deletion


def collect_parts(apps, schema_editor):
    drawings = apps.get_model('base', 'MapDrawing').objects.using(schema_editor.connection.alias)
    # Existing development workspaces may have several rows for one walk.
    from itertools import groupby
    rows = list(drawings.order_by('source_id', 'sequence', 'pk'))
    for _, group in groupby(rows, lambda r: ('source', r.source_id) if r.source_id else ('drawing', r.pk)):
        first, *others = group
        parts = [{**r.geometry, 'fixes': r.fixes} for r in [first, *others]]
        drawings.filter(pk=first.pk).update(parts=parts)
        drawings.filter(pk__in=[r.pk for r in others]).delete()


class Migration(migrations.Migration):
    dependencies = [('base', '0041_region_asset_keys')]
    operations = [
        migrations.AddField('mapdrawing', 'parts', models.JSONField(default=list), preserve_default=False),
        migrations.RunPython(collect_parts, migrations.RunPython.noop),
        migrations.RemoveConstraint('mapdrawing', 'map_drawing_source_sequence'),
        migrations.RemoveField('mapdrawing', 'geometry'),
        migrations.RemoveField('mapdrawing', 'sequence'),
        migrations.RemoveField('mapdrawing', 'fixes'),
        migrations.AlterField('mapdrawing', 'source', models.OneToOneField(null=True,
            on_delete=django.db.models.deletion.SET_NULL, to='base.mappointset')),
    ]
