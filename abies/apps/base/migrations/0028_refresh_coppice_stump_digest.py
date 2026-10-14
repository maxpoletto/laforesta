"""Rebuild parcel summaries using distinct stumps rather than shoot counts."""

from django.db import migrations
from django.db.models import F


def mark_parcel_stump_digest_stale(apps, schema_editor):
    DigestStatus = apps.get_model('base', 'DigestStatus')
    Parcel = apps.get_model('base', 'Parcel')
    alias = schema_editor.connection.alias
    statuses = DigestStatus.objects.using(alias)
    updated = statuses.filter(name='parcels').update(
        stale=True, dirty_seq=F('dirty_seq') + 1,
    )
    # generate_all() can leave a digest file without a tracked status. Register
    # it for existing parcels, but leave a fresh empty installation unchanged.
    if not updated and Parcel.objects.using(alias).exists():
        statuses.create(name='parcels', stale=True, dirty_seq=1)


class Migration(migrations.Migration):
    dependencies = [('base', '0027_other_species_volumes')]
    operations = [
        migrations.RunPython(mark_parcel_stump_digest_stale, migrations.RunPython.noop),
    ]
