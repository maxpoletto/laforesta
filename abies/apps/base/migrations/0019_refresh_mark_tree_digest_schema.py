from django.db import migrations
from django.db.models import F


def mark_tree_digests_stale(apps, schema_editor):
    DigestStatus = apps.get_model('base', 'DigestStatus')
    TreeMark = apps.get_model('base', 'TreeMark')

    # Include empty digests left behind after the last mark was deleted.
    DigestStatus.objects.filter(name__startswith='mark_trees_').update(
        stale=True,
        dirty_seq=F('dirty_seq') + 1,
    )

    # generate_all() can create a digest file without creating its status row,
    # so also register every item that currently has marks.
    item_ids = (
        TreeMark.objects
        .values_list('harvest_plan_item_id', flat=True)
        .distinct()
    )
    for item_id in item_ids:
        name = f'mark_trees_{item_id}'
        DigestStatus.objects.get_or_create(
            name=name,
            defaults={'stale': True, 'dirty_seq': 1},
        )


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0018_observation_photo_metadata'),
    ]

    operations = [
        migrations.RunPython(mark_tree_digests_stale, migrations.RunPython.noop),
    ]
