from django.db import migrations
from django.db.models import F


def mark_tree_position_digests_stale(apps, schema_editor):
    DigestStatus = apps.get_model('base', 'DigestStatus')
    TreeMark = apps.get_model('base', 'TreeMark')
    TreeSample = apps.get_model('base', 'TreeSample')

    # Include empty digests left behind after their final row was deleted.
    for prefix in ('mark_trees_', 'sampled_trees_'):
        DigestStatus.objects.filter(name__startswith=prefix).update(
            stale=True,
            dirty_seq=F('dirty_seq') + 1,
        )

    # generate_all() may have created files without DigestStatus rows. Register
    # every non-empty dynamic digest so it regenerates with the accuracy column.
    digest_ids = (
        ('mark_trees_', TreeMark.objects.values_list(
            'harvest_plan_item_id', flat=True,
        ).distinct()),
        ('sampled_trees_', TreeSample.objects.values_list(
            'sample__survey_id', flat=True,
        ).distinct()),
    )
    for prefix, ids in digest_ids:
        for object_id in ids:
            DigestStatus.objects.get_or_create(
                name=f'{prefix}{object_id}',
                defaults={'stale': True, 'dirty_seq': 1},
            )


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0020_alter_harvesttransition_unique_together'),
    ]

    operations = [
        migrations.RunPython(
            mark_tree_position_digests_stale,
            migrations.RunPython.noop,
        ),
    ]
