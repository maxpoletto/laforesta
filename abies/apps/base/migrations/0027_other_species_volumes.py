"""Fill previously unavailable catch-all estimates, preserving captured data."""

from decimal import Decimal, ROUND_HALF_UP

from django.db import migrations, models
from django.utils import timezone


def fill_other_species_volumes(apps, schema_editor):
    # Frozen names/coefficients: this migration must not follow later changes
    # to translations, equations, or runtime calculation helpers.
    coefficients = {
        'Altra conifera': (-1.8381, 0.037836, 0.39934),
        'Altra latifoglia': (0.81151, 0.038965),
    }
    alias = schema_editor.connection.alias
    TreeMark = apps.get_model('base', 'TreeMark')
    TreeSample = apps.get_model('base', 'TreeSample')
    Item = apps.get_model('base', 'HarvestPlanItem')
    History = apps.get_model('base', 'HistoricalHarvestPlanItem')
    DigestStatus = apps.get_model('base', 'DigestStatus')
    timestamp = timezone.now()
    affected_items = set()
    repaired = False

    for Model in (TreeMark, TreeSample):
        rows = Model.objects.using(alias).filter(
            tree__species__common_name__in=coefficients,
            tree__coppice=False,
            volume_m3__isnull=True, mass_q__isnull=True,
            d_cm__gt=0, h_m__gt=0,
        ).select_related('tree__species')
        for row in rows.iterator(chunk_size=500):
            b = coefficients[row.tree.species.common_name]
            d, h = float(row.d_cm), float(row.h_m)
            d2h = d * d * h
            v_dm3 = b[0] + b[1] * d2h
            if len(b) == 3:
                v_dm3 += b[2] * d
            row.volume_m3 = (Decimal(v_dm3) / Decimal(1000)).quantize(Decimal('0.0001'))
            row.mass_q = (row.volume_m3 * row.tree.species.density).quantize(
                Decimal('0.001'), rounding=ROUND_HALF_UP,
            )
            row.version += 1
            row.modified_at = timestamp
            row.save(using=alias, update_fields=[
                'volume_m3', 'mass_q', 'version', 'modified_at',
            ])
            repaired = True
            if Model is TreeMark:
                affected_items.add(row.harvest_plan_item_id)

    if not repaired:
        return

    fields = [field.attname for field in Item._meta.fields]
    for item in Item.objects.using(alias).filter(pk__in=affected_items):
        total = TreeMark.objects.using(alias).filter(
            harvest_plan_item_id=item.pk,
        ).aggregate(total=models.Sum('volume_m3'))['total']
        total = total.quantize(Decimal('0.001'))
        if item.volume_marked_m3 == total:
            continue
        item.volume_marked_m3 = total
        item.version += 1
        item.modified_at = timestamp
        item.save(using=alias, update_fields=['volume_marked_m3', 'version', 'modified_at'])
        # Establish the repaired value for future audit comparisons without
        # presenting maintenance as a user edit or altering state/dates.
        History.objects.using(alias).create(
            **{field: getattr(item, field) for field in fields},
            history_date=timestamp, history_type='~',
            history_change_reason='initial-audit-baseline',
        )

    # Tree volumes feed marks, samples, Bosco and future-production digests.
    DigestStatus.objects.using(alias).all().update(
        stale=True, dirty_seq=models.F('dirty_seq') + 1,
    )


class Migration(migrations.Migration):
    dependencies = [('base', '0026_species_botanical_type')]
    operations = [
        # Reversing must not discard derived values already used downstream.
        migrations.RunPython(fill_other_species_volumes, migrations.RunPython.noop),
    ]
