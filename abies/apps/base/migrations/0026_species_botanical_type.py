"""Split future catch-all choices without rewriting any existing tree FK."""

from django.db import migrations, models
from django.utils import timezone


def classify_species(apps, schema_editor):
    Species = apps.get_model('base', 'Species')
    History = apps.get_model('base', 'HistoricalSpecies')
    DigestStatus = apps.get_model('base', 'DigestStatus')
    alias = schema_editor.connection.alias
    species = Species.objects.using(alias)
    # Bootstrap requires empty reference tables on a fresh installation.
    if not species.exists():
        return
    conifer_genera = {
        'abies', 'picea', 'pseudotsuga', 'larix', 'pinus', 'taxus', 'cedrus',
        'cupressus', 'juniperus', 'thuja', 'sequoia', 'sequoiadendron',
        'cryptomeria', 'chamaecyparis', 'araucaria', 'podocarpus', 'tsuga',
    }
    broadleaf_genera = {
        'acer', 'alnus', 'betula', 'carpinus', 'castanea', 'fagus', 'quercus',
        'prunus', 'juglans', 'populus', 'robinia', 'sorbus', 'tilia', 'fraxinus',
        'salix', 'ulmus', 'olea', 'ilex',
    }
    conifer_names = {
        'Abete', 'Abete bianco', 'Abete Rosso', 'Douglas', 'Larice',
        'Pino Laricio', 'Pino Marittimo', 'Pino Nero', 'Pino Strobo', 'Tasso',
        'Altra conifera',
    }
    broadleaf_names = {
        'Castagno', 'Faggio', 'Acero', 'Betulla Bianca', 'Carpino', 'Cerro',
        'Farnia', 'Ciliegio', 'Leccio', 'Ontano', 'Noce', 'Pioppo Tremulo',
        'Quercia Roverella', 'Robinia', 'Sorbo', 'Tiglio', 'Altra latifoglia',
    }
    timestamp = timezone.now()
    fields = [field.attname for field in Species._meta.fields]

    def snapshot(obj, *, created=False):
        History.objects.using(alias).create(
            **{field: getattr(obj, field) for field in fields},
            history_date=timestamp, history_type='+' if created else '~',
            history_change_reason=None if created else 'initial-audit-baseline',
        )

    for obj in species.all():
        genus = obj.latin_name.strip().lower().split(' ', 1)[0]
        if obj.common_name == 'Altro':
            kind = ''
        elif genus in conifer_genera or obj.common_name in conifer_names:
            kind = 'conifer'
        elif genus in broadleaf_genera or obj.common_name in broadleaf_names:
            kind = 'broadleaf'
        else:
            # Do not guess the classification of custom, unrecognized taxa.
            kind = ''
        if kind != obj.botanical_type:
            obj.botanical_type = kind
            obj.version += 1
            obj.modified_at = timestamp
            obj.save(update_fields=['botanical_type', 'version', 'modified_at'])
            snapshot(obj)

    for name, kind, latin, names, density, order in (
        ('Altra conifera', 'conifer', 'Abies alba', ('Abete bianco', 'Abete'), '9.00', 997),
        ('Altra latifoglia', 'broadleaf', 'Fagus sylvatica', ('Faggio',), '10.50', 998),
    ):
        source = species.filter(
            models.Q(latin_name__iexact=latin) | models.Q(common_name__in=names),
        ).order_by('sort_order').first()
        obj, created = species.get_or_create(common_name=name, defaults={
            'botanical_type': kind,
            'density': source.density if source else density,
            'pressler_default': source.pressler_default if source else '2.00',
            'sort_order': order,
        })
        if created:
            snapshot(obj, created=True)
    legacy, created = species.get_or_create(common_name='Altro', defaults={
        'sort_order': 999, 'density': '9.00', 'pressler_default': '2.00',
    })
    if created:
        snapshot(legacy, created=True)
    # The species set changes the Prelievi schema and species palettes.
    DigestStatus.objects.using(alias).all().update(
        stale=True, dirty_seq=models.F('dirty_seq') + 1,
    )


class Migration(migrations.Migration):
    dependencies = [('base', '0025_historicalharvesttransition')]
    operations = [
        migrations.AddField(
            model_name='species', name='botanical_type',
            field=models.CharField(blank=True, default='', max_length=10,
                                   choices=[('conifer', 'Conifera'), ('broadleaf', 'Latifoglia')]),
        ),
        migrations.AddField(
            model_name='historicalspecies', name='botanical_type',
            field=models.CharField(blank=True, default='', max_length=10,
                                   choices=[('conifer', 'Conifera'), ('broadleaf', 'Latifoglia')]),
        ),
        # Added species may be referenced after deployment: never delete them
        # on reverse migration or rewrite their trees to a different species.
        migrations.RunPython(classify_species, migrations.RunPython.noop),
    ]
