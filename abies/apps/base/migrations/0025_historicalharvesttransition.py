import django.db.models.deletion
import simple_history.models
from django.conf import settings
from django.db import migrations, models
from django.utils import timezone


def snapshot_existing_transitions(apps, schema_editor):
    """Keep existing before values without inventing historical user writes."""
    Transition = apps.get_model('base', 'HarvestTransition')
    History = apps.get_model('base', 'HistoricalHarvestTransition')
    DigestStatus = apps.get_model('base', 'DigestStatus')
    alias = schema_editor.connection.alias
    timestamp = timezone.now()
    snapshots = []
    fields = ('id', 'harvest_plan_item_id', 'open', 'date', 'note')
    for row in Transition.objects.using(alias).values(*fields).iterator(chunk_size=500):
        snapshots.append(History(
            **row, history_date=timestamp, history_type='+',
            # Mirrored by digests._AUDIT_BASELINE_REASON; these snapshots
            # seed comparisons but do not appear as Controllo events.
            history_change_reason='initial-audit-baseline',
        ))
        if len(snapshots) == 500:
            History.objects.using(alias).bulk_create(snapshots)
            snapshots = []
    if snapshots:
        History.objects.using(alias).bulk_create(snapshots)
    updated = DigestStatus.objects.using(alias).filter(name='audit').update(
        stale=True, dirty_seq=models.F('dirty_seq') + 1,
    )
    if not updated:
        DigestStatus.objects.using(alias).get_or_create(
            name='audit', defaults={'stale': True, 'dirty_seq': 1},
        )


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0024_alter_treemark_h_m_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='HistoricalHarvestTransition',
            fields=[
                ('id', models.BigIntegerField(auto_created=True, blank=True, db_index=True, verbose_name='ID')),
                ('open', models.BooleanField(help_text='True = Apri cantiere, False = Chiudi cantiere.')),
                ('date', models.DateField()),
                ('note', models.CharField(blank=True, help_text='Typically the regional permit number.', max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('harvest_plan_item', models.ForeignKey(blank=True, db_constraint=False, null=True, on_delete=django.db.models.deletion.DO_NOTHING, related_name='+', to='base.harvestplanitem')),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical transizione cantiere',
                'verbose_name_plural': 'historical transizioni cantiere',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.RunPython(snapshot_existing_transitions, migrations.RunPython.noop),
    ]
