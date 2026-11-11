"""Upgrade cached forecasts to the shared plan/calendar snapshot format."""
from django.db import migrations
from django.db.models import F


def mark_calendar_digest_stale(apps, schema_editor):
    alias = schema_editor.connection.alias
    statuses = apps.get_model('base', 'DigestStatus').objects.using(alias)
    updated = statuses.filter(name='future_production').update(
        stale=True, dirty_seq=F('dirty_seq') + 1,
    )
    # generate_all can leave an untracked file, even when there are no plans.
    # Existing deployments have plans or users; leave fresh installs untouched.
    if not updated and (
        apps.get_model('base', 'HarvestPlan').objects.using(alias).exists()
        or apps.get_model('base', 'User').objects.using(alias).exists()
    ):
        statuses.create(name='future_production', stale=True, dirty_seq=1)


class Migration(migrations.Migration):
    dependencies = [('base', '0032_species_equations')]
    operations = [migrations.RunPython(mark_calendar_digest_stale, migrations.RunPython.noop)]
