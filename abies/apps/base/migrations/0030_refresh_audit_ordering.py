"""Regenerate audit digests with precise timestamps and stable event IDs."""

from django.db import migrations
from django.db.models import F


def mark_audit_stale(apps, schema_editor):
    alias = schema_editor.connection.alias
    statuses = apps.get_model('base', 'DigestStatus').objects.using(alias)
    updated = statuses.filter(name='audit').update(stale=True, dirty_seq=F('dirty_seq') + 1)
    # generate_all can publish an untracked digest. Existing deployments have
    # users; leave fresh databases untouched until their first audited write.
    if not updated and apps.get_model('base', 'User').objects.using(alias).exists():
        statuses.create(name='audit', stale=True, dirty_seq=1)


class Migration(migrations.Migration):
    dependencies = [('base', '0029_usednonce_request_hash')]
    operations = [migrations.RunPython(mark_audit_stale, migrations.RunPython.noop)]
