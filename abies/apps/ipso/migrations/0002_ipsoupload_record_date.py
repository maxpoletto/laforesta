import json
import re
from pathlib import Path

from django.db import migrations, models

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _payload_record_date(payload):
    records = payload.get('records', []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return ''
    dates = [
        row.get('date')
        for row in records
        if isinstance(row, dict)
        and isinstance(row.get('date'), str)
        and _DATE_RE.match(row.get('date'))
    ]
    return min(dates) if dates else ''


def backfill_record_date(apps, schema_editor):
    IpsoUpload = apps.get_model('ipso', 'IpsoUpload')
    updates = []
    for upload in IpsoUpload.objects.only('id', 'inbox_path', 'record_date'):
        if upload.record_date:
            continue
        try:
            payload = json.loads(
                (Path(upload.inbox_path) / 'upload.json').read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            continue
        record_date = _payload_record_date(payload)
        if record_date:
            upload.record_date = record_date
            updates.append(upload)
    if updates:
        IpsoUpload.objects.bulk_update(updates, ['record_date'])


class Migration(migrations.Migration):

    dependencies = [
        ('ipso', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='ipsoupload',
            name='record_date',
            field=models.CharField(blank=True, default='', max_length=10),
        ),
        migrations.RunPython(backfill_record_date, migrations.RunPython.noop),
    ]
