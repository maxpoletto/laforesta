# Generated for idempotent Prelievi CSV imports.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prelievi', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='harvest',
            name='import_fingerprint',
            field=models.CharField(blank=True, max_length=67, null=True, unique=True),
        ),
        migrations.AddField(
            model_name='historicalharvest',
            name='import_fingerprint',
            field=models.CharField(blank=True, db_index=True, max_length=67, null=True),
        ),
    ]
