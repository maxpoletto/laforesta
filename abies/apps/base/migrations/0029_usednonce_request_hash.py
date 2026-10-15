from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('base', '0028_refresh_coppice_stump_digest')]
    operations = [
        migrations.AddField(
            model_name='usednonce', name='request_hash',
            field=models.CharField(blank=True, default='', max_length=64),
        ),
    ]
