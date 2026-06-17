# Generated for the Ipso staged-upload integration.

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='IpsoUpload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_id', models.CharField(max_length=64, unique=True)),
                ('mode', models.CharField(max_length=32)),
                ('schema_version', models.IntegerField()),
                ('reference_version', models.CharField(blank=True, max_length=100)),
                ('work_package_id', models.CharField(blank=True, max_length=100)),
                ('operator', models.CharField(blank=True, max_length=100)),
                ('record_count', models.IntegerField(default=0)),
                ('checksum', models.CharField(max_length=64)),
                ('inbox_path', models.CharField(max_length=500)),
                ('state', models.CharField(choices=[('received', 'received'), ('imported', 'imported'), ('rejected', 'rejected'), ('conflict', 'conflict')], default='received', max_length=20)),
                ('received_at', models.DateTimeField(auto_now_add=True)),
                ('imported_at', models.DateTimeField(blank=True, null=True)),
                ('target_type', models.CharField(blank=True, max_length=50)),
                ('target_id', models.IntegerField(blank=True, null=True)),
                ('error_summary', models.TextField(blank=True)),
                ('imported_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='ipso_imports', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-received_at'],
            },
        ),
    ]
