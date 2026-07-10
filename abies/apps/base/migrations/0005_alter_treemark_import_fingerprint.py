from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0004_tree_mark_sample_number_uniqueness'),
    ]

    operations = [
        migrations.AlterField(
            model_name='treemark',
            name='import_fingerprint',
            field=models.CharField(blank=True, max_length=67, null=True),
        ),
    ]
