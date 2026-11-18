"""Refresh cached Controllo rows with the shared Ordine label."""
from importlib import import_module

from django.db import migrations


mark_audit_stale = import_module('apps.base.migrations.0030_refresh_audit_ordering').mark_audit_stale


class Migration(migrations.Migration):
    dependencies = [('base', '0036_economic_class_editor')]
    operations = [migrations.RunPython(mark_audit_stale, mark_audit_stale)]
