from __future__ import annotations

from io import StringIO

from django.core.management import call_command


def _call_check(phase: str) -> str:
    stdout = StringIO()
    call_command('check_release1_tree_observations', phase=phase, stdout=stdout)
    return stdout.getvalue()


def test_release1_preflight_noops_after_release2(db):
    assert 'Release 1 tree-observation preflight OK' in _call_check('pre')


def test_release1_postflight_noops_after_release2(db):
    assert 'Release 1 tree-observation postflight OK' in _call_check('post')
