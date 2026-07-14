import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'bin' / 'mirror-instance-state'


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
    )


def test_dry_run_defaults_to_prod_to_dev():
    proc = _run('--dry-run', '--now', '20260625T210000Z')

    assert proc.returncode == 0, proc.stderr
    assert 'Mirror Abies runtime state: prod -> dev' in proc.stdout
    assert 'abies-prod-20260625T210000Z-mirror-to-dev.tar.gz' in proc.stdout
    assert 'compose/prod.yml' in proc.stdout
    assert 'compose/dev.yml' in proc.stdout
    assert 'pre-mirror-from-prod' in proc.stdout
    assert 'down --remove-orphans' in proc.stdout
    assert 'manage.py migrate --noinput' in proc.stdout
    assert 'from apps.base.digests import generate_all; generate_all()' in proc.stdout
    assert 'up -d' in proc.stdout


def test_dry_run_can_skip_target_side_post_restore_steps():
    proc = _run(
        'prod', 'dev', '--dry-run', '--now', '20260625T210000Z',
        '--no-target-backup', '--no-migrate', '--no-digests', '--no-start',
        '--no-include-ipso-inbox',
    )

    assert proc.returncode == 0, proc.stderr
    assert '--no-include-ipso-inbox' in proc.stdout
    assert 'pre-mirror-from-prod' not in proc.stdout
    assert 'manage.py migrate --noinput' not in proc.stdout
    assert 'generate_all' not in proc.stdout
    assert 'up -d' not in proc.stdout


def test_refuses_prod_target_without_explicit_override():
    proc = _run('dev', 'prod', '--dry-run')

    assert proc.returncode != 0
    assert 'refusing to overwrite prod' in proc.stderr


def test_requires_confirmation_for_noninteractive_destructive_run():
    proc = _run('prod', 'dev', '--now', '20260625T210000Z')

    assert proc.returncode != 0
    assert 'destructive mirror requires --yes' in proc.stderr


def test_sanitizes_reason_for_archive_name():
    proc = _run(
        'prod', 'dev', '--dry-run', '--now', '20260625T210000Z',
        '--reason', 'copy prod now',
    )

    assert proc.returncode == 0, proc.stderr
    assert 'abies-prod-20260625T210000Z-copy-prod-now.tar.gz' in proc.stdout
