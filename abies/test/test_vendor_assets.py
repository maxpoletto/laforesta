"""A dependency update must refresh the actual files served to browsers."""

import runpy
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'leaflet/dist/leaflet.js': 'leaflet/leaflet.js',
    'leaflet/dist/leaflet.css': 'leaflet/leaflet.css',
    'leaflet/dist/images/marker-icon.png': 'leaflet/images/marker-icon.png',
    'chart.js/dist/chart.umd.js': 'chartjs/chart.umd.js',
    '@maxpoletto/sortable-table/src/sortable-table.js': 'sortable-table/sortable-table.js',
    '@maxpoletto/sortable-table/src/sortable-table.css': 'sortable-table/sortable-table.css',
}


@pytest.fixture
def vendor_tree(tmp_path):
    for source_name, target_name in ASSETS.items():
        for parent, name in [
            (tmp_path / 'vendor/node_modules', source_name),
            (tmp_path / 'apps/base/static/vendor', target_name),
        ]:
            target = parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_name.encode())
    return tmp_path


@pytest.fixture
def sync_assets():
    return runpy.run_path(str(PROJECT_ROOT / 'bin/update-vendor'))['sync_vendor_assets']


def test_matching_installed_packages_and_browser_assets_pass(sync_assets, vendor_tree):
    assert sync_assets(vendor_tree, check=True)


@pytest.mark.parametrize('missing', [False, True])
def test_check_detects_drift_without_rewriting_browser_assets(sync_assets, vendor_tree, missing):
    target = vendor_tree / 'apps/base/static/vendor/chartjs/chart.umd.js'
    if missing:
        target.unlink()
    else:
        target.write_bytes(b'previous release')
    assert not sync_assets(vendor_tree, check=True)
    if missing:
        assert not target.exists()
    else:
        assert target.read_bytes() == b'previous release'


def test_update_refreshes_changed_files_and_removes_obsolete_images(sync_assets, vendor_tree):
    target = vendor_tree / 'apps/base/static/vendor/leaflet/images/obsolete.png'
    target.write_bytes(b'old image')
    source = vendor_tree / 'vendor/node_modules/chart.js/dist/chart.umd.js'
    source.write_bytes(b'new chart release')
    assert not sync_assets(vendor_tree, check=True)
    assert sync_assets(vendor_tree)
    assert not target.exists()
    assert (vendor_tree / 'apps/base/static/vendor/chartjs/chart.umd.js').read_bytes() == b'new chart release'
    assert sync_assets(vendor_tree, check=True)


def test_missing_install_is_reported_without_deleting_committed_assets(sync_assets, vendor_tree):
    source = vendor_tree / 'vendor/node_modules/leaflet/dist/leaflet.js'
    source.unlink()
    with pytest.raises(FileNotFoundError):
        sync_assets(vendor_tree)
    assert (vendor_tree / 'apps/base/static/vendor/leaflet/leaflet.js').exists()
