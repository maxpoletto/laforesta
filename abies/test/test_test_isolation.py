"""Regression tests for pytest generated-data isolation."""


def test_generated_data_dirs_are_isolated_from_dev_instance(settings, tmp_path):
    data_dir = settings.BASE_DIR / 'data'
    assert settings.DIGEST_DIR != data_dir / 'digests'
    assert settings.GEO_DIR != data_dir / 'geo'
    assert settings.SATELLITE_DIR != data_dir / 'satellite'
    assert settings.DIGEST_DIR == tmp_path / 'digests'
    assert settings.GEO_DIR == tmp_path / 'geo'
    assert settings.SATELLITE_DIR == tmp_path / 'satellite'
