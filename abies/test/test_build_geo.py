"""Tests for building enriched GeoJSON artifacts."""

import json
from decimal import Decimal

from django.core.management import call_command

from apps.base.models import Parcel


def _feature(region, parcel):
    return {
        'type': 'Feature',
        'properties': {'layer': region, 'name': f'{region}-{parcel}'},
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
    }


def test_build_geo_enriches_terreni_from_imported_parcels(
        tmp_path, db, regions, eclasses):
    """terreni.geojson is enriched from DB parcel metadata only.

    In particular, particelle.geojson is not needed: the coppice flag comes from
    the imported Parcel/Eclass rows, after import_parcels has run.
    """
    Parcel.objects.create(
        name='1', region=regions[0], eclass=eclasses[0], area_ha=Decimal('1.0'),
    )
    Parcel.objects.create(
        name='2', region=regions[0], eclass=eclasses[2], area_ha=Decimal('1.0'),
    )
    src = tmp_path / 'src'
    out = tmp_path / 'out'
    src.mkdir()
    (src / 'terreni.geojson').write_text(json.dumps({
        'type': 'FeatureCollection',
        'features': [
            _feature(regions[0].name, '1'),
            _feature(regions[0].name, '2'),
            _feature(regions[1].name, 'missing'),
        ],
    }), encoding='utf-8')

    call_command('build_geo', src, output_dir=out)

    data = json.loads((out / 'terreni.geojson').read_text(encoding='utf-8'))
    assert data['features'][0]['properties']['coppice'] is False
    assert data['features'][1]['properties']['coppice'] is True
    assert 'coppice' not in data['features'][2]['properties']
    assert not (out / 'particelle.geojson').exists()
