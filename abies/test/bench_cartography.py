"""Opt-in timings for a dense map and one long Ipso walk."""
import gzip
import json
import platform
import subprocess
import uuid
from statistics import median
from time import perf_counter

from apps.base.models import MapDrawing, MapFeature, MapPointSet, Parcel
from apps.impostazioni.drawing_geometry import walk_parts


GRID_SIDE = 12
EDGE_STEPS = 32
FIX_COUNT = 25_000
RUNS = 5
ROOT = '/api/impostazioni/cartography/'


def grid_ring(col, row):
    west, south = 16_000_000 + col * 3200, 38_000_000 + row * 3200
    corners = [(west, south), (west + 3200, south), (west + 3200, south + 3200),
               (west, south + 3200), (west, south)]
    ring = [[(x + (end_x - x) * step / EDGE_STEPS) / 1e6,
             (y + (end_y - y) * step / EDGE_STEPS) / 1e6]
            for (x, y), (end_x, end_y) in zip(corners, corners[1:])
            for step in range(EDGE_STEPS)]
    return ring + [ring[0]]


def report(label, timings, response):
    raw = response.content
    encoded = gzip.decompress(raw) if response.get('Content-Encoding') == 'gzip' else raw
    estimate = len(gzip.compress(encoded, compresslevel=6))
    print(f'{label}: median {median(timings) * 1000:.1f} ms, '
          f'{len(encoded):,} B JSON, {len(raw):,} B Django body, {estimate:,} B gzip estimate')


def run_request(request):
    start = perf_counter()
    response = request()
    elapsed = perf_counter() - start
    assert response.status_code == 200, response.content[:500]
    return elapsed, response


def measure_get(client, url, label):
    client.get(url, HTTP_ACCEPT_ENCODING='gzip')
    results = [run_request(lambda: client.get(url, HTTP_ACCEPT_ENCODING='gzip')) for _ in range(RUNS)]
    report(label, [elapsed for elapsed, _ in results], results[-1][1])
    return results[-1][1]


def measure_undo(parts):
    result = subprocess.run(
        ['node', '--expose-gc', '--input-type=module', '-e', """
import fs from 'node:fs';
import {changeGeometry} from './apps/impostazioni/static/impostazioni/js/draft-edits.js';
const draft = {parts: JSON.parse(fs.readFileSync(0, 'utf8')), active: 0};
global.gc();
const initial = process.memoryUsage().heapUsed;
const timings = [];
for (let i = 0; i < 10; i++) {
    const start = performance.now();
    changeGeometry(draft, current => { current.parts[0].coordinates[0] = [16 + i / 1e6, 38]; });
    timings.push(performance.now() - start);
}
timings.sort((a, b) => a - b);
global.gc();
const retained = process.memoryUsage().heapUsed - initial;
if (draft.history.length !== 10) throw Error('Undo history should hold ten edits');
console.log(`Undo (${process.version}): ${((timings[4] + timings[5]) / 2).toFixed(1)} ms median, ${retained} B extra heap`);
"""],
        input=json.dumps(parts), text=True, capture_output=True, check=True,
    )
    print(result.stdout.strip())


def test_large_compresa(client, writer_user, regions, eclasses):
    region, eclass = regions[0], eclasses[0]
    client.force_login(writer_user)
    cells = [(col, row) for row in range(GRID_SIDE) for col in range(GRID_SIDE)]
    parcels = Parcel.objects.bulk_create([
        Parcel(region=region, eclass=eclass, name=str(i + 1)) for i in range(len(cells))
    ])
    outlines = MapFeature.objects.bulk_create([
        MapFeature(parcel=parcel, geometry={'type': 'Polygon', 'coordinates': [grid_ring(col, row)]})
        for parcel, (col, row) in zip(parcels, cells)
    ])
    print(f'\n{platform.machine()} Python {platform.python_version()}; '
          f'{len(parcels)} parcels × {4 * EDGE_STEPS} vertices, {FIX_COUNT:,} GPS fixes')
    region_url = f'{ROOT}{region.pk}/'
    measure_get(client, region_url, 'Workspace GET without walk')

    # One long trace plus an individually recorded point.
    records = [{
        'client_record_id': i, 'segment': 'trace' if i < FIX_COUNT - 1 else None,
        'lon': 16 + i % 2000 / 1e5, 'lat': 38 + (i // 2000) / 1e4 + (i % 11) / 1e5,
        'acc_m': (25 + i % 36) / 10, 'note': '' if i < FIX_COUNT - 1 else 'Cancello',
    } for i in range(FIX_COUNT)]
    source = MapPointSet.objects.create(region=region, session_id='bench', name='Rilievo lungo',
                                        date='2026-11-18', operator='Operatore', records=records)
    drawing = MapDrawing.objects.create(region=region, source=source, name=source.name, parts=walk_parts(records))

    workspace = measure_get(client, region_url, 'Workspace GET with walk')
    measure_get(client, '/api/geo/map.geojson', 'Published map GET')

    middle = GRID_SIDE // 2
    index = middle * (GRID_SIDE + 1)
    parcel, outline = parcels[index], outlines[index]
    boundaries_url = f'{region_url}parcels/{parcel.pk}/boundaries/'
    measure_get(client, boundaries_url, 'Boundaries GET')
    timings = []
    original_x, y = grid_ring(middle, middle)[EDGE_STEPS + EDGE_STEPS // 2]
    for repeat in range(RUNS + 1):
        snapshot = client.get(boundaries_url).json()
        edited = next(row for row in snapshot['features'] if row['id'] == outline.pk)
        ring = edited['parts'][0]['coordinates'][0]
        vertex = next(i for i, point in enumerate(ring) if point[1] == y and abs(point[0] - original_x) < 0.00002)
        next_x = original_x + 0.00001 if repeat % 2 == 0 else original_x
        body = {'nonce': str(uuid.uuid4()), 'id': parcel.pk, 'review': snapshot['review'],
                'edits': [{'id': outline.pk, 'part': 0, 'ring': 0, 'index': vertex,
                           'point': [next_x, y], 'operation': 'move'}]}
        elapsed, response = run_request(lambda: client.post(
            f'{region_url}parcel-boundaries/', body, content_type='application/json'))
        if repeat:
            timings.append(elapsed)
    report('Boundary save POST', timings, response)

    parts = workspace.json()['drawings'][0]['parts']
    measure_undo(parts)
    timings = []
    for repeat in range(RUNS + 1):
        parts[0]['coordinates'][0][0] = 16 + repeat / 1e6
        body = json.dumps({'nonce': str(uuid.uuid4()), 'id': drawing.pk,
                           'version': drawing.version + repeat, 'parts': parts}, separators=(',', ':'))
        elapsed, response = run_request(lambda: client.post(
            f'{region_url}adjust/', body, content_type='application/json'))
        if repeat:
            timings.append(elapsed)
    report('Drawing save POST', timings, response)
    print(f'Drawing save request: {len(body.encode()):,} B JSON')
