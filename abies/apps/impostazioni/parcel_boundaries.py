"""Read and replay a versioned workspace for one parcel's boundary edits."""

import hashlib
import json
from copy import deepcopy

from shapely import STRtree, make_valid
from shapely.geometry import LineString, Point, Polygon, shape

from apps.base.models import MapFeature, Parcel
from apps.impostazioni.drawing_geometry import checked_parts, feature_geometry, planar_polygon, position
from config import strings as S


def parcel_outlines(region_id):
    return list(MapFeature.objects.filter(parcel__region_id=region_id, published=True)
                .select_related('parcel').order_by('pk'))


def parcel_used(parcel):
    # Do not orphan a grid, plan or other forestry record.
    return any(
        relation.related_model._base_manager.filter(**{relation.field.name: parcel}).exists()
        for relation in Parcel._meta.related_objects if relation.get_accessor_name() != 'map_features'
    )


def polygon_parts(geometry):
    polygons = geometry['coordinates']
    if geometry['type'] == 'Polygon':
        polygons = [polygons]
    return [{'type': 'Polygon', 'coordinates': polygon} for polygon in polygons]


def feature_rings(features):
    for feature in features:
        for part in feature['parts']:
            yield from part['coordinates']


def node_boundaries(features):
    """Expose vertices along coincident edges even if only one parcel recorded them."""
    rings = list(feature_rings(features))
    points = sorted({tuple(point) for ring in rings for point in ring})
    tree = STRtree([Point(point) for point in points])
    for ring in rings:
        noded = [ring[0]]
        for start, end in zip(ring, ring[1:]):
            if start != end:
                segment = LineString([start, end])
                endpoints = {tuple(start), tuple(end)}
                interior = [points[index] for index in tree.query(segment, predicate='covers')
                            if points[index] not in endpoints]
                interior.sort(key=lambda point: segment.project(Point(point)))
                noded.extend(map(list, interior))
            noded.append(end)
        ring[:] = noded


def boundary_workspace(parcel, outlines):
    signature = [parcel.version, [(row.pk, row.version, row.parcel.version) for row in outlines]]
    review = hashlib.sha256(json.dumps(signature, separators=(',', ':')).encode()).hexdigest()
    geometries = [planar_polygon(row.geometry) for row in outlines]
    shapes = [shape(geometry) for geometry in geometries]
    tree = STRtree(shapes)
    owned = [geom for row, geom in zip(outlines, shapes) if row.parcel_id == parcel.pk]
    if not owned:
        return {'review': review, 'features': []}
    # A bounding-box intersection covers neighbors at junctions and can also
    # include parcels whose surveys overlap by a small amount.
    neighbors = {outlines[i].parcel_id for i in tree.query(owned)[1]}
    features = [{'id': outline.pk, 'parcel_id': outline.parcel_id,
                 'name': outline.parcel.name, 'parts': polygon_parts(geometry)}
                for outline, geometry in zip(outlines, geometries) if outline.parcel_id in neighbors]
    node_boundaries(features)
    return {'review': review, 'features': features}


def remove_boundary_vertex(rings, original, index):
    if index >= len(original) - 1:
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
    source = original[:-1]
    vertex = source[index]
    adjacent = (source[index - 1], source[(index + 1) % len(source)])
    targets = []
    for ring in rings:
        open_ring = ring[:-1]
        matches = [i for i, point in enumerate(open_ring) if point == vertex]
        if not matches:
            continue
        i = matches[0]
        edges = (open_ring[i - 1], open_ring[(i + 1) % len(open_ring)])
        # At a junction, merging one pair of edges would break the shared border.
        if len(matches) != 1 or len(open_ring) <= 3 or edges not in (adjacent, adjacent[::-1]):
            raise ValueError(S.CARTOGRAPHY_VERTEX_REQUIRED)
        targets.append((ring, i))
    for ring, i in targets:
        del ring[i]
        ring[-1] = ring[0]


def apply_boundary_edit(rings, original, index, operation, point):
    end = original[index]
    if operation == 'move':
        for ring in rings:
            for i, coord in enumerate(ring):
                if coord == end:
                    ring[i] = point
    else:
        start = original[index - 1]
        for ring in rings:
            for i in range(len(ring) - 1, 0, -1):
                if (ring[i - 1], ring[i]) in ((start, end), (end, start)):
                    ring.insert(i, point)


def boundary_results(parcel, data, outlines):
    """Replay from the database snapshot so the shared side cannot be left behind."""
    state = boundary_workspace(parcel, outlines)
    if data.get('review') != state['review']:
        raise ValueError(S.CARTOGRAPHY_STALE)
    features = state['features']
    original_parts = {f['id']: deepcopy(f['parts']) for f in features}
    owned = {f['id']: f['parts'] for f in features if f['parcel_id'] == parcel.pk}
    edits = data.get('edits', [])
    if not isinstance(edits, list):
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
    rings = list(feature_rings(features))
    try:
        for edit in edits:
            operation = edit['operation']
            fid, part = edit['id'], edit['part']
            if any(type(value) is not int or value < 0 for value in (fid, part)):
                raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
            pieces = owned[fid]
            if operation == 'detach':
                del pieces[part]
                rings = list(feature_rings(features))
                continue
            ring_index, index = edit['ring'], edit['index']
            point = edit.get('point')
            identifiers = (ring_index, index)
            if (any(type(value) is not int or value < 0 for value in identifiers)
                    or operation not in ('move', 'insert', 'remove')
                    or (operation != 'remove' and not position(point))
                    or (operation == 'insert' and index == 0)):
                raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
            ring = pieces[part]['coordinates'][ring_index]
            if operation == 'remove':
                remove_boundary_vertex(rings, ring, index)
            else:
                apply_boundary_edit(rings, ring, index, operation, point)
    except (IndexError, KeyError, TypeError):
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID) from None
    updates = {f['id']: feature_geometry(checked_parts(f['parts']), 'Polygon') if f['parts'] else None
               for f in features
               if f['parts'] != original_parts[f['id']]}
    raw_added = data.get('added', [])
    if not isinstance(raw_added, list):
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
    added = [feature_geometry(checked_parts([part]), 'Polygon') for part in raw_added]
    if None in updates.values() and not added:
        has_outline = any(row.parcel_id == parcel.pk and updates.get(row.pk, row.geometry) is not None
                          for row in outlines)
        if not has_outline and parcel_used(parcel):
            raise ValueError(S.CARTOGRAPHY_PARCEL_OUTLINE_REQUIRED)
    validate_intersections(outlines, updates, added)
    return updates, added


def validate_intersections(outlines, updates, added):
    """Allow correction of legacy overlap, and reject any newly overlapping area."""
    before = [make_valid(shape(planar_polygon(row.geometry))) for row in outlines]
    # An empty geometry keeps indices aligned with the original snapshot.
    shapes = {pk: shape(geometry) if geometry is not None else Polygon()
              for pk, geometry in updates.items()}
    after = [shapes.get(row.pk, geometry) for row, geometry in zip(outlines, before)]
    after.extend(shape(geometry) for geometry in added)
    tree = STRtree(after)
    changed = [i for i, row in enumerate(outlines) if row.pk in updates]
    changed.extend(range(len(outlines), len(after)))
    for i in changed:
        for j in tree.query(after[i], predicate='intersects'):
            if j == i:
                continue
            overlap = after[i].intersection(after[j])
            if overlap.area and (i >= len(before) or j >= len(before)
                                 or overlap.difference(before[i].intersection(before[j])).area > 1e-14):
                raise ValueError(S.CARTOGRAPHY_BOUNDARY_OVERLAPS)


def save_boundaries(parcel, body):
    outlines = parcel_outlines(parcel.region_id)
    updates, added = boundary_results(parcel, body, outlines)
    touched = {parcel.pk} if added else set()
    for outline in outlines:
        if outline.pk in updates:
            touched.add(outline.parcel_id)
            geometry = updates[outline.pk]
            if geometry is None:
                outline.delete()
            else:
                outline.geometry = geometry
                outline.version += 1
                outline.save(update_fields=['geometry', 'version'])
    for geometry in added:
        MapFeature.objects.create(parcel=parcel, geometry=geometry, members={'type': 'Feature'})
    for item in Parcel.objects.filter(pk__in=touched):
        item.version += 1
        item.save(update_fields=['version'])
