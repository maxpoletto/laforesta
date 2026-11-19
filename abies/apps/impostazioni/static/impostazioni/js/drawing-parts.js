// A walk keeps the recording order. Joins are deliberate and follow that order.
export function joinParts(parts, index) {
  const first = parts[index], next = parts[index + 1];
  const coordinates = first.coordinates.concat(next.coordinates);
  const fixes = [first, next].flatMap(part => part.fixes ?? part.coordinates.map(() => null));
  const end = first.coordinates.at(-1);
  if (end.every((v, i) => v === next.coordinates[0][i])) {
    coordinates.splice(first.coordinates.length, 1);
    fixes.splice(first.coordinates.length, 1);
  }
  parts.splice(index, 2, {type: 'LineString', coordinates, fixes});
}

export function closeLine(part) {
  const ring = part.coordinates.slice();
  const fixes = part.fixes?.slice();
  if (!ring[0].every((v, i) => v === ring.at(-1)[i])) {
    ring.push(ring[0].slice());
    fixes?.push(fixes[0]);
  }
  return {type: 'Polygon', coordinates: [ring], ...(fixes && {fixes})};
}

export function handPart(type, vertices) {
  if (!vertices.length) return null;
  if (vertices.length === 1) return {type: 'Point', coordinates: vertices[0]};
  const line = {type: 'LineString', coordinates: vertices};
  return type === 'Polygon' && vertices.length >= 3 ? closeLine(line) : line;
}

// The closing vertex of each ring shares its position with the first vertex.
export function rings(part) {
  if (part.type === 'Point') return [[part.coordinates]];
  return part.type === 'Polygon' ? part.coordinates : [part.coordinates];
}

export function moveVertex(part, ringIndex, i, coord) {
  if (part.type === 'Point') part.coordinates = coord;
  else {
    const ring = rings(part)[ringIndex];
    ring[i] = coord;
    if (part.type === 'Polygon' && i === 0) ring[ring.length - 1] = coord;
  }
}

export function insertVertex(part, ringIndex, i, coord) {
  const all = rings(part);
  const offset = all.slice(0, ringIndex).reduce((sum, ring) => sum + ring.length, 0);
  all[ringIndex].splice(i, 0, coord);
  part.fixes?.splice(offset + i, 0, null);
}
