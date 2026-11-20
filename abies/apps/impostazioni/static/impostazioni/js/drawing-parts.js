const samePosition = (a, b) => a[0] === b[0] && a[1] === b[1];

export function isEndpoint(part, index) {
  return part?.type === 'LineString' && (index === 0 || index === part.coordinates.length - 1);
}

export function joinEndpoints(parts, a, b, {close = true} = {}) {
  if (!isEndpoint(parts[a.part], a.index) || !isEndpoint(parts[b.part], b.index)) return null;
  if (a.part === b.part) {
    const part = parts[a.part];
    if (!close || a.index === b.index || new Set(part.coordinates.map(point => point.join(','))).size < 3) return null;
    parts[a.part] = closeLine(part);
    return a.part;
  }

  const start = Math.min(a.part, b.part), end = Math.max(a.part, b.part);
  // Keep both recording directions when an end meets a start. If one
  // segment must be reversed, preserve the direction of the earlier part.
  let [first, next] = a.part === start ? [a, b] : [b, a];
  if (!first.index) [first, next] = [next, first];
  const orient = (endpoint, reverse) => {
    const part = parts[endpoint.part];
    const fixes = part.fixes ?? part.coordinates.map(() => null);
    return reverse ? {coordinates: part.coordinates.toReversed(), fixes: fixes.toReversed()}
      : {coordinates: part.coordinates, fixes};
  };
  const left = orient(first, !first.index), right = orient(next, !!next.index);
  const coordinates = left.coordinates.concat(right.coordinates);
  const fixes = left.fixes.concat(right.fixes);
  if (samePosition(left.coordinates.at(-1), right.coordinates[0])) {
    const i = left.coordinates.length;
    fixes[i - 1] ??= fixes[i];
    coordinates.splice(i, 1);
    fixes.splice(i, 1);
  }
  parts.splice(end, 1);
  parts.splice(start, 1, {type: 'LineString', coordinates, fixes});
  return start;
}

export function closeLine(part) {
  const ring = part.coordinates.slice();
  const fixes = part.fixes?.slice();
  if (!samePosition(ring[0], ring.at(-1))) {
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

export function canRemoveVertex(part, ringIndex, index) {
  const ring = part?.type === 'Polygon' && part.coordinates[ringIndex];
  return !!ring && ring.length > 4 && index >= 0 && index < ring.length - 1;
}

export function removeVertex(part, ringIndex, index) {
  const all = rings(part);
  const ring = all[ringIndex];
  const offset = all.slice(0, ringIndex).reduce((sum, previous) => sum + previous.length, 0);
  ring.splice(index, 1);
  part.fixes?.splice(offset + index, 1);
  if (part.type === 'Polygon' && index === 0) {
    ring[ring.length - 1] = ring[0];
    if (part.fixes) part.fixes[offset + ring.length - 1] = part.fixes[offset];
  }
}

function splitLine(part, i) {
  return [[0, i + 1], [i, part.coordinates.length]].map(([start, end]) => ({
    type: 'LineString', coordinates: part.coordinates.slice(start, end),
    ...(part.fixes && {fixes: part.fixes.slice(start, end)}),
  }));
}

export function canBreakPart(part, ringIndex, index) {
  if (index < 0) return false;
  if (part?.type === 'LineString') return index > 0 && index < part.coordinates.length - 1;
  return part?.type === 'Polygon' && index < (part.coordinates[ringIndex]?.length ?? 0) - 1;
}

export function breakPart(part, ringIndex, index) {
  if (!canBreakPart(part, ringIndex, index)) return null;
  if (part.type === 'LineString') return splitLine(part, index);
  let offset = 0;
  return part.coordinates.map((ring, r) => {
    const fixes = part.fixes?.slice(offset, offset + ring.length);
    offset += ring.length;
    const cut = values => r === ringIndex ? values.slice(index + 1, -1).concat(values.slice(0, index)) : values;
    return {type: 'LineString', coordinates: cut(ring), ...(fixes && {fixes: cut(fixes)})};
  });
}
