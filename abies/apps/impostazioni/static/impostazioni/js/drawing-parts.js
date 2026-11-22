const samePosition = (a, b) => a[0] === b[0] && a[1] === b[1];

export function isEndpoint(part, index) {
  return part?.type === 'Point' ? index === 0
    : part?.type === 'LineString' && (index === 0 || index === part.coordinates.length - 1);
}

function canCloseLine(part) {
  const [first] = part.coordinates;
  const second = part.coordinates.find(point => !samePosition(point, first));
  return !!second && part.coordinates.some(point => !samePosition(point, first) && !samePosition(point, second));
}

export function canJoinEndpoints(parts, {close = true} = {}) {
  const open = parts.filter(part => isEndpoint(part, 0));
  return open.length > 1 || (close && open[0]?.type === 'LineString' && canCloseLine(open[0]));
}

export function joinEndpoints(parts, a, b, {close = true} = {}) {
  if (!isEndpoint(parts[a.part], a.index) || !isEndpoint(parts[b.part], b.index)) return null;
  if (a.part === b.part) {
    const part = parts[a.part];
    if (!close || a.index === b.index || !canCloseLine(part)) return null;
    parts[a.part] = closeLine(part);
    return a.part;
  }

  const start = Math.min(a.part, b.part), end = Math.max(a.part, b.part);
  // Keep both recording directions when an end meets a start. If one
  // segment must be reversed, preserve the direction of the earlier part.
  let [first, next] = a.part === start ? [a, b] : [b, a];
  const firstIsLine = parts[first.part].type === 'LineString';
  if ((firstIsLine && !first.index) || (!firstIsLine && next.index)) [first, next] = [next, first];
  const orient = (endpoint, reverse) => {
    const part = parts[endpoint.part];
    const [coordinates] = rings(part);
    const fixes = part.fixes ?? coordinates.map(() => null);
    return reverse ? {coordinates: coordinates.toReversed(), fixes: fixes.toReversed()}
      : {coordinates, fixes};
  };
  const left = orient(first, !first.index), right = orient(next, !!next.index);
  const coordinates = left.coordinates.concat(right.coordinates);
  const fixes = left.fixes.concat(right.fixes);
  if (samePosition(left.coordinates.at(-1), right.coordinates[0])) {
    const i = left.coordinates.length;
    if (parts[next.part].type === 'Point' && fixes[i]?.[1]) fixes[i - 1] = fixes[i];
    else fixes[i - 1] ??= fixes[i];
    coordinates.splice(i, 1);
    fixes.splice(i, 1);
  }
  parts.splice(end, 1);
  parts.splice(start, 1, coordinates.length === 1 ? {type: 'Point', coordinates: coordinates[0], fixes}
    : {type: 'LineString', coordinates, fixes});
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

export function reduceVertices(part, spacing, distanceMeters) {
  const closed = part.type === 'Polygon';
  const fixes = part.fixes && [];
  let offset = 0;
  const coordinates = rings(part).map(ring => {
    const last = ring.length - 1;
    let kept = [0];
    for (let i = 1; i < last; i++) {
      const [lon, lat] = ring[kept.at(-1)];
      const [nextLon, nextLat] = ring[i];
      if (distanceMeters(lat, lon, nextLat, nextLon) >= spacing) kept.push(i);
    }
    // Keep the end of a line and the closing coordinate of each ring.
    if (closed && new Set(kept.map(i => ring[i].join(','))).size < 3) {
      kept = ring.map((_, index) => index);
    } else {
      kept.push(last);
    }
    if (fixes) for (const index of kept) fixes.push(part.fixes[offset + index]);
    offset += ring.length;
    return kept.map(index => ring[index]);
  });
  return {...part, coordinates: closed ? coordinates : coordinates[0], ...(fixes && {fixes})};
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
  if (part?.type === 'Point') return ringIndex === 0 && index === 0;
  if (part?.type === 'LineString') return ringIndex === 0 && index >= 0 && index < part.coordinates.length;
  const ring = part?.type === 'Polygon' && part.coordinates[ringIndex];
  return !!ring && ring.length > 4 && index >= 0 && index < ring.length - 1;
}

export function removeVertex(part, ringIndex, index) {
  if (part.type === 'Point') return null;
  const all = rings(part);
  const ring = all[ringIndex];
  const offset = all.slice(0, ringIndex).reduce((sum, previous) => sum + previous.length, 0);
  ring.splice(index, 1);
  part.fixes?.splice(offset + index, 1);
  if (part.type === 'LineString' && ring.length === 1) {
    part.type = 'Point';
    part.coordinates = ring[0];
  } else if (part.type === 'Polygon' && index === 0) {
    ring[ring.length - 1] = ring[0];
    if (part.fixes) part.fixes[offset + ring.length - 1] = part.fixes[offset];
  }
  return part;
}

function splitLine(part, i) {
  return [[0, i + 1], [i, part.coordinates.length]].map(([start, end]) => ({
    type: 'LineString', coordinates: part.coordinates.slice(start, end),
    ...(part.fixes && {fixes: part.fixes.slice(start, end)}),
  }));
}

export function canBreakPart(part, ringIndex, index) {
  if (index < 0) return false;
  if (part?.type === 'LineString') return ringIndex === 0 && index > 0 && index < part.coordinates.length - 1;
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
