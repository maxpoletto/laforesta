export const vertexKey = ([lon, lat]) => `${lon},${lat}`;

export function boundaryTargetMode(draft) {
  return draft?.tool === 'Polygon' || (draft?.kind === 'parcel' && !draft.tool && !draft.join
    && draft.parts[draft.active]?.type === 'Polygon');
}

// One map target can belong to several parcel outlines at a junction.
export function referenceVertices(references, visible = () => true) {
  const targets = new Map();
  for (const {name, ring} of references) {
    for (let index = 0; index < ring.length - 1; index++) {
      const [lon, lat] = ring[index];
      if (!visible(lon, lat)) continue;
      const key = vertexKey(ring[index]);
      if (!targets.has(key)) targets.set(key, {point: [lon, lat], positions: []});
      targets.get(key).positions.push({name, ring, index});
    }
  }
  return targets;
}

function ringPath(ring, start, end, step) {
  const size = ring.length - 1;
  const path = [ring[start].slice(0, 2)];
  for (let i = start; i !== end;) {
    i = (i + step + size) % size;
    path.push(ring[i].slice(0, 2));
  }
  return path;
}

function distance(a, b) {
  // A local approximation is enough to compare runs within a parcel.
  const longitudeScale = Math.cos((a[1] + b[1]) * Math.PI / 360);
  return Math.hypot((a[0] - b[0]) * longitudeScale, a[1] - b[1]);
}

export function traceBetween(from, to, allowed = () => true) {
  if (!from || !to || from === to) return null;
  let best = null, shortest = Infinity;
  for (const start of from.positions) {
    for (const end of to.positions) {
      if (start.ring !== end.ring || start.index === end.index) continue;
      for (const step of [1, -1]) {
        const path = ringPath(start.ring, start.index, end.index, step);
        if (!allowed(path)) continue;
        let length = 0;
        for (let i = 1; i < path.length; i++) length += distance(path[i - 1], path[i]);
        if (length >= shortest) continue;
        shortest = length;
        best = path;
      }
    }
  }
  return best;
}

// Fit a run into an incident edge, preserving the rest of the edited ring.
export function traceInsertion(ring, anchor, path) {
  const size = ring.length - 1;
  const end = path.at(-1);
  const forward = {index: anchor + 1, endpoint: ring[(anchor + 1) % size], reverse: false};
  const back = {index: anchor || size, endpoint: ring[(anchor + size - 1) % size], reverse: true};
  const exact = [forward, back].find(edge => vertexKey(edge.endpoint) === vertexKey(end));
  const edge = exact ?? (distance(end, forward.endpoint) <= distance(end, back.endpoint) ? forward : back);
  const points = path.slice(1, exact ? -1 : undefined);
  const used = new Set(ring.map(vertexKey));
  if (points.some(point => used.has(vertexKey(point)))) return null;
  if (edge.reverse) points.reverse();
  return {index: edge.index, points};
}
