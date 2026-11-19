import { insertVertex, moveVertex, rings } from './drawing-parts.js';

const same = (a, b) => a[0] === b[0] && a[1] === b[1];

export function editBoundary(draft, operation, ringIndex, index, point) {
  const part = draft.parts[draft.active];
  const owner = draft.features.find(feature => feature.parts.includes(part));
  if (!owner) {
    // This piece will be created in its final state, so it needs no replay.
    const edit = operation === 'move' ? moveVertex : insertVertex;
    edit(part, ringIndex, index, point);
    return;
  }
  draft.edits.push({id: owner.id, part: owner.parts.indexOf(part), ring: ringIndex, index, operation, point});
  const ring = rings(part)[ringIndex];
  const end = ring[index];
  const start = ring[index - 1];
  for (const feature of draft.features) {
    let changed = false;
    for (const piece of feature.parts) {
      rings(piece).forEach((coords, r) => {
        if (operation === 'move') {
          // moveVertex also moves the closing copy of the first vertex.
          for (let i = 0; i < coords.length - 1; i++) {
            if (!same(coords[i], end)) continue;
            moveVertex(piece, r, i, point);
            changed = true;
          }
        } else {
          for (let i = coords.length - 1; i > 0; i--) {
            const a = coords[i - 1], b = coords[i];
            if (!(same(a, start) && same(b, end)) && !(same(a, end) && same(b, start))) continue;
            insertVertex(piece, r, i, point);
            changed = true;
          }
        }
      });
    }
    if (changed && feature.parcel_id !== draft.parcelId && !draft.neighbors.includes(feature.name)) {
      draft.neighbors.push(feature.name);
    }
  }
}
