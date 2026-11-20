import { canRemoveVertex, insertVertex, moveVertex, removeVertex, rings } from './drawing-parts.js';

const same = (a, b) => a[0] === b[0] && a[1] === b[1];

function markNeighbor(draft, feature) {
  if (feature.parcel_id !== draft.parcelId && !draft.neighbors.includes(feature.name)) {
    draft.neighbors.push(feature.name);
  }
}

export function detachBoundary(draft, source) {
  const owner = draft.features.find(feature => feature.parts.includes(source));
  if (owner) {
    const part = owner.parts.indexOf(source);
    draft.edits.push({id: owner.id, part, operation: 'detach'});
    owner.parts.splice(part, 1);
  } else {
    draft.added.splice(draft.added.indexOf(source), 1);
  }
}

export function boundaryVertexRemoval(draft, ringIndex, index) {
  const source = draft.parts[draft.active];
  if (!canRemoveVertex(source, ringIndex, index)) return null;
  const owner = draft.features.find(feature => feature.parts.includes(source));
  if (!owner) return {targets: [{part: source, ringIndex, index}]};

  const ring = rings(source)[ringIndex].slice(0, -1);
  const vertex = ring[index];
  const before = ring.at(index - 1);
  const after = ring[(index + 1) % ring.length];
  const targets = [];
  for (const feature of draft.features) {
    for (const part of feature.parts) {
      const all = rings(part);
      for (let r = 0; r < all.length; r++) {
        const open = all[r].slice(0, -1);
        const first = open.findIndex(point => same(point, vertex));
        if (first < 0) continue;
        const previous = open.at(first - 1);
        const following = open[(first + 1) % open.length];
        const shared = (same(previous, before) && same(following, after))
          || (same(previous, after) && same(following, before));
        if (!shared || !canRemoveVertex(part, r, first)
            || open.slice(first + 1).some(point => same(point, vertex))) return null;
        targets.push({feature, part, ringIndex: r, index: first});
      }
    }
  }
  return {owner, source, targets};
}

export function removeBoundaryVertex(draft, ringIndex, index, removal) {
  const {owner, source, targets} = removal;
  if (owner) {
    draft.edits.push({id: owner.id, part: owner.parts.indexOf(source),
      ring: ringIndex, index, operation: 'remove'});
  }
  for (const target of targets) {
    removeVertex(target.part, target.ringIndex, target.index);
    if (target.feature) markNeighbor(draft, target.feature);
  }
}

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
    if (changed) markNeighbor(draft, feature);
  }
}
