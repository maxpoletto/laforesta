const UNDO_LIMIT = 10;

function remember(draft, undo) {
  const history = draft.history ??= [];
  history.push(undo);
  if (history.length > UNDO_LIMIT) history.shift();
}

function undoPoint(draft) {
  draft.vertices.pop();
}

export function addSketchPoint(draft, point) {
  const last = draft.vertices.at(-1);
  if (last?.[0] === point[0] && last[1] === point[1]) return false;
  draft.vertices.push(point);
  remember(draft, undoPoint);
  return true;
}

export function changeGeometry(draft, change) {
  const {parts, active, dirty, features, edits, added, neighbors, tool, vertices} = draft;
  // Clone these together so a parcel's parts still refer to the same
  // objects as its boundary workspace and its pending additions.
  const snapshot = structuredClone({parts, active, dirty, features, edits, added, neighbors, tool, vertices});
  if (change(draft) === false) return false;
  remember(draft, current => Object.assign(current, snapshot));
  draft.dirty = true;
  return true;
}

export function undoEdit(draft) {
  const undo = draft.history?.pop();
  if (!undo) return false;
  undo(draft);
  return true;
}

export function cancelSketch(draft) {
  while (draft.history?.at(-1) === undoPoint) draft.history.pop();
  delete draft.tool;
  delete draft.vertices;
}
