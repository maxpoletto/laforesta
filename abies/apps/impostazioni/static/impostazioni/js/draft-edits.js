import { reduceVertices } from './drawing-parts.js';

const UNDO_LIMIT = 10;
export const REDUCTION_STEP_M = 5;

function remember(draft, undo) {
  const history = draft.history ??= [];
  history.push(undo);
  if (history.length > UNDO_LIMIT) history.shift();
}

export function addSketchPoint(draft, point) {
  return addSketchRun(draft, [point]);
}

export function addSketchRun(draft, points) {
  const last = draft.vertices.at(-1);
  const [point] = points;
  if (!point) return false;
  if (last?.[0] === point[0] && last[1] === point[1]) return false;
  const traceReady = draft.traceReady;
  for (const vertex of points) draft.vertices.push(vertex);
  const undo = current => {
    current.vertices.length -= points.length;
    current.traceReady = traceReady;
  };
  undo.sketch = true;
  remember(draft, undo);
  return true;
}

export function changeGeometry(draft, change) {
  const {parts, active, dirty, features, edits, added, neighbors, tool, vertices, traceReady, reduction} = draft;
  // Clone these together so a parcel's parts still refer to the same
  // objects as its boundary workspace and its pending additions.
  const snapshot = structuredClone({parts, active, dirty, features, edits, added, neighbors, tool, vertices, traceReady, reduction});
  if (change(draft) === false) return false;
  delete draft.reduction;
  remember(draft, current => Object.assign(current, snapshot));
  draft.dirty = true;
  return true;
}

export function hasActiveDrawingPath(draft) {
  return !!draft && !draft.kind && !draft.tool && !draft.returnSelection
    && ['LineString', 'Polygon'].includes(draft.parts[draft.active]?.type);
}

export function reduceActivePart(draft, distanceMeters) {
  if (!hasActiveDrawingPath(draft)) return false;
  const source = draft.reduction?.source ?? draft.parts[draft.active];
  const spacing = (draft.reduction?.spacing ?? 0) + REDUCTION_STEP_M;
  changeGeometry(draft, current => {
    current.parts[current.active] = reduceVertices(source, spacing, distanceMeters);
  });
  draft.reduction = {source, spacing};
  return true;
}

export function undoEdit(draft) {
  const undo = draft.history?.pop();
  if (!undo) return false;
  undo(draft);
  return true;
}

export function cancelSketch(draft) {
  while (draft.history?.at(-1)?.sketch) draft.history.pop();
  delete draft.tool;
  delete draft.vertices;
  delete draft.traceReady;
  delete draft.following;
}
