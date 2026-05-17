// GPS-driven parcel auto-detection and per-tree sticky-override.
//
// Two pure factories, exported separately so each can be unit-tested:
//
//   createLocator(features) — point-in-polygon with hysteresis. The
//     caller passes a compresa-scoped feature subset whose elements
//     have already been bbox-indexed (via geo.buildBboxIndex). Each
//     GPS fix is fed via onFix({lat, lng, acc}); subscribers receive
//     the new feature (or null) whenever the committed parcel changes.
//     Commits require CONSECUTIVE_FIXES same-candidate fixes AND a fix
//     accuracy smaller than the point's distance to the relevant
//     polygon boundary.
//
//   createOverride() — sticky 'auto' | 'manual' choice across trees in
//     a session. setAuto/setManual transition; resolve(autoName)
//     returns the effective particella; isMismatch(autoName) is the
//     condition for the red-border warning state.
//
// Both factories are pure JS — no DOM, no GPS, no globals — so the
// state machines are fully testable from node.
'use strict';

// Resolve geo helpers from whichever container has them: in node tests
// `require('./geo.js')` yields the module.exports; in the browser
// geo.js is a sibling classic <script> whose top-level functions are
// already globals on `globalThis`. We do NOT destructure at the top
// level — top-level `const findContainingParcel = …` would collide
// with the global `function findContainingParcel(…)` from geo.js and
// the whole script would fail to parse. Dereference inside the
// functions instead.
const _geo = (typeof require !== 'undefined' && typeof module !== 'undefined')
  ? require('./geo.js')
  : globalThis;

// Hysteresis: same-candidate streak length required before a transition
// is considered. With the GPS callback firing every ~1–2 s under canopy
// (see ipso/CLAUDE.md GPS gotcha), three fixes is ~3–6 s of dwell.
const CONSECUTIVE_FIXES = 3;

function createLocator(features) {
  let committed = null;
  let candidate = null;
  let candidateCount = 0;
  const subscribers = [];

  function onFix(fix) {
    const cand = _geo.findContainingParcel(fix.lng, fix.lat, features);
    if (cand === committed) {
      candidate = null;
      candidateCount = 0;
      return;
    }
    if (cand === candidate) {
      candidateCount++;
    } else {
      candidate = cand;
      candidateCount = 1;
    }
    if (candidateCount < CONSECUTIVE_FIXES) return;

    // Distance to the boundary we are crossing: the new candidate's
    // boundary if we're entering one, otherwise the currently committed
    // parcel's (i.e. confirming we have left it).
    let boundaryDist;
    if (cand) {
      boundaryDist = _geo.distanceToBoundaryMeters(fix.lng, fix.lat, cand);
    } else if (committed) {
      boundaryDist = _geo.distanceToBoundaryMeters(fix.lng, fix.lat, committed);
    } else {
      boundaryDist = Infinity;
    }
    if (fix.acc >= boundaryDist) return;

    committed = cand;
    candidate = null;
    candidateCount = 0;
    for (const cb of subscribers) cb(committed);
  }

  return {
    onFix,
    subscribe(cb) { subscribers.push(cb); },
    getCommitted() { return committed; },
  };
}

function createOverride() {
  let mode = 'auto';
  let manual = '';
  return {
    setAuto() { mode = 'auto'; },
    setManual(name) { mode = 'manual'; manual = name; },
    getMode() { return mode; },
    getManual() { return manual; },
    resolve(autoName) {
      return mode === 'manual' ? manual : (autoName || '');
    },
    isMismatch(autoName) {
      return mode === 'manual' && manual !== (autoName || '');
    },
  };
}

if (typeof module !== 'undefined') {
  module.exports = { createLocator, createOverride, CONSECUTIVE_FIXES };
}
