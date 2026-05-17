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

// geo.js's functions are reached as free identifiers: in the browser
// they are globals declared by the preceding classic <script>; in node
// tests the test harness puts them on globalThis before requiring this
// file (see test/tests.js).

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
    const cand = findContainingParcel(fix.lng, fix.lat, features);
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

    // We've seen the same candidate for CONSECUTIVE_FIXES in a row, so
    // `cand === candidate` here. Use the streak variable for clarity.
    // Distance to the boundary we are crossing: the new candidate's
    // boundary if we're entering one, otherwise the currently committed
    // parcel's (i.e. confirming we have left it).
    let boundaryDist;
    if (candidate) {
      boundaryDist = distanceToBoundaryMeters(fix.lng, fix.lat, candidate);
    } else if (committed) {
      boundaryDist = distanceToBoundaryMeters(fix.lng, fix.lat, committed);
    } else {
      boundaryDist = Infinity;
    }
    if (fix.acc >= boundaryDist) return;

    committed = candidate;
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
