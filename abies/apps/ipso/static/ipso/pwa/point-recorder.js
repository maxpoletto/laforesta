// Keep the most accurate fix in each short burst of GPS measurements.
'use strict';

const TRACE_WINDOW_MS = 1000;

function createPointRecorder({ append, newId, onError, onTraceSaved }) {
  let trace = null;
  let segment = null;
  let previousTime = 0;
  let windowStart = -Infinity;
  let best = null;
  let flushTimer = null;
  let entry = null;
  let saving = false;
  let writes = Promise.resolve();

  function enqueue(row) {
    const write = writes.catch(() => {}).then(() => append(row));
    writes = write;
    // stop() must see failures from writes still in flight. Later explicit
    // retries and notes can proceed once a failure has been reported.
    write.catch(() => {
      if (writes === write) writes = Promise.resolve();
    });
    return write;
  }

  function position(fix) {
    if (!fix || !Number.isFinite(fix.t) || Date.now() - fix.t >= GPS_STALE_MS ||
        !Number.isFinite(fix.lat) || !Number.isFinite(fix.lon) ||
        !Number.isFinite(fix.acc) || fix.acc < 0) return null;
    return { lat: fix.lat, lon: fix.lon, acc_m: fix.acc, taken_at: new Date(fix.t).toISOString() };
  }

  function finishEntry() {
    // A heartbeat can re-emit the last cached fix while the operator is
    // typing. Only a fix acquired after the note closes can resume the trace.
    previousTime = Math.max(previousTime, Date.now());
    entry = null;
  }

  function flushWindow() {
    if (!best) return writes;
    clearTimeout(flushTimer);
    flushTimer = null;
    const activeTrace = trace;
    const write = enqueue({ ...best, segment, note: '' });
    best = null;
    write.then(() => {
      // A slow IndexedDB commit can finish after another trace starts.
      if (trace !== activeTrace) return;
      trace.saved++;
      onTraceSaved?.();
    }, error => {
      if (trace === activeTrace) {
        interrupt();
        trace = null;
      }
      onError(error);
    });
    return write;
  }

  function interrupt() {
    const write = flushWindow();
    segment = null;
    windowStart = -Infinity;
    return write;
  }

  function onFix(fix) {
    const point = position(fix);
    if (!point) {
      interrupt();
      return;
    }
    if (entry && !entry.fix) entry.fix = point;
    if (!trace || entry || fix.t < windowStart) return;
    if (fix.t < windowStart + TRACE_WINDOW_MS) {
      // A running minimum needs one slot even if a burst has hundreds of fixes.
      // Keep the first on ties. A heartbeat cannot reopen a flushed window.
      if (best && point.acc_m < best.acc_m) best = point;
      previousTime = Math.max(previousTime, fix.t);
      return;
    }
    if (fix.t <= previousTime) return;
    flushWindow();
    if (!segment || fix.t - previousTime >= GPS_STALE_MS) segment = newId();
    windowStart = fix.t;
    previousTime = fix.t;
    best = point;
    flushTimer = setTimeout(flushWindow, TRACE_WINDOW_MS);
  }

  return {
    get tracing() { return trace !== null; },
    get savedFixes() { return trace?.saved ?? 0; },
    get editing() { return !!entry; },
    get canSave() { return !!entry?.fix && !saving; },
    get saving() { return saving; },
    onFix,
    toggle() {
      if (entry) return;
      interrupt();
      trace = trace ? null : { saved: 0 };
    },
    interrupt,
    begin(fix) {
      if (entry) return;
      interrupt();
      entry = { fix: position(fix) };
    },
    cancel() { if (!saving) finishEntry(); },
    async save(note) {
      if (!entry?.fix || saving) return;
      saving = true;
      try {
        await enqueue({ ...entry.fix, segment: null, note: note.trim() });
        finishEntry();
      } finally {
        saving = false;
      }
    },
    stop() {
      const write = interrupt();
      trace = null;
      return write;
    },
  };
}

if (typeof module !== 'undefined') module.exports = { createPointRecorder };
