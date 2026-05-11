// GPS manager. Wraps navigator.geolocation.watchPosition() with:
//  - persistent in-memory cache of the latest fix;
//  - staleness threshold (10 s -> dot turns red);
//  - subscriber callback for UI updates;
//  - self-healing watcher (heartbeat-driven restart) — Android's GPS
//    subsystem aggressively power-saves and watchPosition often goes
//    silent after a short idle without firing the error callback. The
//    fix is to clear-and-rewatch periodically whenever fixes stop
//    arriving.
//
// Start the watcher when entering `recording`; stop it on exit, to save
// battery while the operator is in pre_session or done screens.
'use strict';

// Accuracy thresholds calibrated for forest GPS. Open-sky phone GPS gives
// 3-10 m; under tree canopy 15-30 m is typical and a perfectly usable
// per-tree fix for marking. Anything > 50 m is "where in this stand are
// we" rather than "which tree", so we flag it red. The v0.1.1 values
// (10 / 25) were calibrated for open-sky and made the dot always-red
// under canopy even though the underlying fix was fine.
const GPS_GREEN_M = 20;
const GPS_YELLOW_M = 50;
const GPS_STALE_MS = 10000;
const GPS_RESTART_THRESHOLD_MS = 8000;  // restart if no fix for this long
const GPS_HEARTBEAT_MS = 2000;
const GPS_ERROR_BACKOFF_MS = 1000;

function createGps(onChange) {
  let watchId = null;
  let lastFix = null;       // { lat, lng, acc, t }
  let lastError = null;
  let restartCount = 0;
  let restartingAt = 0;     // ms timestamp; suppress duplicate restarts
  let heartbeatTimer = null;
  let visibilityHandler = null;
  let errorBackoffTimer = null;

  function emit() {
    onChange && onChange(state());
  }

  function state() {
    const fix = lastFix;
    if (!fix) {
      return { fix: null, tier: 'red', error: lastError, restarts: restartCount };
    }
    const age = Date.now() - fix.t;
    let tier;
    if (age > GPS_STALE_MS) tier = 'red';
    else if (fix.acc <= GPS_GREEN_M) tier = 'green';
    else if (fix.acc <= GPS_YELLOW_M) tier = 'yellow';
    else tier = 'red';
    return { fix, tier, error: null, age, restarts: restartCount };
  }

  function snapshot() {
    // Returns {lat, lng, acc_m} or null. Doesn't update internal state.
    if (!lastFix) return null;
    if (Date.now() - lastFix.t > GPS_STALE_MS) return null;
    return {
      lat: lastFix.lat,
      lng: lastFix.lng,
      acc_m: Math.round(lastFix.acc),
    };
  }

  function attachWatcher() {
    if (!navigator.geolocation) {
      lastError = 'unsupported';
      emit();
      return;
    }
    watchId = navigator.geolocation.watchPosition(
      (pos) => {
        lastFix = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          acc: pos.coords.accuracy,
          t: Date.now(),
        };
        lastError = null;
        restartingAt = 0;
        emit();
      },
      (err) => {
        if (err && err.code === err.PERMISSION_DENIED) {
          // Terminal: stop trying — restarting just thrashes the prompt.
          lastError = 'denied';
          stopWatcherOnly();
          emit();
          return;
        }
        // TIMEOUT or POSITION_UNAVAILABLE — schedule a restart.
        lastError = 'error';
        emit();
        if (!errorBackoffTimer) {
          errorBackoffTimer = setTimeout(() => {
            errorBackoffTimer = null;
            restart('error');
          }, GPS_ERROR_BACKOFF_MS);
        }
      },
      // maximumAge:0 forces the OS to provide a fresh hardware reading.
      // maximumAge:>0 lets some Androids skip polling, which is exactly
      // the watcher-goes-silent failure mode we're patching.
      { enableHighAccuracy: true, maximumAge: 0, timeout: 30000 }
    );
  }

  function stopWatcherOnly() {
    if (watchId != null) {
      try { navigator.geolocation.clearWatch(watchId); } catch (_) {}
      watchId = null;
    }
  }

  function restart(reason) {
    // Guard against piling up restarts within a short window.
    const now = Date.now();
    if (now - restartingAt < 1500) return;
    restartingAt = now;
    restartCount += 1;
    stopWatcherOnly();
    if (errorBackoffTimer) {
      clearTimeout(errorBackoffTimer);
      errorBackoffTimer = null;
    }
    attachWatcher();
  }

  function heartbeat() {
    // Two jobs: re-emit so the UI can downgrade the dot to red when the
    // last fix goes stale, AND restart the watcher if no fix has arrived
    // for long enough. The threshold (8 s) is tighter than the staleness
    // threshold (10 s) so the indicator never settles on red before we
    // try to recover.
    emit();
    const now = Date.now();
    const ageOrInfinity = lastFix ? now - lastFix.t : Infinity;
    if (ageOrInfinity > GPS_RESTART_THRESHOLD_MS) {
      restart('heartbeat');
    }
  }

  function onVisibility() {
    if (document.visibilityState === 'visible') {
      // Force a restart on resume — Android often pauses watchPosition
      // callbacks while the page is hidden, and the cleanest recovery
      // is a fresh registration.
      restart('visibility');
    }
  }

  function start() {
    if (watchId != null) return;
    attachWatcher();
    heartbeatTimer = setInterval(heartbeat, GPS_HEARTBEAT_MS);
    visibilityHandler = onVisibility;
    document.addEventListener('visibilitychange', visibilityHandler);
  }

  function stop() {
    stopWatcherOnly();
    if (heartbeatTimer != null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
    if (visibilityHandler) {
      document.removeEventListener('visibilitychange', visibilityHandler);
      visibilityHandler = null;
    }
    if (errorBackoffTimer) {
      clearTimeout(errorBackoffTimer);
      errorBackoffTimer = null;
    }
  }

  return { start, stop, state, snapshot };
}
