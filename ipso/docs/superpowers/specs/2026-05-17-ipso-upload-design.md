# ipso → server upload: design

Status: draft (2026-05-17)

## Goal

Extend ipso so that at the end of a marking session, the operator's CSV
is uploaded to a server-side endpoint in addition to being saved
locally to the phone's Downloads folder.

Primary objectives, in priority order:

1. **No field-data loss.** Operators work in rough conditions (steep
   slopes, weather, sometimes hours from a road). A successful session
   that fails to deliver its data is unacceptable.
2. **Simplicity.** No new heavy dependencies. The server-side component
   is a small standalone Python service; abies is not on the path.
3. **Forward-compatible with abies.** The wire format and idempotency
   model survive into the future abies-hosted endpoint with no client
   change beyond a base-URL swap.

Out of scope: the eventual abies integration itself; any UI for
browsing uploaded files; per-operator authentication; per-tree
streaming upload; uploading the 20-tree backup CSVs.

## High-level shape

Two new components, plus targeted changes to ipso.

```
┌─────────────────────────┐                    ┌──────────────────────────────────────┐
│   ipso (PWA on phone)   │                    │   ipso VM (laforesta.it)             │
│                         │                    │                                      │
│  ┌─────────────────┐    │   POST /upload     │  ┌───────────────────────────────┐  │
│  │ upload.js       │────┼───────HTTPS────────┼──│ Apache (ipso vhost)           │  │
│  │ upload-config.js│    │  Bearer + UUID hdr │  │   ProxyPass /upload →         │  │
│  └─────────────────┘    │                    │  │     127.0.0.1:<port>          │  │
│         ▲               │                    │  └────────────┬──────────────────┘  │
│         │ state         │                    │               ▼                     │
│  ┌─────────────────┐    │                    │  ┌───────────────────────────────┐  │
│  │ screen-upload   │    │                    │  │ ipso-upload.service (systemd) │  │
│  │ (state machine) │    │                    │  │   single-file Python stdlib   │  │
│  └─────────────────┘    │                    │  │   HTTP server                 │  │
│         │               │                    │  └────────────┬──────────────────┘  │
│         ▼               │                    │               ▼                     │
│  ┌─────────────────┐    │                    │  /var/lib/ipso-upload/uploads/      │
│  │ store.js (idx5) │    │                    │    <session-uuid>.csv               │
│  │ + upload_status │    │                    │    <session-uuid>.meta.json         │
│  └─────────────────┘    │                    │                                     │
└─────────────────────────┘                    │  Office workstation:                │
                                               │    periodic rsync from uploads/     │
                                               └──────────────────────────────────────┘
```

## Wire format

The protocol is small and stable. It is designed so the eventual
abies-hosted endpoint can accept the same requests verbatim.

### Request

```
POST /upload HTTP/1.1
Host: ipso.laforesta.it
Authorization: Bearer <shared-token>
Content-Type: text/csv; charset=utf-8
Content-Length: <N>
X-Ipso-Session-Id: <uuid>
X-Ipso-Schema-Version: 5

<CSV bytes — exactly what ipso writes to disk, UTF-8 with BOM,
 `;` separator, comma decimal, CRLF; see ipso/README.md for the
 column list.>
```

### Responses

| Status | Body                                                              | Meaning                                              | Client action                                                       |
|--------|-------------------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------|
| 200    | `{"ok": true, "stored_as": "<uuid>.csv", "duplicate": false}`     | Stored.                                              | Mark `uploaded`. Continue to Done.                                  |
| 200    | `{"ok": true, "stored_as": "<uuid>.csv", "duplicate": true}`      | Server already had identical content (idempotent).   | Same as above.                                                      |
| 401    | `{"ok": false, "error": "auth"}`                                  | Token rejected.                                      | Hard error. Stop retries. Surface to operator with bail prompt.     |
| 409    | `{"ok": false, "error": "conflict"}`                              | Session UUID exists with different content.          | Hard error. Stop retries. Surface "contatta l'ufficio". Bail only.  |
| 413    | `{"ok": false, "error": "too_large"}`                             | Body exceeded `max_body_bytes`.                      | Hard error. Stop retries. Surface to operator.                      |
| 422    | `{"ok": false, "error": "invalid_csv", "detail": "..."}`          | Body failed sanity check.                            | Hard error. Stop retries. Surface to operator.                      |
| 429    | `{"ok": false, "error": "rate_limited"}`                          | Server rate limit.                                   | Soft error. Continue retry schedule.                                |
| 5xx    | `{"ok": false, "error": "server"}`                                | Server-side failure.                                 | Soft error. Continue retry schedule.                                |
| (none) | network failure / timeout                                         | DNS / TCP / TLS / read timeout.                      | Soft error. Continue retry schedule.                                |

The client classifies errors into two buckets:

- **Hard errors** (`401`, `409`, `413`, `422`): a bug or misconfiguration.
  Stop retrying. The operator can still bail to local-only and the
  CSV is safe on the phone.
- **Soft errors** (`429`, `5xx`, network): retry forever according to
  the backoff schedule until success or the operator bails.

## Server: `ipso-upload`

### Footprint

- Single Python 3 file (~150 LoC), stdlib only (`http.server`,
  `socketserver`, `hmac`, `json`, `os`, `time`, `logging`, `urllib`).
  No Flask, no FastAPI, no third-party deps.
- Runs as an unprivileged systemd service `ipso-upload.service`.
  Listens on `127.0.0.1:<port>` (port chosen at deploy time, e.g.
  `8765`).
- Source lives in a new repo path `ipso/upload-server/` (server,
  systemd unit template, README). Build/install lives alongside ipso.
- Provisioned on the VM by `../system/ansible/foresta.yml`: install
  the file, install the systemd unit, create the `ipso-upload` system
  user, create `/var/lib/ipso-upload/uploads/` (owned by
  `ipso-upload:ipso-upload`, mode `0750`), install
  `/etc/ipso-upload/config.json` (root-owned, mode `0600`, deployed
  from ansible vault), add a `ProxyPass`/`ProxyPassReverse` pair for
  `/upload → http://127.0.0.1:<port>/upload` to the ipso vhost, reload
  Apache. The shared `static-ssl.conf.j2` template
  (`ipso/CLAUDE.md` § Deployment) does not currently expose a hook
  for per-vhost extra directives. Two options, decision deferred to
  implementation: (a) add an optional `extra_directives` variable to
  the template (small change, also benefits future vhosts), or
  (b) clone the template into a `static-ssl-with-proxy.conf.j2`
  used only by the ipso vhost. Option (a) preferred.

### Config

`/etc/ipso-upload/config.json`:

```json
{
  "bind_host": "127.0.0.1",
  "bind_port": 8765,
  "token": "<random-32-byte-base64>",
  "upload_dir": "/var/lib/ipso-upload/uploads",
  "rate_limit_per_minute": 10,
  "expected_bom": "﻿",
  "expected_header_prefix": "Data;Compresa;Particella;Catastrofata;"
}
```

The same token value lives in the developer's
`ipso/secrets/upload_config.json` (see below). Rotation is a two-step:
deploy server config, then deploy ipso build.

**Body size is enforced by Apache,** not by the WSGI service. The
ansible-managed `static_body_limit` (currently 1 MB) on the ipso
vhost is the single source of truth. A 1 MB cap leaves ~10× headroom
over the largest plausible session (~1000 trees ≈ ~100 KB). The
WSGI service carries a matching hardcoded `MAX_BODY_BYTES = 1048576`
constant as defense-in-depth: if it ever receives `Content-Length`
greater than that (e.g. someone deploys without Apache in front), it
returns `413` before reading the body. The constant and the ansible
value must be kept in sync by convention; if they diverge it is a
deployment bug to fix in code.

### Request handling

1. Method must be `POST`, path must be exactly `/upload`. Else `404`
   or `405`.
2. `Authorization` header must be `Bearer <token>`, compared with
   `hmac.compare_digest`. Else `401`.
3. `Content-Length` must be present and ≤ `max_body_bytes`. Else
   `413`. (No chunked / streaming bodies.)
4. `X-Ipso-Session-Id` must be present, must match a UUID-shaped
   regex. Else `422`.
5. Per-IP rate-limit token bucket (in-process dict, no persistence).
   The service is bound to `127.0.0.1`, so the meaningful client IP
   is in the `X-Forwarded-For` header set by Apache. Use the
   left-most value of that header if present, else fall back to
   `REMOTE_ADDR` (defensive only — would always be `127.0.0.1` in
   the deployed path). Refill rate =
   `rate_limit_per_minute / 60` tokens/second, capacity =
   `rate_limit_per_minute`. If exhausted: `429`.
6. Read body fully. Sanity-check:
   - First bytes equal `expected_bom`.
   - Next bytes start with `expected_header_prefix`.
   - Else `422` with a short `detail`.
7. Compute destination `path = upload_dir + "/" + session_id + ".csv"`.
8. Try `os.open(path, O_WRONLY | O_CREAT | O_EXCL, 0o640)` to a
   `.part` sibling, write body, `fsync`, `close`, `os.rename` to final
   path. (Atomic on POSIX.)
9. If `O_EXCL` failed because the final path already exists: read
   the existing file's bytes, `hmac.compare_digest` against incoming
   body. Equal → `200` with `"duplicate": true`. Different → `409`.
10. Write sidecar `<session_id>.meta.json` next to the CSV, containing
    (parsed cheaply from the CSV header line + first data row, since
    every column the office cares about is already in the CSV):
    `{ "operatore": "...", "compresa": "...", "catastrofata": 0|1,
       "tree_count": N, "received_at": "<ISO8601>",
       "schema_version": "<X-Ipso-Schema-Version>",
       "remote_addr": "<client IP>" }`.
    A zero-tree session (which ipso currently allows on Termina) has
    no data row to parse — write `"operatore": ""`, `"compresa": ""`,
    `"catastrofata": null`, `"tree_count": 0`. The CSV header line is
    always present.
    Best-effort: failure to write the sidecar logs a warning but does
    not change the success response (the CSV is what matters).
11. Log a single structured line to journald with
    `session_id`, `bytes`, `duplicate`, `operatore`, `compresa`,
    `tree_count`, `client_ip`, `elapsed_ms`.
12. Respond `200` with the JSON described above.

### Failure modes the server must survive

- Disk full → CSV write fails → `5xx` to client → client retries.
- Process restart mid-write → `.part` file lingers but is not visible
  via the final filename; cleaned up by a tiny startup sweep that
  removes `*.part` older than 60 seconds.
- Same UUID, different content → `409` (signals a bug; never a normal
  flow).
- Same UUID, same content → `200 duplicate=true` (the retry-safe path).

### What the server does NOT do (v1)

- No deletes, no edits, no `GET` listing endpoint.
- No nonce table / DB. Filesystem `O_EXCL` is the idempotency
  mechanism.
- No outbound email, webhook, or push notification.
- No HTTPS termination (Apache does that).
- No multi-tenancy. Single shared token, single upload directory.

## Client: ipso PWA changes

### New / changed modules

- `src/upload.js` (new). Public surface:
  ```js
  // Throws AuthError | ConflictError | InvalidCsvError | TooLargeError
  // (hard) or NetworkError | RateLimitError | ServerError (soft).
  async function uploadSession({ sessionId, csvText, signal });
  ```
  Implementation: `fetch(`${UPLOAD_BASE}/upload`, { method: 'POST',
  headers: { ... }, body: csvText, signal })`. Maps response status →
  typed error or success object. ~60 LoC including JSDoc.
- `src/upload-config.js` (generated at build time, **never committed**).
  ```js
  const UPLOAD_BASE = "https://ipso.laforesta.it";
  const UPLOAD_TOKEN = "<base64>";
  ```
  Generated by `tools/build_upload_config.py` from
  `secrets/upload_config.json` (gitignored, lives only on the
  developer's laptop). The Makefile's `build` target runs this script
  and refuses to proceed if `secrets/upload_config.json` is missing —
  same pattern as `terreni.geojson`. The generated file is included
  in the SW precache list so it's available after install.
- `src/store.js` (changed). `SCHEMA_VERSION` bumps from `4` to `5`.
  Session row gains:
  - `upload_status`: `null | 'uploaded' | 'local_only'` — `null` means
    the operator never finished this session (status is `OPEN`) or it
    is an old row from before this feature; `'uploaded'` means the
    server has it; `'local_only'` means the operator explicitly bailed.
  - `uploaded_at`: ISO-string or `null`.
  Add `setSessionUploadStatus(db, id, uploadStatus)`.
  New session status constant `STATUS_PENDING_UPLOAD` between
  `STATUS_OPEN` and `STATUS_EXPORTED`.
  `listOpenSessions` is renamed conceptually to `listResumableSessions`
  and returns rows with `status IN (STATUS_OPEN, STATUS_PENDING_UPLOAD)`.
  (Backwards-compatible: rows with `status = STATUS_EXPORTED` and no
  `upload_status` field are treated as already-delivered-via-CSV; they
  do not appear in any modal.)
- `src/app.js` (changed). New screen state `screen-upload`. State
  machine described below.
- `src/strings.js` (changed). New constants:
  - `S.UPLOAD_TITLE` ("Caricamento in corso")
  - `S.UPLOAD_ATTEMPT(n)` ("Tentativo N")
  - `S.UPLOAD_BAIL` ("Annulla caricamento e salva solo sul telefono")
  - `S.UPLOAD_SUCCESS_TOAST` ("Caricamento completato")
  - `S.UPLOAD_LOCAL_ONLY_TOAST` ("Salvato solo sul telefono")
  - `S.UPLOAD_ERROR_AUTH` ("Errore di autenticazione. Contatta lo sviluppatore.")
  - `S.UPLOAD_ERROR_CONFLICT` ("La sessione risulta già caricata con contenuto diverso. Contatta l'ufficio.")
  - `S.UPLOAD_ERROR_INVALID` ("Il server ha rifiutato il file. Contatta lo sviluppatore.")
  - `S.UPLOAD_ERROR_TOO_LARGE` ("File troppo grande per il server. Contatta lo sviluppatore.")
  - `S.UPLOAD_RESUME_TITLE` ("Sessioni in attesa di caricamento")
  - `S.UPLOAD_RESUME_DO_NOW` ("Carica ora")
  - `S.UPLOAD_RESUME_KEEP_LOCAL` ("Mantieni solo locale")
  - `S.UPLOAD_DURATION_LABEL` ("Tempo trascorso")  ← for the in-progress display
- `src/index.html` (changed). New `<section id="screen-upload">` with
  title, status text, attempt counter, elapsed-time display, and bail
  button. Same `.screen` class pattern as the existing screens.
- `src/style.css` (changed). Styles for `screen-upload` — kept
  consistent with the existing screen styling.
- `src/sw.js` (changed). Precache list includes `upload-config.js`.
  Cache name automatically tracks the bumped `APP_VERSION` per
  existing pattern.
- `src/version.js` (changed). `APP_VERSION` bumped (the SW relies on
  this for cache rollover — see ipso/CLAUDE.md `Versioning`).
- `tools/build_upload_config.py` (new). ~30 LoC: read JSON, emit JS.
- `Makefile` (changed). `build` target depends on
  `build/upload-config.js`; new rule generates it from
  `secrets/upload_config.json`. `clean` removes it. `deploy` is
  unchanged in shape — same `rsync` of `build/`.

### Termina → upload state machine (`screen-upload`)

```
                                 ┌──────────────────────────┐
   Termina pressed ──────────────│ enterUpload(session)     │
                                 │   1. setSessionStatus    │
                                 │      (PENDING_UPLOAD)    │
                                 │   2. downloadFinal()     │  ← always, before any network
                                 │      (local CSV is the   │
                                 │       trust anchor)      │
                                 │   3. show screen-upload  │
                                 │   4. attempt = 1         │
                                 │   5. start spinner       │
                                 └────────────┬─────────────┘
                                              ▼
                                  ┌──────────────────────────┐
                                  │ tryUpload()              │
                                  │   await upload.upload()  │◀────────┐
                                  └────────┬─────────────────┘         │
                                           │                           │
                  ┌─────── success ────────┤                           │
                  ▼                        ▼                           │
       ┌──────────────────────┐  ┌────────────────────┐                │
       │ markUploaded()       │  │ classify error     │                │
       │   upload_status =    │  └──┬─────────┬───────┘                │
       │   'uploaded'         │     │         │                        │
       │   uploaded_at = now  │     │ hard    │ soft                   │
       │   status = EXPORTED  │     ▼         ▼                        │
       │   → screen-done      │  ┌────────┐ ┌────────────────┐         │
       └──────────────────────┘  │ stop   │ │ sleep(backoff) │         │
                                 │ show   │ │ attempt += 1   │         │
                                 │ error  │ │ → tryUpload()  │─────────┘
                                 │ text   │ └────────────────┘
                                 │ enable │
                                 │ bail   │
                                 └────────┘
                                       ▲
                                       │ (bail also available throughout)
                                       │
                                 ┌──────────────────────┐
                                 │ Bail tapped:         │
                                 │   abort retry        │
                                 │   markLocalOnly()    │
                                 │     upload_status =  │
                                 │     'local_only'     │
                                 │     status = EXPORTED│
                                 │     → screen-done    │
                                 └──────────────────────┘
```

**Backoff schedule.** `[2, 4, 8, 16, 30, 30, 30, ...]` seconds.
Capped at 30 s. Pure JS (`setTimeout`) — no Background Sync API
(unsupported on iOS PWA, not worth the complexity).

**AbortController.** Each `tryUpload()` call uses an
`AbortController`; the bail handler aborts the in-flight request so
the operator isn't forced to wait out a 30 s server hang.

**`downloadFinal()` is called on every entry to `screen-upload`,
not just on the Termina path.** That means a session reached via the
resume modal's "Carica ora" action triggers a fresh local CSV
download before the upload spinner starts. Mobile browsers
(Android Chrome, Firefox) auto-rename duplicate filenames
(`... (1).csv`, `... (2).csv`) rather than overwriting, so the
operator never loses an earlier copy. Rationale: aligns with the
"no field data loss" priority — if the original CSV was deleted
from Downloads (operator cleanup, OS storage management, factory
reset since), the operator still ends up with a copy. The cost on
the happy path is one extra "... (1).csv" sitting in Downloads,
which is trivial.

**Bail-button availability.** The bail button is mounted at t=0 and
stays interactive throughout: during retries, while sleeping between
retries, and during the terminal hard-error state. It is the only
escape out of a stopped retry loop. The "enable bail" wording in the
diagram is shorthand for "this state is the one where bail is the
only sensible action" — not a state transition that toggles the
button's enabled property.

**Visibility.** While `screen-upload` is mounted, the wake lock is
re-acquired (same `setupWakeLockVisibility` machinery). GPS is stopped
(it was stopped already by Termina).

### Resume-on-open modal extension

The existing `showResumeModal` already handles `STATUS_OPEN` sessions.
Extend it to also list `STATUS_PENDING_UPLOAD` sessions with a
different action set:

- For `STATUS_OPEN`: existing actions — `[Riprendi] [Esporta CSV] [Scarta]`.
- For `STATUS_PENDING_UPLOAD`: new actions — `[Carica ora] [Mantieni solo locale]`.
  - `Carica ora`: enter `screen-upload` for that session.
  - `Mantieni solo locale`: set `upload_status = 'local_only'`,
    `status = STATUS_EXPORTED`, remove from list.

The modal title becomes "Sessioni da gestire" (generic) if both kinds
are present. List entries are visually identical (date · compresa ·
operator · tree_count) but the second line of actions differs.

### Build / deploy

- `secrets/upload_config.json` lives only on the developer's laptop
  and is `.gitignore`d.
- `make build` regenerates `build/upload-config.js`. Refuses to
  proceed if the secrets file is missing — same posture as the
  existing reference data.
- `make deploy` is unchanged in shape — it `rsync`s `build/`. The
  ipso vhost serves `upload-config.js` like any other static file.
- The shared token in `secrets/upload_config.json` must match the
  one in `/etc/ipso-upload/config.json` on the VM. Rotation
  procedure is documented in `ipso/upload-server/README.md`: update
  server config + restart `ipso-upload.service`, then `make deploy`
  on ipso. There is a brief window (seconds) where in-flight uploads
  from the old token may `401`; the retry loop handles this by
  failing hard (operator bails to local-only; office knows to expect
  manual CSV for that window).

## Testing

### Server

- `ipso/upload-server/test_server.py` runs the server against a
  temp `upload_dir` on `127.0.0.1` with `ephemeral` port, makes HTTP
  requests via `urllib`, and asserts:
  - Happy path: 200, file exists, content matches, meta sidecar
    written, journald line emitted.
  - Idempotent retry: same UUID + same body → 200 duplicate=true,
    file unchanged.
  - Conflict: same UUID + different body → 409, file unchanged.
  - Bad token → 401.
  - Missing token → 401.
  - Wrong content type → 422 (validation-shaped error, not 415).
  - Missing `X-Ipso-Session-Id` → 422.
  - Malformed UUID → 422.
  - Body exceeds `max_body_bytes` → 413.
  - Body missing BOM → 422.
  - Body missing header line → 422.
  - Rate limit triggers `429` after `N+1` requests within a minute.
  - Concurrent identical uploads (two threads, same UUID, same body)
    → both return 200 (one stores, one sees duplicate).
  - Concurrent conflicting uploads → one 200, one 409.

### Client

- Extend `ipso/test/tests.js` with `upload.js` mocked-fetch tests:
  - Happy path: returns `{ok: true, duplicate: false}`.
  - 200 duplicate=true: returns `{ok: true, duplicate: true}`.
  - 401 → throws `AuthError`.
  - 409 → throws `ConflictError`.
  - 413 → throws `TooLargeError`.
  - 422 → throws `InvalidCsvError`.
  - 429 → throws `RateLimitError`.
  - 5xx → throws `ServerError`.
  - Network failure (`fetch` rejects) → throws `NetworkError`.
  - Sends the right headers (Authorization, Content-Type, session
    UUID, schema version) and body bytes.
- Backoff schedule (pure function, easy to test):
  - First six attempts produce `[2, 4, 8, 16, 30, 30]` second waits.
- Store v5 round-trip:
  - Sessions with `upload_status = null` survive a read/write cycle.
  - `setSessionUploadStatus` round-trip.
  - `listResumableSessions` returns OPEN and PENDING_UPLOAD, excludes
    EXPORTED with `upload_status = 'uploaded'` or `'local_only'`.

### Manual smoke

After deploy, on a real phone:

1. Run a short session (3–4 trees), Termina, observe upload modal,
   observe success transition.
2. Repeat with Wi-Fi disabled, observe retries (backoff increases),
   bail, observe local-only Done screen, re-enable Wi-Fi, reopen app,
   observe resume modal, tap Carica ora, observe success.
3. Repeat with malformed token in `upload-config.js` (auth hard
   error), observe error text, bail.
4. Force-kill the app mid-upload retry loop, reopen, observe resume
   modal lists the session, tap Carica ora, observe success.

## Migration to abies

When abies's `martellata` / `rilevamento` import is built:

1. Add a Django view (probably `apps/ingest/views.py`) accepting the
   same `POST /upload` shape with the same headers and same
   responses. The view bypasses session auth (it is machine-to-
   machine) and uses the same shared-token model — or moves to
   per-user secrets as a separate change.
2. Switch the request idempotency mechanism from filesystem `O_EXCL`
   to a `uploaded_sessions(session_id PRIMARY KEY, ...)` table; map
   the same status codes.
3. Parse the CSV into `tree_mark` / `tree_sample` rows inside the
   request and run them through the normal Django write path
   (mark_stale → digest invalidation).
4. Change `UPLOAD_BASE` in `secrets/upload_config.json`, redeploy
   ipso. Phones pick up the new SW on next full close.
5. Decommission `ipso-upload.service` and remove the ProxyPass.
6. The `<session-uuid>.csv` files in `/var/lib/ipso-upload/uploads/`
   remain as historical archive (or get imported into abies as a
   one-shot ETL).

The wire format does not change. The retry / bail / resume client
behaviour does not change. Phones that have not been updated
(unlikely but possible) continue to hit the standalone endpoint until
its DNS / ProxyPass is removed.

## Out of scope / explicit non-goals (v1)

- No per-operator token. Operator identity comes from the CSV only.
- No per-operator UI for the token (no Impostazioni screen).
- No upload of the 20-tree backup CSVs — they remain local-only.
- No Background Sync API / push notifications.
- No server-side admin UI / file browser.
- No outbound email or webhooks.
- No metrics / dashboard.

## Risks

- **Token leakage.** The shared token ships in the public PWA bundle;
  anyone who can `curl https://ipso.laforesta.it/upload-config.js`
  can read it. Mitigations: HTTPS-only, server rate limit, body size
  cap, BOM/header sanity check. If abuse is observed, the fallback
  plan is to gate the PWA itself behind HTTP basic auth or move to
  an operator-entered token (see "Future" below).
- **Disk fill on the server.** A misbehaving client (or attacker who
  obtained the token) could try to fill `/var/lib/ipso-upload/`.
  Mitigations: `max_body_bytes` per request, per-IP rate limit, and
  ops should add a nagios/alertmanager-style disk-free alert on the
  VM as a separate change.
- **Schema rev requires data wipe.** `SCHEMA_VERSION = 5` is
  forward-compatible at the IndexedDB level (`onupgradeneeded` only
  creates stores; no migration runs), but per ipso/CLAUDE.md the
  pre-launch posture is "wipe site data by hand". Existing
  pre-feature sessions on operators' phones will see no upload UI
  and will require manual CSV delivery as before.
- **Two phones, same session UUID.** Cosmically improbable
  (`crypto.randomUUID()`), but the 409 path correctly handles it
  without data loss.

## Future (not in v1)

- Per-operator token, entered on the phone via a new Impostazioni
  screen. Server config grows to a map of token → operator. Server
  also cross-checks the CSV's Operatore column against the
  token-derived operator. This is the natural step if shared-token
  abuse becomes a real problem before abies launches.
- Upload of backup CSVs at the same `/upload` endpoint, with a
  different idempotency key (`<session-uuid>:backup:<seq>`). Would
  let the office see in-progress sessions without waiting for
  Termina. Cost: more chatter, more server state.
- Background retries via the Background Sync API on Android (where
  supported). Would let an operator close the app while a session is
  pending upload and have it eventually succeed without their
  attention. Not portable to iOS.
