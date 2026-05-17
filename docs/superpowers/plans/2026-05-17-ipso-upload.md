# ipso → server upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Termina e carica" flow to ipso that uploads the session CSV to a small server-side endpoint, with reliable retries and a bail-out path. Local CSV download remains the trust anchor.

**Architecture:** Two new pieces. (1) `ipso-upload`: a single-file Python stdlib HTTP server on the ipso VM, bound to `127.0.0.1`, fronted by Apache `ProxyPass`. Stores CSVs keyed by session UUID with O_EXCL idempotency. (2) Client-side ipso changes: new `upload.js` module, a build-time-generated `upload-config.js`, a new `screen-upload` state machine, and a `SCHEMA_VERSION` bump that adds `upload_status` to session rows. Resume modal extended to surface `PENDING_UPLOAD` sessions.

**Spec:** `docs/superpowers/specs/2026-05-17-ipso-upload-design.md` — read this before starting; the plan below is the build sequence, not the rationale.

**Tech Stack:** Python 3 stdlib (`http.server`, `socketserver`, `hmac`, `json`); vanilla JS ES6+ in the browser (no framework, no bundler); IndexedDB; `unittest` for server tests; ipso's existing `node test/tests.js` for client pure-logic tests; systemd for service management; Apache for HTTPS termination + `ProxyPass`.

**Test discipline:** Server logic is TDD'd against a localhost-bound server in a temp dir. Client pure-logic (backoff, error classification, status-set helpers) is TDD'd in `test/tests.js`. DOM-heavy code (`screen-upload` wiring) is verified by manual smoke test on a real phone after deploy — call this out, don't pretend it's covered by unit tests.

**Commit cadence:** One commit per task. Use Conventional Commits prefix (`feat`, `test`, `refactor`, `chore`, `docs`).

---

## File map

**New files**

- `ipso/upload-server/server.py` — the stdlib HTTP server.
- `ipso/upload-server/test_server.py` — `unittest`-based tests.
- `ipso/upload-server/config.example.json` — committed reference config.
- `ipso/upload-server/ipso-upload.service` — systemd unit template.
- `ipso/upload-server/README.md` — operator-facing notes (deploy, rotate token, logs).
- `ipso/upload-server/Makefile` — `test`, `run` (for local dev), `clean`.
- `ipso/tools/build_upload_config.py` — read `ipso/secrets/upload_config.json` → emit `ipso/build/upload-config.js`.
- `ipso/secrets/upload_config.json.example` — committed reference; actual file is gitignored.
- `ipso/src/upload.js` — client-side upload module (typed errors, backoff helper, `uploadSession`).

**Modified files**

- `ipso/Makefile` — wire `build_upload_config.py` into `build`; new `test` runs both node tests and `upload-server` tests.
- `ipso/.gitignore` — add `secrets/`.
- `ipso/src/store.js` — `SCHEMA_VERSION = 5`, new `STATUS_PENDING_UPLOAD`, `upload_status`/`uploaded_at` fields, `setSessionUploadStatus`, rename `listOpenSessions` → `listResumableSessions`, new pure helper `isResumableStatus`.
- `ipso/src/strings.js` — new `S.UPLOAD_*` and `S.UPLOAD_RESUME_*` strings.
- `ipso/src/index.html` — new `<section id="screen-upload">`; load `upload.js` and `upload-config.js`.
- `ipso/src/style.css` — styling for `screen-upload`.
- `ipso/src/app.js` — `enterUpload()` and state machine; `showResumeModal` extended for `PENDING_UPLOAD`.
- `ipso/src/sw.js` — add `upload.js` and `upload-config.js` to `SHELL`.
- `ipso/src/version.js` — bump `APP_VERSION` to `0.5.0`.
- `ipso/test/tests.js` — new test blocks for `upload.js` (mocked fetch) and the new `store.js` pure helper.
- `ipso/CLAUDE.md` — extend the "Storage" section for `upload_status`; new top-level "Upload" subsection covering the wire format and the resume flow.
- `ipso/README.md` — short user-facing paragraph about the new upload behavior + bail option.

---

## Build order (high-level)

1. **Server first** (Tasks 1–9). Independently testable; lets us prove the wire format before touching the client.
2. **Client build plumbing** (Tasks 10–12). `secrets/` directory, `tools/build_upload_config.py`, Makefile wiring.
3. **Client schema bump + pure helpers** (Tasks 13–15). `store.js` changes; testable bits in `tests.js`.
4. **Client `upload.js`** (Tasks 16–18). Pure-logic module with mocked-fetch tests.
5. **Client UI + state machine** (Tasks 19–23). `screen-upload`, `app.js` wiring, `index.html`, `style.css`, `strings.js`.
6. **Service worker + version + docs** (Tasks 24–26).
7. **Manual end-to-end smoke** (Task 27).

---

## Task 1: Server — happy-path test for `POST /upload`

**Files:**
- Create: `ipso/upload-server/test_server.py`
- Create: `ipso/upload-server/Makefile`
- Create: `ipso/upload-server/server.py` (empty stub so import resolves)

- [ ] **Step 1: Write the failing test**

`ipso/upload-server/test_server.py`:

```python
"""Tests for ipso-upload. Run with: make test (in this directory)."""

import http.client
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest

import server


VALID_TOKEN = "test-token-do-not-use-in-prod"

# Minimal CSV with one row, matching ipso's wire format.
BOM = "﻿"
CSV_HEADER = (
    "Data;Compresa;Particella;Catastrofata;Numero;Specie;"
    "D_cm;H_m;H_measured;Lat;Lng;Acc_m;Operatore"
)
CSV_ROW = "11/05/2026;Serra;1;0;;Abete;42;24;0;38,425310;16,120440;7;Mario Rossi"
SAMPLE_CSV = BOM + CSV_HEADER + "\r\n" + CSV_ROW + "\r\n"

VALID_HEADERS = {
    "Authorization": f"Bearer {VALID_TOKEN}",
    "Content-Type": "text/csv; charset=utf-8",
    "X-Ipso-Session-Id": "11111111-2222-3333-4444-555555555555",
    "X-Ipso-Schema-Version": "5",
}


def _pick_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ServerHarness:
    """Spin up server in a thread; stop on close()."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="ipso-upload-test-")
        self.port = _pick_port()
        cfg = {
            "bind_host": "127.0.0.1",
            "bind_port": self.port,
            "token": VALID_TOKEN,
            "upload_dir": os.path.join(self.tmp, "uploads"),
            "rate_limit_per_minute": 1000,
            "expected_bom": BOM,
            "expected_header_prefix": "Data;Compresa;Particella;Catastrofata;",
        }
        os.makedirs(cfg["upload_dir"])
        self._cfg_path = os.path.join(self.tmp, "config.json")
        with open(self._cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        self.cfg = cfg

        self._httpd = server.make_server(cfg)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()
        # Tight poll for ready
        for _ in range(50):
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                c.request("OPTIONS", "/upload")
                c.getresponse().read()
                c.close()
                break
            except ConnectionRefusedError:
                time.sleep(0.02)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        body_bytes = body.encode("utf-8") if isinstance(body, str) else body
        h = dict(headers or {})
        if body_bytes is not None and "Content-Length" not in h:
            h["Content-Length"] = str(len(body_bytes))
        conn.request(method, path, body=body_bytes, headers=h)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, dict(resp.getheaders()), data

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)
        shutil.rmtree(self.tmp, ignore_errors=True)


class HappyPathTest(unittest.TestCase):
    def setUp(self):
        self.h = ServerHarness()

    def tearDown(self):
        self.h.close()

    def test_post_creates_file_and_returns_200(self):
        status, headers, body = self.h.request(
            "POST", "/upload", body=SAMPLE_CSV, headers=VALID_HEADERS
        )
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["duplicate"])
        self.assertEqual(
            payload["stored_as"],
            "11111111-2222-3333-4444-555555555555.csv",
        )
        path = os.path.join(
            self.h.cfg["upload_dir"],
            "11111111-2222-3333-4444-555555555555.csv",
        )
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as fh:
            self.assertEqual(fh.read().decode("utf-8"), SAMPLE_CSV)


if __name__ == "__main__":
    unittest.main()
```

`ipso/upload-server/Makefile`:

```make
.PHONY: test run clean

test:
	python3 -m unittest -v test_server.py

# Run against config.example.json for local hand-testing (uploads to /tmp).
run:
	python3 server.py config.example.json

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
```

`ipso/upload-server/server.py`:

```python
"""ipso-upload — placeholder until Task 2."""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ipso/upload-server && make test
```

Expected: FAIL — `AttributeError: module 'server' has no attribute 'make_server'`.

- [ ] **Step 3: Implement `make_server` and the bare happy-path POST**

Replace `ipso/upload-server/server.py` with:

```python
"""ipso-upload — receive CSVs from ipso and store them on disk.

Single-file stdlib HTTP server. Listens on 127.0.0.1; Apache fronts it
with HTTPS + ProxyPass. See ipso/upload-server/README.md for the
deploy story; docs/superpowers/specs/2026-05-17-ipso-upload-design.md
for the spec.
"""

import errno
import hmac
import http
import http.server
import json
import logging
import os
import re
import socketserver
import sys
import time

# Mirrors Apache's ansible-managed static_body_limit. Defense-in-depth:
# if the WSGI service is ever exposed without Apache in front, requests
# above this size fail closed.
MAX_BODY_BYTES = 1024 * 1024

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

log = logging.getLogger("ipso-upload")


def _json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(cfg):
    token_bytes = cfg["token"].encode("utf-8")
    upload_dir = cfg["upload_dir"]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.info("%s - %s", self.address_string(), fmt % args)

        def do_POST(self):
            if self.path != "/upload":
                _json_response(self, 404, {"ok": False, "error": "not_found"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

            session_id = self.headers.get("X-Ipso-Session-Id", "")
            dest = os.path.join(upload_dir, session_id + ".csv")

            tmp = dest + ".part"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(body)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.rename(tmp, dest)
            except FileExistsError:
                os.unlink(tmp)
                raise

            _json_response(self, 200, {
                "ok": True,
                "stored_as": session_id + ".csv",
                "duplicate": False,
            })

    return Handler


def make_server(cfg):
    addr = (cfg["bind_host"], cfg["bind_port"])
    return socketserver.ThreadingTCPServer(addr, make_handler(cfg))


def main(argv):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) != 2:
        print("usage: server.py <config.json>", file=sys.stderr)
        return 2
    with open(argv[1], "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    os.makedirs(cfg["upload_dir"], exist_ok=True)
    srv = make_server(cfg)
    log.info("listening on %s:%d", cfg["bind_host"], cfg["bind_port"])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ipso/upload-server && make test
```

Expected: PASS — `Ran 1 test in <1s OK`.

- [ ] **Step 5: Commit**

```bash
cd /home/maxp/src/laforesta
git add ipso/upload-server/server.py ipso/upload-server/test_server.py ipso/upload-server/Makefile
git commit -m "feat(ipso-upload): scaffold server with happy-path test"
```

---

## Task 2: Server — auth check (401 paths)

**Files:**
- Modify: `ipso/upload-server/test_server.py`
- Modify: `ipso/upload-server/server.py`

- [ ] **Step 1: Add failing tests**

Append to `test_server.py`:

```python
class AuthTest(unittest.TestCase):
    def setUp(self):
        self.h = ServerHarness()

    def tearDown(self):
        self.h.close()

    def test_missing_authorization_returns_401(self):
        h = {k: v for k, v in VALID_HEADERS.items() if k != "Authorization"}
        status, _, body = self.h.request("POST", "/upload", SAMPLE_CSV, h)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"ok": False, "error": "auth"})

    def test_bad_token_returns_401(self):
        h = dict(VALID_HEADERS, **{"Authorization": "Bearer wrong"})
        status, _, body = self.h.request("POST", "/upload", SAMPLE_CSV, h)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"ok": False, "error": "auth"})

    def test_malformed_authorization_returns_401(self):
        h = dict(VALID_HEADERS, **{"Authorization": "Basic abc"})
        status, _, body = self.h.request("POST", "/upload", SAMPLE_CSV, h)
        self.assertEqual(status, 401)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ipso/upload-server && make test
```

Expected: 3 new failures.

- [ ] **Step 3: Implement auth in `do_POST`**

Add at the top of `do_POST` (before reading the body), in `server.py`:

```python
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                _json_response(self, 401, {"ok": False, "error": "auth"})
                return
            presented = auth[len("Bearer "):].encode("utf-8")
            if not hmac.compare_digest(presented, token_bytes):
                _json_response(self, 401, {"ok": False, "error": "auth"})
                return
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS (4 tests now).

- [ ] **Step 5: Commit**

```bash
git add ipso/upload-server/server.py ipso/upload-server/test_server.py
git commit -m "feat(ipso-upload): enforce bearer-token auth"
```

---

## Task 3: Server — session-id and body validation (422 paths)

**Files:**
- Modify: `ipso/upload-server/test_server.py`
- Modify: `ipso/upload-server/server.py`

- [ ] **Step 1: Add failing tests**

Append to `test_server.py`:

```python
class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.h = ServerHarness()

    def tearDown(self):
        self.h.close()

    def test_missing_session_id_returns_422(self):
        h = {k: v for k, v in VALID_HEADERS.items() if k != "X-Ipso-Session-Id"}
        status, _, body = self.h.request("POST", "/upload", SAMPLE_CSV, h)
        self.assertEqual(status, 422)
        payload = json.loads(body)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "invalid_csv")

    def test_malformed_session_id_returns_422(self):
        h = dict(VALID_HEADERS, **{"X-Ipso-Session-Id": "not-a-uuid"})
        status, _, body = self.h.request("POST", "/upload", SAMPLE_CSV, h)
        self.assertEqual(status, 422)

    def test_wrong_content_type_returns_422(self):
        h = dict(VALID_HEADERS, **{"Content-Type": "application/json"})
        status, _, body = self.h.request("POST", "/upload", SAMPLE_CSV, h)
        self.assertEqual(status, 422)

    def test_body_without_bom_returns_422(self):
        no_bom = SAMPLE_CSV[1:]  # strip BOM
        status, _, body = self.h.request("POST", "/upload", no_bom, VALID_HEADERS)
        self.assertEqual(status, 422)

    def test_body_without_expected_header_returns_422(self):
        bad = BOM + "wrong;header;line\r\n"
        status, _, body = self.h.request("POST", "/upload", bad, VALID_HEADERS)
        self.assertEqual(status, 422)

    def test_wrong_method_returns_405(self):
        status, _, body = self.h.request("GET", "/upload", None, VALID_HEADERS)
        self.assertEqual(status, 405)

    def test_wrong_path_returns_404(self):
        status, _, body = self.h.request("POST", "/other", SAMPLE_CSV, VALID_HEADERS)
        self.assertEqual(status, 404)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: 7 new failures.

- [ ] **Step 3: Implement validation**

Replace `do_POST` in `server.py` (this is a substantial rewrite around the same skeleton):

```python
        def do_GET(self):
            _json_response(self, 405, {"ok": False, "error": "method_not_allowed"})

        def do_POST(self):
            if self.path != "/upload":
                _json_response(self, 404, {"ok": False, "error": "not_found"})
                return

            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                _json_response(self, 401, {"ok": False, "error": "auth"})
                return
            presented = auth[len("Bearer "):].encode("utf-8")
            if not hmac.compare_digest(presented, token_bytes):
                _json_response(self, 401, {"ok": False, "error": "auth"})
                return

            ctype = self.headers.get("Content-Type", "")
            if not ctype.lower().startswith("text/csv"):
                _json_response(self, 422, {
                    "ok": False, "error": "invalid_csv",
                    "detail": "wrong content type",
                })
                return

            session_id = self.headers.get("X-Ipso-Session-Id", "")
            if not UUID_RE.match(session_id):
                _json_response(self, 422, {
                    "ok": False, "error": "invalid_csv",
                    "detail": "missing or malformed session id",
                })
                return

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                _json_response(self, 422, {
                    "ok": False, "error": "invalid_csv",
                    "detail": "body is not UTF-8",
                })
                return

            if not text.startswith(cfg["expected_bom"]):
                _json_response(self, 422, {
                    "ok": False, "error": "invalid_csv",
                    "detail": "missing BOM",
                })
                return
            after_bom = text[len(cfg["expected_bom"]):]
            if not after_bom.startswith(cfg["expected_header_prefix"]):
                _json_response(self, 422, {
                    "ok": False, "error": "invalid_csv",
                    "detail": "unexpected header",
                })
                return

            dest = os.path.join(upload_dir, session_id + ".csv")
            tmp = dest + ".part"
            try:
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            except FileExistsError:
                # Concurrent in-flight write to the same uuid — treat as a
                # transient error so the client retries.
                _json_response(self, 503, {"ok": False, "error": "server"})
                return
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.rename(tmp, dest)

            _json_response(self, 200, {
                "ok": True,
                "stored_as": session_id + ".csv",
                "duplicate": False,
            })

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS (11 tests now).

- [ ] **Step 5: Commit**

```bash
git add ipso/upload-server/server.py ipso/upload-server/test_server.py
git commit -m "feat(ipso-upload): validate session id, content type, BOM, header"
```

---

## Task 4: Server — body size limit (413)

**Files:**
- Modify: `ipso/upload-server/test_server.py`
- Modify: `ipso/upload-server/server.py`

- [ ] **Step 1: Add failing test**

Append to `test_server.py`:

```python
class SizeLimitTest(unittest.TestCase):
    def setUp(self):
        self.h = ServerHarness()

    def tearDown(self):
        self.h.close()

    def test_content_length_above_limit_returns_413(self):
        # Forge a Content-Length above MAX_BODY_BYTES. We send a smaller
        # body — the server should reject based on the header alone, before
        # reading.
        big = str(server.MAX_BODY_BYTES + 1)
        h = dict(VALID_HEADERS, **{"Content-Length": big})
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.h.port, timeout=2
        )
        conn.putrequest("POST", "/upload")
        for k, v in h.items():
            conn.putheader(k, v)
        conn.endheaders()
        # Server should respond 413 before we send any body bytes.
        resp = conn.getresponse()
        self.assertEqual(resp.status, 413)
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL.

- [ ] **Step 3: Enforce size limit**

Insert, in `do_POST`, immediately after the session-id check and before reading the body:

```python
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                _json_response(self, 413, {"ok": False, "error": "too_large"})
                return
            body = self.rfile.read(length) if length else b""
```

(Replace the existing `length = ...` / `body = ...` pair.)

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS (12 tests now).

- [ ] **Step 5: Commit**

```bash
git add ipso/upload-server/server.py ipso/upload-server/test_server.py
git commit -m "feat(ipso-upload): enforce MAX_BODY_BYTES from Content-Length"
```

---

## Task 5: Server — idempotency (duplicate 200, conflict 409)

**Files:**
- Modify: `ipso/upload-server/test_server.py`
- Modify: `ipso/upload-server/server.py`

- [ ] **Step 1: Add failing tests**

Append to `test_server.py`:

```python
class IdempotencyTest(unittest.TestCase):
    def setUp(self):
        self.h = ServerHarness()

    def tearDown(self):
        self.h.close()

    def test_identical_resend_returns_200_duplicate_true(self):
        a = self.h.request("POST", "/upload", SAMPLE_CSV, VALID_HEADERS)
        self.assertEqual(a[0], 200)
        b = self.h.request("POST", "/upload", SAMPLE_CSV, VALID_HEADERS)
        self.assertEqual(b[0], 200)
        payload = json.loads(b[2])
        self.assertTrue(payload["duplicate"])

    def test_different_body_same_uuid_returns_409(self):
        self.h.request("POST", "/upload", SAMPLE_CSV, VALID_HEADERS)
        changed = SAMPLE_CSV.replace("Mario Rossi", "Anna Bianchi")
        status, _, body = self.h.request("POST", "/upload", changed, VALID_HEADERS)
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(body), {"ok": False, "error": "conflict"})

        # The original file is preserved.
        path = os.path.join(
            self.h.cfg["upload_dir"],
            "11111111-2222-3333-4444-555555555555.csv",
        )
        with open(path, "rb") as fh:
            self.assertEqual(fh.read().decode("utf-8"), SAMPLE_CSV)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: 2 new failures.

- [ ] **Step 3: Implement idempotency**

Replace the file-write block in `do_POST` (the `try: fd = os.open ...` through `os.rename(tmp, dest)`) with:

```python
            dest = os.path.join(upload_dir, session_id + ".csv")
            try:
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            except FileExistsError:
                # Already on disk under this uuid — idempotent re-send if
                # bytes match, otherwise a conflict (different content for
                # the same session id).
                with open(dest, "rb") as fh:
                    existing = fh.read()
                if hmac.compare_digest(existing, body):
                    _json_response(self, 200, {
                        "ok": True,
                        "stored_as": session_id + ".csv",
                        "duplicate": True,
                    })
                else:
                    _json_response(self, 409, {
                        "ok": False, "error": "conflict",
                    })
                return

            tmp = dest + ".part"
            os.rename(dest, tmp)  # carry the O_EXCL claim into the .part name
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.rename(tmp, dest)

            _json_response(self, 200, {
                "ok": True,
                "stored_as": session_id + ".csv",
                "duplicate": False,
            })
```

Wait — that `os.rename(dest, tmp)` racing pattern is awkward. Simpler approach: open the `.part` file with `O_EXCL` first, write, then `os.link` (or `os.rename`) onto the final destination. The final destination's existence is the idempotency signal. Replace the block above with this cleaner version:

```python
            dest = os.path.join(upload_dir, session_id + ".csv")
            tmp = dest + ".part"

            # Idempotency check first: if the canonical file exists, we
            # are a re-send. Compare bytes; same → 200 duplicate; different
            # → 409. This avoids the O_EXCL-on-.part race entirely.
            if os.path.exists(dest):
                with open(dest, "rb") as fh:
                    existing = fh.read()
                if hmac.compare_digest(existing, body):
                    _json_response(self, 200, {
                        "ok": True,
                        "stored_as": session_id + ".csv",
                        "duplicate": True,
                    })
                else:
                    _json_response(self, 409, {
                        "ok": False, "error": "conflict",
                    })
                return

            try:
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            except FileExistsError:
                # Concurrent in-flight write to the same uuid — let the
                # client retry; idempotency catches it on the next attempt.
                _json_response(self, 503, {"ok": False, "error": "server"})
                return
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.rename(tmp, dest)

            _json_response(self, 200, {
                "ok": True,
                "stored_as": session_id + ".csv",
                "duplicate": False,
            })
```

(There is a small TOCTOU window between the `os.path.exists(dest)` check and the `os.open(tmp, O_EXCL)`. The window is harmless: a true racer either lands a different uuid, or completes before us and we degrade to the duplicate/conflict branch on retry. We do **not** need stricter serialization for v1.)

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS (14 tests now).

- [ ] **Step 5: Commit**

```bash
git add ipso/upload-server/server.py ipso/upload-server/test_server.py
git commit -m "feat(ipso-upload): idempotent retries via UUID-keyed dest file"
```

---

## Task 6: Server — meta sidecar + rate limit

**Files:**
- Modify: `ipso/upload-server/test_server.py`
- Modify: `ipso/upload-server/server.py`

- [ ] **Step 1: Add failing tests**

Append to `test_server.py`:

```python
class MetaSidecarTest(unittest.TestCase):
    def setUp(self):
        self.h = ServerHarness()

    def tearDown(self):
        self.h.close()

    def test_meta_sidecar_written_with_parsed_fields(self):
        self.h.request("POST", "/upload", SAMPLE_CSV, VALID_HEADERS)
        meta_path = os.path.join(
            self.h.cfg["upload_dir"],
            "11111111-2222-3333-4444-555555555555.meta.json",
        )
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self.assertEqual(meta["operatore"], "Mario Rossi")
        self.assertEqual(meta["compresa"], "Serra")
        self.assertEqual(meta["catastrofata"], 0)
        self.assertEqual(meta["tree_count"], 1)
        self.assertEqual(meta["schema_version"], "5")
        self.assertIn("received_at", meta)
        self.assertIn("remote_addr", meta)

    def test_meta_sidecar_handles_zero_tree_session(self):
        empty = BOM + CSV_HEADER + "\r\n"
        self.h.request("POST", "/upload", empty, VALID_HEADERS)
        meta_path = os.path.join(
            self.h.cfg["upload_dir"],
            "11111111-2222-3333-4444-555555555555.meta.json",
        )
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self.assertEqual(meta["operatore"], "")
        self.assertEqual(meta["compresa"], "")
        self.assertIsNone(meta["catastrofata"])
        self.assertEqual(meta["tree_count"], 0)


class RateLimitTest(unittest.TestCase):
    def setUp(self):
        # Override the harness with a tight rate limit.
        self.h = ServerHarness()
        # Stop and re-make with a small limit.
        self.h.close()
        self.h = ServerHarness()
        self.h.close()
        # Hand-make so we can set rate_limit_per_minute.
        self.h = ServerHarness()
        # The simpler thing: directly mutate cfg and restart.
        self.h.close()

        self.h = _RateLimitHarness(rate=3)

    def tearDown(self):
        self.h.close()

    def test_too_many_requests_returns_429(self):
        # 3 should pass; 4th should be 429.
        for _ in range(3):
            status, _, _ = self.h.request(
                "POST", "/upload",
                SAMPLE_CSV, VALID_HEADERS,
            )
            self.assertEqual(status, 200)
        status, _, body = self.h.request(
            "POST", "/upload", SAMPLE_CSV, VALID_HEADERS
        )
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(body), {"ok": False, "error": "rate_limited"})


class _RateLimitHarness(ServerHarness):
    def __init__(self, rate):
        # Skip parent __init__; reimplement with a custom rate.
        self.tmp = tempfile.mkdtemp(prefix="ipso-upload-rl-test-")
        self.port = _pick_port()
        cfg = {
            "bind_host": "127.0.0.1",
            "bind_port": self.port,
            "token": VALID_TOKEN,
            "upload_dir": os.path.join(self.tmp, "uploads"),
            "rate_limit_per_minute": rate,
            "expected_bom": BOM,
            "expected_header_prefix": "Data;Compresa;Particella;Catastrofata;",
        }
        os.makedirs(cfg["upload_dir"])
        self.cfg = cfg
        self._httpd = server.make_server(cfg)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()
        for _ in range(50):
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                c.request("OPTIONS", "/upload")
                c.getresponse().read()
                c.close()
                break
            except ConnectionRefusedError:
                time.sleep(0.02)
```

(Yes, the awkward setUp dance in `RateLimitTest` is intentional — drop it and use the `_RateLimitHarness` directly. Simpler refactor:)

Replace the `RateLimitTest.setUp`/`tearDown` with:

```python
class RateLimitTest(unittest.TestCase):
    def setUp(self):
        self.h = _RateLimitHarness(rate=3)

    def tearDown(self):
        self.h.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: 3 new failures (2 meta + 1 rate).

- [ ] **Step 3: Implement meta sidecar + rate limit**

Add at module level in `server.py`:

```python
class TokenBucket:
    """Per-key token bucket for rate limiting. Per-minute refill."""

    def __init__(self, per_minute):
        self.cap = per_minute
        self.rate = per_minute / 60.0
        self._buckets = {}  # key -> [tokens, last_refill_ts]

    def take(self, key):
        now = time.monotonic()
        slot = self._buckets.get(key)
        if slot is None:
            self._buckets[key] = [self.cap - 1, now]
            return True
        tokens, last = slot
        tokens = min(self.cap, tokens + (now - last) * self.rate)
        last = now
        if tokens >= 1:
            slot[0] = tokens - 1
            slot[1] = last
            return True
        slot[0] = tokens
        slot[1] = last
        return False


def _client_ip(handler):
    xff = handler.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return handler.client_address[0]


def _meta_from_csv(text, schema_version, remote_addr):
    after_bom = text.lstrip("﻿")
    lines = after_bom.splitlines()
    # Header is always present; first data row may be absent.
    header = lines[0].split(";") if lines else []
    rows = lines[1:]
    if not rows:
        return {
            "operatore": "",
            "compresa": "",
            "catastrofata": None,
            "tree_count": 0,
            "schema_version": schema_version,
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                         time.gmtime()),
            "remote_addr": remote_addr,
        }
    idx = {name: i for i, name in enumerate(header)}
    first = rows[0].split(";")
    def cell(name):
        i = idx.get(name)
        return first[i] if i is not None and i < len(first) else ""
    cat_cell = cell("Catastrofata")
    return {
        "operatore": cell("Operatore"),
        "compresa": cell("Compresa"),
        "catastrofata": int(cat_cell) if cat_cell in ("0", "1") else None,
        "tree_count": len(rows),
        "schema_version": schema_version,
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                     time.gmtime()),
        "remote_addr": remote_addr,
    }
```

In `make_handler`, just after `upload_dir = cfg["upload_dir"]`:

```python
    bucket = TokenBucket(cfg.get("rate_limit_per_minute", 10))
```

In `do_POST`, immediately after the session-id check and before the size check:

```python
            client_ip = _client_ip(self)
            if not bucket.take(client_ip):
                _json_response(self, 429, {
                    "ok": False, "error": "rate_limited",
                })
                return
```

In `do_POST`, immediately after the successful `os.rename(tmp, dest)` (for the non-duplicate path) and before the success response:

```python
            try:
                meta = _meta_from_csv(
                    text, self.headers.get("X-Ipso-Schema-Version", ""),
                    client_ip,
                )
                meta_path = os.path.join(upload_dir, session_id + ".meta.json")
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh)
            except Exception as exc:  # best-effort
                log.warning("meta sidecar write failed for %s: %s",
                            session_id, exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: PASS (17 tests now).

- [ ] **Step 5: Commit**

```bash
git add ipso/upload-server/server.py ipso/upload-server/test_server.py
git commit -m "feat(ipso-upload): meta sidecar + per-IP rate limit"
```

---

## Task 7: Server — startup `.part` sweep + structured logging

**Files:**
- Modify: `ipso/upload-server/test_server.py`
- Modify: `ipso/upload-server/server.py`

- [ ] **Step 1: Add failing test**

Append to `test_server.py`:

```python
class StartupSweepTest(unittest.TestCase):
    def test_old_part_files_removed_on_start(self):
        tmp = tempfile.mkdtemp(prefix="ipso-upload-sweep-")
        uploads = os.path.join(tmp, "uploads")
        os.makedirs(uploads)
        stale = os.path.join(uploads, "abc.part")
        with open(stale, "w") as fh:
            fh.write("x")
        old = time.time() - 3600
        os.utime(stale, (old, old))
        fresh = os.path.join(uploads, "xyz.part")
        with open(fresh, "w") as fh:
            fh.write("y")
        # Brand new — should be left alone.
        server.sweep_stale_parts(uploads, older_than_s=60)
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(fresh))
        shutil.rmtree(tmp)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL.

- [ ] **Step 3: Implement sweep**

Add at module level in `server.py`:

```python
def sweep_stale_parts(upload_dir, older_than_s=60):
    """Remove .part files older than older_than_s seconds.

    Called at server startup to clean up after a previous process that
    crashed mid-write. Stale .part files are not visible via the final
    filename, so they are pure leakage.
    """
    cutoff = time.time() - older_than_s
    for name in os.listdir(upload_dir):
        if not name.endswith(".part"):
            continue
        path = os.path.join(upload_dir, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.unlink(path)
                log.info("swept stale part: %s", name)
        except FileNotFoundError:
            pass
```

In `main`, call it just before `srv.serve_forever()`:

```python
    sweep_stale_parts(cfg["upload_dir"])
```

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS (18 tests now).

- [ ] **Step 5: Add structured per-request log**

Replace the success-response block in the non-duplicate path so it logs first. In `do_POST`, replace the final `_json_response(self, 200, ...)` (the one after the sidecar write) with:

```python
            log.info(
                "upload session_id=%s bytes=%d duplicate=false "
                "operatore=%r compresa=%r tree_count=%d client_ip=%s",
                session_id, len(body),
                meta.get("operatore", ""), meta.get("compresa", ""),
                meta.get("tree_count", 0), client_ip,
            )
            _json_response(self, 200, {
                "ok": True,
                "stored_as": session_id + ".csv",
                "duplicate": False,
            })
```

And similarly, in the duplicate branch, just before the `_json_response(... duplicate: True)` call, add:

```python
                    log.info(
                        "upload session_id=%s bytes=%d duplicate=true "
                        "client_ip=%s",
                        session_id, len(body), client_ip,
                    )
```

- [ ] **Step 6: Run tests to verify they still pass**

```bash
cd ipso/upload-server && make test
```

Expected: PASS, 18 tests.

- [ ] **Step 7: Commit**

```bash
git add ipso/upload-server/server.py ipso/upload-server/test_server.py
git commit -m "feat(ipso-upload): startup .part sweep + structured request log"
```

---

## Task 8: Server — example config + systemd unit + README

**Files:**
- Create: `ipso/upload-server/config.example.json`
- Create: `ipso/upload-server/ipso-upload.service`
- Create: `ipso/upload-server/README.md`

- [ ] **Step 1: Write `config.example.json`**

```json
{
  "bind_host": "127.0.0.1",
  "bind_port": 8765,
  "token": "REPLACE-WITH-RANDOM-32-BYTE-BASE64-VALUE",
  "upload_dir": "/var/lib/ipso-upload/uploads",
  "rate_limit_per_minute": 10,
  "expected_bom": "﻿",
  "expected_header_prefix": "Data;Compresa;Particella;Catastrofata;"
}
```

- [ ] **Step 2: Write `ipso-upload.service`**

```ini
[Unit]
Description=ipso CSV upload receiver
After=network.target

[Service]
Type=simple
User=ipso-upload
Group=ipso-upload
ExecStart=/usr/bin/python3 /opt/ipso-upload/server.py /etc/ipso-upload/config.json
Restart=on-failure
RestartSec=5
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
NoNewPrivileges=true
ReadWritePaths=/var/lib/ipso-upload

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Write `README.md`**

```markdown
# ipso-upload

Tiny stdlib HTTP server that accepts session CSVs from the ipso PWA
and stores them on disk keyed by session UUID. See
`docs/superpowers/specs/2026-05-17-ipso-upload-design.md` for the full
spec; this README is the operator-facing notes.

## Layout on the VM

| Path                                 | Owner                | Notes                                |
|--------------------------------------|----------------------|--------------------------------------|
| `/opt/ipso-upload/server.py`         | `root:root`, `0755`  | The server binary (copy of this file)|
| `/etc/ipso-upload/config.json`       | `root:root`, `0600`  | Token + paths + rate limit           |
| `/etc/systemd/system/ipso-upload.service` | `root:root`     | systemd unit                         |
| `/var/lib/ipso-upload/uploads/`      | `ipso-upload:ipso-upload`, `0750` | Where CSVs and meta sidecars land |
| Apache vhost (ipso.laforesta.it)     | -                    | Adds `ProxyPass /upload`             |

## Local dev

    cd ipso/upload-server
    cp config.example.json config.json   # then set token + upload_dir
    make run

## Tests

    cd ipso/upload-server
    make test

## Rotating the token

Two-step rotation:

1. Update `/etc/ipso-upload/config.json` on the VM and
   `sudo systemctl restart ipso-upload`.
2. On the developer laptop, update `ipso/secrets/upload_config.json`
   with the new token, then `cd ipso && make deploy`.

In-flight uploads that span the rotation window will return `401`;
the client treats this as a hard error and the operator can bail to
local-only delivery for that one session.

## Logs

    sudo journalctl -u ipso-upload -f

Each upload emits a single structured line including the session id,
operator, compresa, tree count, byte size, and source IP.

## Retrieval (office side)

The office workstation runs (e.g., from cron):

    rsync -av --remove-source-files \
      ipso.laforesta.it:/var/lib/ipso-upload/uploads/*.csv \
      ipso.laforesta.it:/var/lib/ipso-upload/uploads/*.meta.json \
      ./incoming/

`--remove-source-files` keeps the server directory bounded. The meta
sidecar file moves with its CSV.
```

- [ ] **Step 4: Commit**

```bash
git add ipso/upload-server/config.example.json ipso/upload-server/ipso-upload.service ipso/upload-server/README.md
git commit -m "docs(ipso-upload): example config, systemd unit, deploy README"
```

---

## Task 9: Server — full test sweep + lint pass

- [ ] **Step 1: Run the full suite**

```bash
cd ipso/upload-server && make test
```

Expected: ~18 tests pass, exit 0.

- [ ] **Step 2: Hand-test against `config.example.json`**

```bash
cd ipso/upload-server
sed 's|/var/lib/ipso-upload/uploads|/tmp/ipso-upload-dev|' \
    config.example.json > config.local.json
mkdir -p /tmp/ipso-upload-dev
python3 server.py config.local.json &
sleep 1
curl -i \
  -H 'Authorization: Bearer REPLACE-WITH-RANDOM-32-BYTE-BASE64-VALUE' \
  -H 'Content-Type: text/csv; charset=utf-8' \
  -H 'X-Ipso-Session-Id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' \
  -H 'X-Ipso-Schema-Version: 5' \
  --data-binary $'\xef\xbb\xbfData;Compresa;Particella;Catastrofata;Numero;Specie;D_cm;H_m;H_measured;Lat;Lng;Acc_m;Operatore\r\n11/05/2026;Serra;1;0;;Abete;42;24;0;38,425310;16,120440;7;Mario Rossi\r\n' \
  http://127.0.0.1:8765/upload
kill %1
ls /tmp/ipso-upload-dev/
```

Expected: `HTTP/1.0 200 OK`, JSON body, two files in the directory (`.csv` + `.meta.json`).

- [ ] **Step 3: Commit (chore: keep `config.local.json` in `.gitignore`)**

Add `config.local.json` to `ipso/upload-server/.gitignore`:

```
config.local.json
__pycache__/
```

```bash
git add ipso/upload-server/.gitignore
git commit -m "chore(ipso-upload): gitignore local dev config"
```

---

## Task 10: Client — `secrets/` directory + `build_upload_config.py`

**Files:**
- Create: `ipso/secrets/upload_config.json.example`
- Create: `ipso/tools/build_upload_config.py`
- Modify: `ipso/.gitignore`

- [ ] **Step 1: Add the gitignore entry**

Append to `ipso/.gitignore`:

```
# Real upload secret (only on developer laptop). Example is committed.
secrets/upload_config.json
```

- [ ] **Step 2: Create the example secret file**

`ipso/secrets/upload_config.json.example`:

```json
{
  "upload_base": "https://ipso.laforesta.it",
  "token": "REPLACE-WITH-RANDOM-32-BYTE-BASE64-VALUE"
}
```

- [ ] **Step 3: Create the build script**

`ipso/tools/build_upload_config.py`:

```python
"""Generate build/upload-config.js from secrets/upload_config.json.

Writes a tiny JS file exposing UPLOAD_BASE and UPLOAD_TOKEN as globals,
loaded by index.html before upload.js. The token is visible to anyone
who can fetch the PWA bundle — see the spec for the explicit risk
tradeoff. The file is regenerated on every `make build` so a stale copy
cannot survive a token rotation.
"""

import json
import sys


HEADER = (
    "// AUTOGENERATED by tools/build_upload_config.py. Do not edit.\n"
    "// Source: ipso/secrets/upload_config.json (gitignored).\n"
    "'use strict';\n"
)


def main(argv):
    if len(argv) != 3:
        print(
            "usage: build_upload_config.py <secrets.json> <out.js>",
            file=sys.stderr,
        )
        return 2
    with open(argv[1], "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    base = cfg["upload_base"]
    token = cfg["token"]
    with open(argv[2], "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        fh.write("const UPLOAD_BASE = " + json.dumps(base) + ";\n")
        fh.write("const UPLOAD_TOKEN = " + json.dumps(token) + ";\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Hand-verify**

```bash
cd ipso
cp secrets/upload_config.json.example secrets/upload_config.json
mkdir -p build
python3 tools/build_upload_config.py \
  secrets/upload_config.json build/upload-config.js
cat build/upload-config.js
```

Expected output:

```
// AUTOGENERATED by tools/build_upload_config.py. Do not edit.
// Source: ipso/secrets/upload_config.json (gitignored).
'use strict';
const UPLOAD_BASE = "https://ipso.laforesta.it";
const UPLOAD_TOKEN = "REPLACE-WITH-RANDOM-32-BYTE-BASE64-VALUE";
```

- [ ] **Step 5: Commit**

```bash
git add ipso/.gitignore ipso/secrets/upload_config.json.example ipso/tools/build_upload_config.py
git commit -m "feat(ipso): build-time generator for upload-config.js"
```

---

## Task 11: Client — Makefile wiring

**Files:**
- Modify: `ipso/Makefile`

- [ ] **Step 1: Edit `ipso/Makefile`**

Replace the `build` target with:

```make
build:
	@if [ ! -f secrets/upload_config.json ]; then \
	  echo "ERROR: secrets/upload_config.json missing."; \
	  echo "  Copy secrets/upload_config.json.example and fill in the token."; \
	  echo "  See docs/superpowers/specs/2026-05-17-ipso-upload-design.md"; \
	  exit 1; \
	fi
	mkdir -p build
	rsync -a --delete src/ build/
	python3 tools/build_reference.py     build/reference.json
	python3 tools/vendor_geo.py          build/geo.js
	python3 tools/build_upload_config.py secrets/upload_config.json build/upload-config.js
	cp $(BOSCO_DATA)/terreni.geojson     build/terreni.geojson
```

Replace the `test` target with:

```make
test: build
	node test/tests.js
	$(MAKE) -C upload-server test
```

- [ ] **Step 2: Verify build still works**

```bash
cd ipso && make clean && make build
ls build/upload-config.js
```

Expected: file exists.

- [ ] **Step 3: Verify build fails cleanly without the secret**

```bash
cd ipso && mv secrets/upload_config.json secrets/_stash && make clean
make build || true
mv secrets/_stash secrets/upload_config.json
```

Expected: error message about missing `secrets/upload_config.json`, exit non-zero.

- [ ] **Step 4: Commit**

```bash
git add ipso/Makefile
git commit -m "build(ipso): wire upload-config + upload-server tests into make"
```

---

## Task 12: Client — `store.js` schema bump (pure helper TDD)

**Files:**
- Modify: `ipso/test/tests.js`
- Modify: `ipso/src/store.js`

- [ ] **Step 1: Add failing tests for the pure status helpers**

Append to `ipso/test/tests.js` (at the bottom, before the Summary block):

```javascript
// ---------------------------------------------------------------------------
// store.js — pure helpers (the IndexedDB code is exercised only in the
// browser; here we lock the status-set contract that the resume flow relies
// on).
// ---------------------------------------------------------------------------

console.log('\nstore.js (pure helpers)');

const { Store } = require('../build/store.js');

assertEqual(Store.SCHEMA_VERSION, 5, 'store: SCHEMA_VERSION bumped to 5');

assertEqual(Store.STATUS_OPEN, 'open', 'store: STATUS_OPEN constant');
assertEqual(Store.STATUS_PENDING_UPLOAD, 'pending_upload',
            'store: STATUS_PENDING_UPLOAD constant');
assertEqual(Store.STATUS_EXPORTED, 'exported', 'store: STATUS_EXPORTED constant');
assertEqual(Store.STATUS_ABANDONED, 'abandoned',
            'store: STATUS_ABANDONED constant');

assertEqual(Store.isResumableStatus(Store.STATUS_OPEN), true,
            'store: OPEN is resumable');
assertEqual(Store.isResumableStatus(Store.STATUS_PENDING_UPLOAD), true,
            'store: PENDING_UPLOAD is resumable');
assertEqual(Store.isResumableStatus(Store.STATUS_EXPORTED), false,
            'store: EXPORTED is not resumable');
assertEqual(Store.isResumableStatus(Store.STATUS_ABANDONED), false,
            'store: ABANDONED is not resumable');
assertEqual(Store.isResumableStatus(null), false,
            'store: null is not resumable');
assertEqual(Store.isResumableStatus('nonsense'), false,
            'store: unknown status is not resumable');
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ipso && make build && node test/tests.js
```

Expected: FAILs (SCHEMA_VERSION still 4; new constants and helper missing).

- [ ] **Step 3: Update `store.js`**

In `ipso/src/store.js`, change line ~19:

```javascript
const SCHEMA_VERSION = 5;
```

In the constant block (line ~25), add a new status:

```javascript
const STATUS_OPEN = 'open';
const STATUS_PENDING_UPLOAD = 'pending_upload';
const STATUS_EXPORTED = 'exported';
const STATUS_ABANDONED = 'abandoned';
```

Add a pure helper just below the status constants:

```javascript
function isResumableStatus(s) {
  return s === STATUS_OPEN || s === STATUS_PENDING_UPLOAD;
}
```

In `startSession`, extend the row literal to include the new fields (around line ~110):

```javascript
  const row = {
    id,
    schema_version: SCHEMA_VERSION,
    status: STATUS_OPEN,
    started_at: new Date().toISOString(),
    exported_at: null,
    data: fields.data,
    compresa: fields.compresa,
    catastrofata,
    operatore: fields.operatore || '',
    tree_count: 0,
    upload_status: null,
    uploaded_at: null,
  };
```

Replace `listOpenSessions` (line ~125) with `listResumableSessions`:

```javascript
async function listResumableSessions(db) {
  // Returns all sessions in a status that wants operator follow-up:
  // STATUS_OPEN (resume the recording) or STATUS_PENDING_UPLOAD
  // (retry the upload or confirm local-only).
  return tx(db, [STORE_SESSIONS], 'readonly', async (t) => {
    const all = await req(t.objectStore(STORE_SESSIONS).getAll());
    return all.filter((row) => isResumableStatus(row.status));
  });
}
```

Add a new helper next to `setSessionStatus`:

```javascript
async function setSessionUploadStatus(db, id, uploadStatus) {
  await tx(db, [STORE_SESSIONS], 'readwrite', async (t) => {
    const store = t.objectStore(STORE_SESSIONS);
    const row = await req(store.get(id));
    if (!row) throw new Error('ipso: session not found: ' + id);
    row.upload_status = uploadStatus;
    if (uploadStatus === 'uploaded') {
      row.uploaded_at = new Date().toISOString();
    }
    store.put(row);
  });
}
```

Update the public surface (bottom of `store.js`):

```javascript
const Store = {
  DB_NAME, SCHEMA_VERSION,
  STATUS_OPEN, STATUS_PENDING_UPLOAD, STATUS_EXPORTED, STATUS_ABANDONED,
  isResumableStatus,
  openDb,
  startSession, getSession, listResumableSessions, setSessionStatus,
  setSessionUploadStatus,
  addTree, listTrees, updateTree, deleteTree, lastTree,
  uuid,
};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ipso && make build && node test/tests.js
```

Expected: PASS for the new block.

- [ ] **Step 5: Commit**

```bash
git add ipso/src/store.js ipso/test/tests.js
git commit -m "feat(ipso): store v5 — upload_status, PENDING_UPLOAD, resumable filter"
```

---

## Task 13: Client — fix existing `listOpenSessions` callers

**Files:**
- Modify: `ipso/src/app.js`

- [ ] **Step 1: Find callers**

```bash
cd ipso && grep -n 'listOpenSessions' src/ test/
```

Expected: one match in `src/app.js:105`.

- [ ] **Step 2: Update the caller**

In `ipso/src/app.js`, change line ~105:

```javascript
  const open = await Store.listResumableSessions(State.db);
```

(Rename the local variable for clarity later; the resume modal will handle both statuses in Task 21.)

- [ ] **Step 3: Rebuild + verify tests still pass**

```bash
cd ipso && make test
```

Expected: PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add ipso/src/app.js
git commit -m "refactor(ipso): rename listOpenSessions caller to listResumableSessions"
```

---

## Task 14: Client — `upload.js` typed errors + backoff (TDD)

**Files:**
- Create: `ipso/src/upload.js`
- Modify: `ipso/test/tests.js`

- [ ] **Step 1: Add failing tests for backoff + classify**

Append to `ipso/test/tests.js`:

```javascript
// ---------------------------------------------------------------------------
// upload.js — pure helpers (backoff schedule, response classifier). The
// network-touching uploadSession() is exercised via a mocked globalThis.fetch
// further below.
// ---------------------------------------------------------------------------

console.log('\nupload.js (pure helpers)');

const upload = require('../build/upload.js');

// Backoff schedule: 2,4,8,16,30,30,30,...
assertEqual(upload.backoffMs(1), 2000, 'backoff: attempt 1');
assertEqual(upload.backoffMs(2), 4000, 'backoff: attempt 2');
assertEqual(upload.backoffMs(3), 8000, 'backoff: attempt 3');
assertEqual(upload.backoffMs(4), 16000, 'backoff: attempt 4');
assertEqual(upload.backoffMs(5), 30000, 'backoff: attempt 5 (cap)');
assertEqual(upload.backoffMs(6), 30000, 'backoff: attempt 6 (capped)');
assertEqual(upload.backoffMs(99), 30000, 'backoff: attempt 99 (capped)');
assertEqual(upload.backoffMs(0), 0, 'backoff: 0 attempts = 0 ms');

// Error classification
assertEqual(upload.classifyHttp(200), 'ok', 'classify: 200');
assertEqual(upload.classifyHttp(401), 'hard:auth', 'classify: 401');
assertEqual(upload.classifyHttp(409), 'hard:conflict', 'classify: 409');
assertEqual(upload.classifyHttp(413), 'hard:too_large', 'classify: 413');
assertEqual(upload.classifyHttp(422), 'hard:invalid_csv', 'classify: 422');
assertEqual(upload.classifyHttp(429), 'soft:rate_limited', 'classify: 429');
assertEqual(upload.classifyHttp(500), 'soft:server', 'classify: 500');
assertEqual(upload.classifyHttp(502), 'soft:server', 'classify: 502');
assertEqual(upload.classifyHttp(503), 'soft:server', 'classify: 503');
assertEqual(upload.classifyHttp(599), 'soft:server', 'classify: 599');
assertEqual(upload.classifyHttp(418), 'hard:invalid_csv',
            'classify: unknown 4xx -> hard:invalid_csv');
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: `Cannot find module ../build/upload.js`.

- [ ] **Step 3: Implement `upload.js`**

`ipso/src/upload.js`:

```javascript
// Upload an ipso session CSV to the ipso-upload endpoint.
//
// Pure-logic helpers (backoff, classify) are testable in node; the
// network-touching uploadSession() is exercised in browser via the
// screen-upload state machine, and in node tests via a mocked
// globalThis.fetch.
//
// See docs/superpowers/specs/2026-05-17-ipso-upload-design.md for the
// wire format and retry contract.
'use strict';

const BACKOFF_SCHEDULE_MS = [2000, 4000, 8000, 16000];
const BACKOFF_CAP_MS = 30000;

// 1-based attempt number → wait BEFORE that attempt. Attempt 0 is "no
// wait yet", attempt N>0 picks index N-1 from the schedule (or the cap).
function backoffMs(attempt) {
  if (!Number.isFinite(attempt) || attempt <= 0) return 0;
  const i = Math.floor(attempt) - 1;
  if (i < BACKOFF_SCHEDULE_MS.length) return BACKOFF_SCHEDULE_MS[i];
  return BACKOFF_CAP_MS;
}

// HTTP status → outcome class. The state machine drives bail/retry off the
// "hard:" / "soft:" prefix; the suffix lets the UI pick an error string.
function classifyHttp(status) {
  if (status === 200) return 'ok';
  if (status === 401) return 'hard:auth';
  if (status === 409) return 'hard:conflict';
  if (status === 413) return 'hard:too_large';
  if (status === 422) return 'hard:invalid_csv';
  if (status === 429) return 'soft:rate_limited';
  if (status >= 500 && status < 600) return 'soft:server';
  // Anything else 4xx-shaped: treat as a bug. Hard error stops retries.
  return 'hard:invalid_csv';
}

// Network / aborted-fetch failures classify the same way as 5xx (soft).
function classifyNetwork() { return 'soft:network'; }

class UploadError extends Error {
  constructor(klass, detail) {
    super(klass + (detail ? ': ' + detail : ''));
    this.klass = klass;          // 'hard:auth' | 'soft:server' | ...
    this.detail = detail || '';
  }
}

// Posts the CSV. Resolves with { duplicate: bool } on 200, throws
// UploadError otherwise. Caller passes signal for cancellation.
//
// base + token + schemaVersion come from upload-config.js (browser
// globals); the function accepts them as args so tests can inject a
// fake config without mucking with globals.
async function uploadSession(args) {
  const { base, token, schemaVersion, sessionId, csvText, signal } = args;
  let resp;
  try {
    resp = await fetch(base + '/upload', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'text/csv; charset=utf-8',
        'X-Ipso-Session-Id': sessionId,
        'X-Ipso-Schema-Version': '' + schemaVersion,
      },
      body: csvText,
      signal,
    });
  } catch (e) {
    if (e && e.name === 'AbortError') throw new UploadError('aborted');
    throw new UploadError(classifyNetwork(), e && e.message);
  }
  const klass = classifyHttp(resp.status);
  if (klass === 'ok') {
    let payload = {};
    try { payload = await resp.json(); } catch (_) {}
    return { duplicate: !!payload.duplicate, storedAs: payload.stored_as };
  }
  let detail = '';
  try {
    const payload = await resp.json();
    detail = payload && payload.error ? payload.error : '';
  } catch (_) {}
  throw new UploadError(klass, detail);
}

const upload = {
  BACKOFF_SCHEDULE_MS, BACKOFF_CAP_MS,
  backoffMs, classifyHttp, classifyNetwork,
  UploadError, uploadSession,
};

if (typeof module !== 'undefined') module.exports = upload;
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ipso && make build && node test/tests.js
```

Expected: PASS for backoff + classify blocks.

- [ ] **Step 5: Commit**

```bash
git add ipso/src/upload.js ipso/test/tests.js
git commit -m "feat(ipso): upload.js — typed errors, backoff schedule, uploadSession"
```

---

## Task 15: Client — `uploadSession` tests with mocked `fetch`

**Files:**
- Modify: `ipso/test/tests.js`

- [ ] **Step 1: Add failing tests**

Append to `ipso/test/tests.js`:

```javascript
// uploadSession with mocked globalThis.fetch.

console.log('\nupload.uploadSession (mocked fetch)');

async function withMockFetch(handler, fn) {
  const original = globalThis.fetch;
  globalThis.fetch = handler;
  try { return await fn(); } finally { globalThis.fetch = original; }
}

function mockResponse(status, payload) {
  return Promise.resolve({
    status,
    json: () => Promise.resolve(payload || {}),
  });
}

(async () => {
  // Happy path
  await withMockFetch(
    (url, init) => mockResponse(200, { ok: true, duplicate: false, stored_as: 'X.csv' }),
    async () => {
      const r = await upload.uploadSession({
        base: 'https://h', token: 't', schemaVersion: 5,
        sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        csvText: 'data',
      });
      assertEqual(r, { duplicate: false, storedAs: 'X.csv' },
                  'uploadSession: 200 returns duplicate=false');
    }
  );

  // 200 duplicate
  await withMockFetch(
    () => mockResponse(200, { ok: true, duplicate: true, stored_as: 'X.csv' }),
    async () => {
      const r = await upload.uploadSession({
        base: 'https://h', token: 't', schemaVersion: 5,
        sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        csvText: 'data',
      });
      assertEqual(r, { duplicate: true, storedAs: 'X.csv' },
                  'uploadSession: 200 returns duplicate=true');
    }
  );

  // 401 → UploadError 'hard:auth'
  await withMockFetch(
    () => mockResponse(401, { ok: false, error: 'auth' }),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession 401: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'hard:auth', 'uploadSession 401 -> hard:auth');
      }
    }
  );

  // 409 → UploadError 'hard:conflict'
  await withMockFetch(
    () => mockResponse(409, { ok: false, error: 'conflict' }),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession 409: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'hard:conflict', 'uploadSession 409 -> hard:conflict');
      }
    }
  );

  // 503 → UploadError 'soft:server'
  await withMockFetch(
    () => mockResponse(503, { ok: false, error: 'server' }),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession 503: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'soft:server', 'uploadSession 503 -> soft:server');
      }
    }
  );

  // Network error → 'soft:network'
  await withMockFetch(
    () => Promise.reject(new TypeError('network down')),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession network: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'soft:network',
                    'uploadSession network failure -> soft:network');
      }
    }
  );

  // Headers + body sent correctly
  let captured;
  await withMockFetch(
    (url, init) => {
      captured = { url, init };
      return mockResponse(200, { ok: true, duplicate: false, stored_as: 'X.csv' });
    },
    async () => {
      await upload.uploadSession({
        base: 'https://example.invalid', token: 'tok',
        schemaVersion: 5,
        sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        csvText: 'BODY',
      });
    }
  );
  assertEqual(captured.url, 'https://example.invalid/upload',
              'uploadSession: url base + /upload');
  assertEqual(captured.init.method, 'POST', 'uploadSession: method=POST');
  assertEqual(captured.init.headers.Authorization, 'Bearer tok',
              'uploadSession: Authorization header');
  assertEqual(captured.init.headers['Content-Type'], 'text/csv; charset=utf-8',
              'uploadSession: Content-Type header');
  assertEqual(captured.init.headers['X-Ipso-Session-Id'],
              'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
              'uploadSession: X-Ipso-Session-Id header');
  assertEqual(captured.init.headers['X-Ipso-Schema-Version'], '5',
              'uploadSession: X-Ipso-Schema-Version header');
  assertEqual(captured.init.body, 'BODY', 'uploadSession: body=csvText');
})();
```

Note: the existing `tests.js` is synchronous. The async block above will run to completion before `process.exit` fires because Node keeps the event loop alive until the microtask queue is empty. The final `process.exit(failed > 0 ? 1 : 0)` at the bottom of `tests.js` runs synchronously after the async block resolves. Verify this by running it.

- [ ] **Step 2: Run tests to verify they fail then pass**

```bash
cd ipso && make build && node test/tests.js
```

Expected: PASS (initially the test file was already written in Task 14 to depend on `upload.js`, so this is purely a mocked-fetch addition).

- [ ] **Step 3: Commit**

```bash
git add ipso/test/tests.js
git commit -m "test(ipso): mocked-fetch coverage for uploadSession"
```

---

## Task 16: Client — `strings.js` additions

**Files:**
- Modify: `ipso/src/strings.js`

- [ ] **Step 1: Append new constants**

In `ipso/src/strings.js`, before the `pill(rec)` method (around line ~95), insert:

```javascript
  // Upload screen
  UPLOAD_TITLE: 'Caricamento in corso',
  UPLOAD_ATTEMPT: (n) => `Tentativo ${n}`,
  UPLOAD_BAIL: 'Annulla caricamento e salva solo sul telefono',
  UPLOAD_SUCCESS_TOAST: 'Caricamento completato',
  UPLOAD_LOCAL_ONLY_TOAST: 'Salvato solo sul telefono',
  UPLOAD_ERROR_AUTH:
    'Errore di autenticazione. Contatta lo sviluppatore.',
  UPLOAD_ERROR_CONFLICT:
    'La sessione risulta già caricata con contenuto diverso. ' +
    'Contatta l\'ufficio.',
  UPLOAD_ERROR_INVALID:
    'Il server ha rifiutato il file. Contatta lo sviluppatore.',
  UPLOAD_ERROR_TOO_LARGE:
    'File troppo grande per il server. Contatta lo sviluppatore.',
  UPLOAD_ERROR_NETWORK: 'Errore di rete. Riprovo…',
  UPLOAD_ERROR_SERVER: 'Errore del server. Riprovo…',
  UPLOAD_ERROR_RATE_LIMITED: 'Server occupato. Riprovo…',
  UPLOAD_NEXT_RETRY_IN: (s) => `Prossimo tentativo fra ${s} s`,

  // Resume modal — upload variant
  UPLOAD_RESUME_TITLE: 'Sessioni in attesa di caricamento',
  UPLOAD_RESUME_DO_NOW: 'Carica ora',
  UPLOAD_RESUME_KEEP_LOCAL: 'Mantieni solo locale',
  UPLOAD_DONE_BODY: (n) =>
    `${n} alber${n === 1 ? 'o' : 'i'} caricat${n === 1 ? 'o' : 'i'} sul server.`,
```

- [ ] **Step 2: Verify tests still pass**

```bash
cd ipso && make test
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add ipso/src/strings.js
git commit -m "feat(ipso): Italian strings for upload screen + resume variant"
```

---

## Task 17: Client — `index.html` new screen + script tags

**Files:**
- Modify: `ipso/src/index.html`

- [ ] **Step 1: Insert the `screen-upload` section**

In `ipso/src/index.html`, immediately after the `<!-- ============= DONE ============= -->` block (before the modals comment, around line 131), insert a new section. Actually — put it BEFORE the DONE section so the upload screen and the done screen are adjacent:

```html
    <!-- ============= UPLOAD ============= -->
    <section id="screen-upload" class="screen" hidden>
      <h1 id="upload-title"></h1>
      <div class="upload-status">
        <div class="upload-spinner" id="upload-spinner" aria-hidden="true"></div>
        <p class="upload-attempt" id="upload-attempt"></p>
        <p class="upload-detail" id="upload-detail"></p>
      </div>
      <button id="btn-upload-bail" class="btn-secondary btn-big" type="button"></button>
    </section>
```

- [ ] **Step 2: Add the new script tags**

At the bottom of `index.html`, insert (in order, AFTER `version.js` and `strings.js`, BEFORE `app.js`):

```html
  <script src="upload-config.js"></script>
  <script src="upload.js"></script>
```

The full script block at the bottom becomes:

```html
  <script src="version.js"></script>
  <script src="strings.js"></script>
  <script src="csv.js"></script>
  <script src="ipso.js"></script>
  <script src="session.js"></script>
  <script src="download.js"></script>
  <script src="gps.js"></script>
  <script src="numpad.js"></script>
  <script src="store.js"></script>
  <script src="geo.js"></script>
  <script src="parcel-locator.js"></script>
  <script src="upload-config.js"></script>
  <script src="upload.js"></script>
  <script src="app.js"></script>
```

- [ ] **Step 3: Verify build**

```bash
cd ipso && make build
ls build/upload.js build/upload-config.js
```

Expected: both files present.

- [ ] **Step 4: Commit**

```bash
git add ipso/src/index.html
git commit -m "feat(ipso): screen-upload markup + upload.js/upload-config.js wiring"
```

---

## Task 18: Client — `style.css` for `screen-upload`

**Files:**
- Modify: `ipso/src/style.css`

- [ ] **Step 1: Append upload styles**

Append to `ipso/src/style.css`:

```css
/* ============= UPLOAD SCREEN ============= */

#screen-upload {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px 16px;
  gap: 24px;
  min-height: 60vh;
}

#screen-upload h1 {
  text-align: center;
  margin: 0;
}

.upload-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  flex: 1;
  justify-content: center;
}

.upload-spinner {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  border: 6px solid #d8e3d6;
  border-top-color: #1f5b1a;
  animation: upload-spin 1s linear infinite;
}

@keyframes upload-spin {
  to { transform: rotate(360deg); }
}

.upload-attempt {
  font-size: 18px;
  margin: 0;
}

.upload-detail {
  font-size: 14px;
  color: #555;
  text-align: center;
  margin: 0;
  max-width: 30em;
}

.upload-detail.error {
  color: #a02020;
}

/* The bail button stays prominent — the operator can hit it anytime. */
#btn-upload-bail {
  width: 100%;
  max-width: 360px;
}
```

- [ ] **Step 2: Commit**

```bash
git add ipso/src/style.css
git commit -m "feat(ipso): style screen-upload (spinner + attempt + bail)"
```

---

## Task 19: Client — `app.js` `enterUpload()` state machine

**Files:**
- Modify: `ipso/src/app.js`

- [ ] **Step 1: Wire the upload screen**

Add to `State` (around line ~24), inside the existing object:

```javascript
  upload: null,       // { sessionId, attempt, abortController, retryTimer } | null
```

Add inside `wireRecording()` (around line ~301, before `wireDone()`), append a new wire function called `wireUpload()` and call it from `boot()`. Actually, do it as a new function. Insert just before `wireDone()`:

```javascript
function wireUpload() {
  document.getElementById('upload-title').textContent = S.UPLOAD_TITLE;
  document.getElementById('btn-upload-bail').textContent = S.UPLOAD_BAIL;
  document.getElementById('btn-upload-bail').addEventListener('click', onUploadBail);
}
```

Call it from `boot()` after `wireRecording();` and before `wireDone();` (around line ~98):

```javascript
  wirePreSession();
  wireRecording();
  wireUpload();
  wireDone();
```

- [ ] **Step 2: Replace `onEnd` with the upload flow**

Find `onEnd` (currently around line ~662) and replace it:

```javascript
async function onEnd() {
  hideModal('modal-confirm-end');
  try {
    const trees = await Store.listTrees(State.db, State.session.id);
    trees.sort((a, b) => a.seq - b.seq);
    await Store.setSessionStatus(State.db, State.session.id, Store.STATUS_PENDING_UPLOAD);
    State.session.status = Store.STATUS_PENDING_UPLOAD;
    const csvText = csv.formatFile(State.session, trees);
    // Always download the local CSV first — it is the trust anchor and
    // the operator must not lose data if the upload never succeeds.
    downloadFinal(State.session, trees);
    enterUploadScreen(State.session.id, csvText, trees.length);
  } catch (e) {
    showToast('Errore esportazione: ' + e.message);
  }
}
```

- [ ] **Step 3: Add `enterUploadScreen` and the retry loop**

Insert near the existing upload-related functions (right after `onEnd`):

```javascript
function enterUploadScreen(sessionId, csvText, treeCount) {
  // Reset any prior state.
  if (State.upload && State.upload.retryTimer) {
    clearTimeout(State.upload.retryTimer);
  }
  if (State.upload && State.upload.abortController) {
    try { State.upload.abortController.abort(); } catch (_) {}
  }
  State.upload = {
    sessionId,
    csvText,
    treeCount,
    attempt: 0,
    abortController: null,
    retryTimer: null,
  };
  document.getElementById('upload-detail').textContent = '';
  document.getElementById('upload-detail').classList.remove('error');
  showScreen('screen-upload');
  acquireWakeLock();
  scheduleUploadAttempt(0);
}

function scheduleUploadAttempt(waitMs) {
  if (!State.upload) return;
  if (waitMs > 0) {
    const secs = Math.ceil(waitMs / 1000);
    document.getElementById('upload-detail').textContent =
      S.UPLOAD_NEXT_RETRY_IN(secs);
    State.upload.retryTimer = setTimeout(runUploadAttempt, waitMs);
  } else {
    runUploadAttempt();
  }
}

async function runUploadAttempt() {
  if (!State.upload) return;
  State.upload.retryTimer = null;
  State.upload.attempt += 1;
  document.getElementById('upload-attempt').textContent =
    S.UPLOAD_ATTEMPT(State.upload.attempt);

  const ac = new AbortController();
  State.upload.abortController = ac;
  try {
    await upload.uploadSession({
      base: UPLOAD_BASE,
      token: UPLOAD_TOKEN,
      schemaVersion: Store.SCHEMA_VERSION,
      sessionId: State.upload.sessionId,
      csvText: State.upload.csvText,
      signal: ac.signal,
    });
    await onUploadSuccess();
  } catch (err) {
    if (err && err.klass === 'aborted') return;  // bail handled elsewhere
    onUploadAttemptFailed(err);
  } finally {
    State.upload && (State.upload.abortController = null);
  }
}

function onUploadAttemptFailed(err) {
  if (!State.upload) return;
  const klass = (err && err.klass) || 'soft:network';
  const isHard = klass.startsWith('hard:');
  const detailEl = document.getElementById('upload-detail');
  detailEl.classList.toggle('error', isHard);
  detailEl.textContent = uploadErrorMessage(klass);
  if (isHard) {
    // Stop retrying; operator must bail.
    document.getElementById('upload-attempt').textContent = '';
    return;
  }
  scheduleUploadAttempt(upload.backoffMs(State.upload.attempt));
}

function uploadErrorMessage(klass) {
  switch (klass) {
    case 'hard:auth':         return S.UPLOAD_ERROR_AUTH;
    case 'hard:conflict':     return S.UPLOAD_ERROR_CONFLICT;
    case 'hard:invalid_csv':  return S.UPLOAD_ERROR_INVALID;
    case 'hard:too_large':    return S.UPLOAD_ERROR_TOO_LARGE;
    case 'soft:rate_limited': return S.UPLOAD_ERROR_RATE_LIMITED;
    case 'soft:server':       return S.UPLOAD_ERROR_SERVER;
    case 'soft:network':      return S.UPLOAD_ERROR_NETWORK;
    default:                  return klass;
  }
}

async function onUploadSuccess() {
  if (!State.upload) return;
  const treeCount = State.upload.treeCount;
  try {
    await Store.setSessionUploadStatus(
      State.db, State.upload.sessionId, 'uploaded'
    );
    await Store.setSessionStatus(
      State.db, State.upload.sessionId, Store.STATUS_EXPORTED
    );
  } catch (e) {
    showToast('Errore salvataggio stato upload: ' + e.message);
  }
  showToast(S.UPLOAD_SUCCESS_TOAST);
  endUploadScreen(treeCount, true);
}

async function onUploadBail() {
  if (!State.upload) return;
  if (State.upload.retryTimer) {
    clearTimeout(State.upload.retryTimer);
    State.upload.retryTimer = null;
  }
  if (State.upload.abortController) {
    try { State.upload.abortController.abort(); } catch (_) {}
  }
  const treeCount = State.upload.treeCount;
  try {
    await Store.setSessionUploadStatus(
      State.db, State.upload.sessionId, 'local_only'
    );
    await Store.setSessionStatus(
      State.db, State.upload.sessionId, Store.STATUS_EXPORTED
    );
  } catch (e) {
    showToast('Errore salvataggio stato: ' + e.message);
  }
  showToast(S.UPLOAD_LOCAL_ONLY_TOAST);
  endUploadScreen(treeCount, false);
}

function endUploadScreen(treeCount, uploaded) {
  State.upload = null;
  // Repurpose the existing done screen, but tailor the body text.
  document.getElementById('done-title').textContent = S.DONE_TITLE;
  document.getElementById('done-body').textContent = uploaded
    ? S.UPLOAD_DONE_BODY(treeCount)
    : S.DONE_BODY(treeCount);
  releaseWakeLock();
  showScreen('screen-done');
}
```

- [ ] **Step 4: Build + run tests**

```bash
cd ipso && make test
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ipso/src/app.js
git commit -m "feat(ipso): screen-upload state machine wired to Termina"
```

---

## Task 20: Client — resume modal extension for `PENDING_UPLOAD`

**Files:**
- Modify: `ipso/src/app.js`

- [ ] **Step 1: Update `showResumeModal`**

Find `showResumeModal` (around line ~844) and replace it:

```javascript
function showResumeModal(sessions) {
  // Mixed list: STATUS_OPEN sessions need resume/export/discard; new
  // STATUS_PENDING_UPLOAD sessions need carica-ora / mantieni-solo-locale.
  const hasUpload = sessions.some(
    (s) => s.status === Store.STATUS_PENDING_UPLOAD
  );
  const hasOpen = sessions.some(
    (s) => s.status === Store.STATUS_OPEN
  );
  document.getElementById('resume-title').textContent = hasUpload && !hasOpen
    ? S.UPLOAD_RESUME_TITLE
    : S.RESUME_TITLE;
  // Body line is generic enough for either case; leave RESUME_BODY in place.

  const list = document.getElementById('resume-list');
  list.replaceChildren();
  for (const s of sessions) {
    const li = document.createElement('li');
    li.className = 'resume-item';
    const meta = document.createElement('div');
    meta.className = 'resume-meta';
    meta.textContent =
      formatItalianDate(s.data) + ' · ' + S.where(s) +
      ' · ' + (s.operatore || '—') + ' · ' + (s.tree_count || 0) + ' alberi';
    li.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'resume-actions';
    if (s.status === Store.STATUS_PENDING_UPLOAD) {
      const carica = mkBtn(S.UPLOAD_RESUME_DO_NOW, 'btn-primary', async () => {
        hideModal('modal-resume');
        const trees = await Store.listTrees(State.db, s.id);
        trees.sort((a, b) => a.seq - b.seq);
        const csvText = csv.formatFile(s, trees);
        // Re-download the local CSV on every entry to screen-upload —
        // the browser auto-renames duplicates so this can never lose
        // the original copy. See spec.
        downloadFinal(s, trees);
        State.session = s;
        enterUploadScreen(s.id, csvText, trees.length);
      });
      const local = mkBtn(S.UPLOAD_RESUME_KEEP_LOCAL, 'btn-secondary', async () => {
        await Store.setSessionUploadStatus(State.db, s.id, 'local_only');
        await Store.setSessionStatus(State.db, s.id, Store.STATUS_EXPORTED);
        li.remove();
        if (!list.children.length) {
          hideModal('modal-resume');
          showScreen('screen-pre');
        }
      });
      actions.appendChild(carica);
      actions.appendChild(local);
    } else {
      const resume = mkBtn(S.RESUME_RESUME, 'btn-primary', async () => {
        State.session = s;
        State.lastTreeRow = await Store.lastTree(State.db, s.id);
        hideModal('modal-resume');
        enterRecording();
      });
      const exp = mkBtn(S.RESUME_EXPORT, 'btn-secondary', async () => {
        const trees = await Store.listTrees(State.db, s.id);
        trees.sort((a, b) => a.seq - b.seq);
        await Store.setSessionStatus(State.db, s.id, Store.STATUS_EXPORTED);
        downloadFinal(s, trees);
        li.remove();
        if (!list.children.length) {
          hideModal('modal-resume');
          showScreen('screen-pre');
        }
      });
      const discard = mkBtn(S.RESUME_DISCARD, 'btn-danger', async () => {
        await Store.setSessionStatus(State.db, s.id, Store.STATUS_ABANDONED);
        li.remove();
        if (!list.children.length) {
          hideModal('modal-resume');
          showScreen('screen-pre');
        }
      });
      actions.appendChild(resume);
      actions.appendChild(exp);
      actions.appendChild(discard);
    }
    li.appendChild(actions);
    list.appendChild(li);
  }
  showModal('modal-resume');
}
```

- [ ] **Step 2: Build + tests**

```bash
cd ipso && make test
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add ipso/src/app.js
git commit -m "feat(ipso): resume modal handles PENDING_UPLOAD sessions"
```

---

## Task 21: Client — service worker precache + version bump

**Files:**
- Modify: `ipso/src/sw.js`
- Modify: `ipso/src/version.js`

- [ ] **Step 1: Add upload files to `SHELL`**

In `ipso/src/sw.js`, extend the `SHELL` array (around line 13). Insert `./upload-config.js` and `./upload.js`:

```javascript
const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './style.css',
  './version.js',
  './app.js',
  './csv.js',
  './ipso.js',
  './session.js',
  './strings.js',
  './download.js',
  './gps.js',
  './numpad.js',
  './store.js',
  './geo.js',
  './parcel-locator.js',
  './upload.js',
  './upload-config.js',
  './reference.json',
  './terreni.geojson',
  './img/f.gif',
  './img/l.gif',
  './img/icon-192.png',
  './img/icon-512.png',
  './img/icon-512-maskable.png',
];
```

- [ ] **Step 2: Bump `APP_VERSION`**

In `ipso/src/version.js`:

```javascript
const APP_VERSION = '0.5.0';
```

- [ ] **Step 3: Verify build**

```bash
cd ipso && make build
ls build/upload-config.js build/upload.js
grep 'ipso-v0.5.0' build/sw.js || echo 'cache name follows APP_VERSION at runtime via importScripts'
```

(The literal `ipso-v` is built from `APP_VERSION` at runtime, so a literal match against the bumped version isn't expected; the importScripts pattern handles it.)

- [ ] **Step 4: Commit**

```bash
git add ipso/src/sw.js ipso/src/version.js
git commit -m "feat(ipso): precache upload assets + bump APP_VERSION to 0.5.0"
```

---

## Task 22: Docs — update `ipso/CLAUDE.md` and `ipso/README.md`

**Files:**
- Modify: `ipso/CLAUDE.md`
- Modify: `ipso/README.md`

- [ ] **Step 1: Extend `ipso/CLAUDE.md`**

After the `# Storage` section's v4 description (around line ~123), update the schema description to v5 and extend the session-row description:

```markdown
v5 shape:

- **session** row: `{id, schema_version, status, started_at,
  exported_at, data, compresa, catastrofata, operatore, tree_count,
  upload_status, uploaded_at}`.  `upload_status` is `null` for OPEN
  sessions and old (pre-v5) rows; `'uploaded'` or `'local_only'`
  once the operator finishes a session.  `status` gains a
  `pending_upload` value between `open` and `exported` — used while
  the upload retry loop is live.
- **tree** row: unchanged from v4.
```

Append a new top-level section at the bottom:

```markdown
# Upload to server

On Termina, the session CSV is uploaded to `${UPLOAD_BASE}/upload`
in addition to being written to Downloads.  The local CSV is the
trust anchor — it is written before any network call and on every
re-entry to `screen-upload` (browsers auto-rename duplicates).

The upload screen retries soft errors (5xx, 429, network) forever
on a `[2,4,8,16,30,30,...]` second backoff capped at 30 s, until
either success or the operator hits the bail button (saving local-
only).  Hard errors (401, 409, 413, 422) stop the retry loop and
require the operator to bail.

If a session ends in `STATUS_PENDING_UPLOAD` (app killed mid-retry
loop, or bail-then-want-to-retry-later), the next app open shows
that session in the resume modal with `[Carica ora]` /
`[Mantieni solo locale]` actions.

Wire format: see
`docs/superpowers/specs/2026-05-17-ipso-upload-design.md`.

The shared token ships in `build/upload-config.js`, generated by
`tools/build_upload_config.py` from `secrets/upload_config.json`
(gitignored).  Build refuses to proceed without it.  The same value
lives in `/etc/ipso-upload/config.json` on the VM; rotation is a
two-step process documented in `ipso/upload-server/README.md`.
```

- [ ] **Step 2: Update `ipso/README.md`**

In the "Using the app" section, after the "Termina e esporta CSV" paragraph (around line ~129), insert:

```markdown
After Termina, the app also uploads the same CSV to the server (the
office sees it in the inbox without manual handoff).  A progress
screen shows the attempt counter; if there's no signal, it retries
every few seconds (capped at 30 s).  You can hit **Annulla
caricamento e salva solo sul telefono** to keep the local copy and
move on — the upload will be offered again at the next app open if
you change your mind.
```

- [ ] **Step 3: Commit**

```bash
git add ipso/CLAUDE.md ipso/README.md
git commit -m "docs(ipso): document v5 schema bump and the upload flow"
```

---

## Task 23: Full local verification

- [ ] **Step 1: Clean build + full test sweep**

```bash
cd ipso
make clean
make test
```

Expected: server tests pass (~18), client tests pass.

- [ ] **Step 2: Serve locally and smoke-test the upload flow against the local server**

In one terminal:

```bash
cd ipso/upload-server
sed 's|/var/lib/ipso-upload/uploads|/tmp/ipso-upload-dev|' \
    config.example.json > config.local.json
mkdir -p /tmp/ipso-upload-dev
python3 server.py config.local.json
```

In another terminal:

```bash
cd ipso
# Point dev secrets at the local server.
cat > secrets/upload_config.json <<'EOF'
{
  "upload_base": "http://127.0.0.1:8765",
  "token": "REPLACE-WITH-RANDOM-32-BYTE-BASE64-VALUE"
}
EOF
make serve
```

In a browser at `http://localhost:8000/`:
- Start a new session.
- Add 2 trees.
- Hit Termina, confirm.
- Watch the upload screen show "Tentativo 1", then transition to Done.
- Verify two files in `/tmp/ipso-upload-dev/` (`<uuid>.csv` + `<uuid>.meta.json`).

- [ ] **Step 3: Smoke-test the bail path**

- Kill the upload server (`Ctrl+C` in its terminal).
- Start another session, add 1 tree, Termina, confirm.
- Watch the upload screen tick through retries.
- Hit "Annulla caricamento e salva solo sul telefono".
- Land on Done. Refresh the browser.
- Confirm the resume modal lists the session under "Sessioni non chiuse" wait — actually, since the operator chose `local_only`, the session is `STATUS_EXPORTED` and should NOT appear in the modal. **Verify it does not.**

- [ ] **Step 4: Smoke-test the recovery path**

- Start another session, add 1 tree, Termina, confirm.
- Watch the upload spinner. Reload the browser tab WITHOUT bailing or waiting for success.
- The session was set to `STATUS_PENDING_UPLOAD` before the network call. After the reload, the resume modal should list it with `[Carica ora]` / `[Mantieni solo locale]`.
- Restart the upload server.
- Tap "Carica ora", verify it transitions to Done with the success message.
- Verify a fresh local CSV download landed (browser names it with " (1)" suffix).

- [ ] **Step 5: Commit any local-only test config additions**

Nothing to commit if you only mutated `secrets/upload_config.json` (gitignored) and `config.local.json` (gitignored).

---

## Task 24: Pre-deploy ansible alignment

This task does **not** belong to this repo — it is a checklist for the parallel `../system/ansible/` change. Do not block this plan on it; flag it to the user as required before ipso is deployed.

- [ ] **Step 1: Confirm the ipso vhost can host `ProxyPass /upload`**

Inspect `../system/ansible/templates/apache2-sites/static-ssl.conf.j2`. If it does not expose an `extra_directives` hook, add one (preferred — small change, benefits future vhosts) or clone into `static-ssl-with-proxy.conf.j2` (alternative).

- [ ] **Step 2: Add an ansible role for `ipso-upload`**

Outline (per the spec):

- Create `ipso-upload` system user + group.
- `mkdir -p /var/lib/ipso-upload/uploads`, owned `ipso-upload:ipso-upload`, mode `0750`.
- `mkdir -p /etc/ipso-upload`, mode `0700`.
- Render `/etc/ipso-upload/config.json` from an ansible-vault template (token + paths).
- Copy `ipso/upload-server/server.py` to `/opt/ipso-upload/server.py`.
- Copy `ipso/upload-server/ipso-upload.service` to `/etc/systemd/system/ipso-upload.service`.
- `systemctl daemon-reload && systemctl enable --now ipso-upload`.

- [ ] **Step 3: Add the vhost extra directives**

For the ipso vhost, add (via the new template hook):

```
ProxyPass        /upload http://127.0.0.1:8765/upload
ProxyPassReverse /upload http://127.0.0.1:8765/upload
```

Reload Apache.

- [ ] **Step 4: Verify**

```bash
curl -I https://ipso.laforesta.it/upload
```

Expected: `401 Unauthorized` (no token).

- [ ] **Step 5: Place the token in `ipso/secrets/upload_config.json` on the developer laptop**

Same value as in `/etc/ipso-upload/config.json`.

---

## Task 25: Deploy + production smoke

- [ ] **Step 1: Update `ipso/secrets/upload_config.json` with production base + token**

```json
{
  "upload_base": "https://ipso.laforesta.it",
  "token": "<the-real-token-from-ansible-vault>"
}
```

- [ ] **Step 2: Deploy**

```bash
cd ipso && make deploy
```

Expected: build + tests pass, rsync to `ipso.laforesta.it`.

- [ ] **Step 3: Smoke-test on a real phone**

- Open the PWA on the phone (clear site data first if prompted, since `SCHEMA_VERSION` bumped — see `ipso/CLAUDE.md` Storage section).
- Run a short session (3 trees), Termina, confirm upload screen, observe success transition.
- `ssh ipso.laforesta.it 'ls -la /var/lib/ipso-upload/uploads/ | tail'` — confirm CSV + meta sidecar present.
- `ssh ipso.laforesta.it 'sudo journalctl -u ipso-upload --since "5 minutes ago"'` — confirm structured log line.

- [ ] **Step 4: Smoke-test offline retry**

- Toggle airplane mode ON.
- Run a short session (1 tree), Termina, confirm.
- Watch the upload screen attempt + back off.
- Toggle airplane mode OFF.
- Confirm the next attempt succeeds, screen transitions to Done.

- [ ] **Step 5: Smoke-test bail**

- Toggle airplane mode ON.
- Run a short session (1 tree), Termina, confirm.
- Hit Annulla.
- Confirm local-only Done screen.
- Toggle airplane mode OFF.
- Reload the PWA — resume modal should not surface this session (it was marked local_only).
- The CSV is in Downloads.

- [ ] **Step 6: Smoke-test killed-mid-retry recovery**

- Toggle airplane mode ON.
- Run a short session (1 tree), Termina, confirm upload screen.
- Force-quit the app while spinner is going.
- Toggle airplane mode OFF.
- Open the PWA. Resume modal should list the session with `[Carica ora]` / `[Mantieni solo locale]`.
- Tap `Carica ora`, confirm success.

---

## Self-review

**Spec coverage** (against `docs/superpowers/specs/2026-05-17-ipso-upload-design.md`):

| Spec area                                | Task(s)        |
|------------------------------------------|----------------|
| Server: happy path 200                    | 1              |
| Server: 401 paths                         | 2              |
| Server: 404, 405, 422 validation          | 3              |
| Server: 413 from Content-Length           | 4              |
| Server: 200 duplicate + 409 conflict      | 5              |
| Server: meta sidecar + rate limit         | 6              |
| Server: startup .part sweep + log         | 7              |
| Server: example config + systemd + README | 8              |
| Server: full local hand-test              | 9              |
| Client: build-time `upload-config.js`     | 10, 11         |
| Client: schema bump + status helpers      | 12, 13         |
| Client: `upload.js` typed errors+backoff  | 14             |
| Client: `uploadSession` over fetch        | 14, 15         |
| Client: Italian strings                   | 16             |
| Client: `screen-upload` markup + style    | 17, 18         |
| Client: `enterUpload` state machine       | 19             |
| Client: resume modal extension            | 20             |
| Client: SW precache + version bump        | 21             |
| Docs (CLAUDE.md + README)                 | 22             |
| Local verification                        | 23             |
| Ansible (out-of-repo)                     | 24 (checklist) |
| Deploy + production smoke                 | 25             |

No spec section is unrepresented.

**Placeholder scan**

- No `TBD`, `TODO`, or "implement later" tokens in tasks.
- Every code step shows the exact code.
- Every commit step shows the exact command.

**Type consistency**

- `Store.STATUS_PENDING_UPLOAD === 'pending_upload'` — used in `store.js`, `app.js` (resume modal + `onEnd`).
- `upload.uploadSession({ base, token, schemaVersion, sessionId, csvText, signal })` — same signature in tests and `app.js`.
- `upload.UploadError.klass` — `'hard:auth' | 'hard:conflict' | 'hard:invalid_csv' | 'hard:too_large' | 'soft:rate_limited' | 'soft:server' | 'soft:network' | 'aborted'` — used consistently in tests, `onUploadAttemptFailed`, `uploadErrorMessage`.
- `Store.setSessionUploadStatus(db, id, 'uploaded' | 'local_only')` — both call sites pass valid values.
- Backoff schedule `[2, 4, 8, 16, 30]` seconds matches the spec.
