"""ipso-upload — receive CSVs from ipso and store them on disk.

Single-file stdlib HTTP server. Listens on 127.0.0.1; Apache fronts it
with HTTPS + ProxyPass. See ipso/upload-server/README.md for the
deploy story; docs/superpowers/specs/2026-05-17-ipso-upload-design.md
for the spec.
"""

import hmac
import http
import http.server
import json
import logging
import os
import re
import socketserver
import sys
import threading
import time

# owner rw, group r, world none — matches the systemd unit's intended umask
UPLOAD_FILE_MODE = 0o640

# Mirrors Apache's ansible-managed static_body_limit. Defense-in-depth:
# if the WSGI service is ever exposed without Apache in front, requests
# above this size fail closed.
MAX_BODY_BYTES = 1024 * 1024

# Wire protocol literals
BEARER_PREFIX = "Bearer "
CSV_CONTENT_TYPE_PREFIX = "text/csv"

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


class TokenBucket:
    """Per-key token bucket for rate limiting. Per-minute refill.

    Thread-safe: ThreadingTCPServer dispatches each request on its own
    thread, so concurrent take() calls from the same source can race
    without the lock.
    """

    def __init__(self, per_minute):
        self.cap = per_minute
        self.rate = per_minute / 60.0
        self._buckets = {}  # key -> [tokens, last_refill_ts]
        self._lock = threading.Lock()

    def take(self, key):
        now = time.monotonic()
        with self._lock:
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


def _meta_from_csv(after_bom, schema_version, remote_addr):
    # Caller has already validated and stripped the BOM.
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


def make_handler(cfg):
    upload_dir = cfg["upload_dir"]
    bucket = TokenBucket(cfg.get("rate_limit_per_minute", 10))
    token_bytes = cfg["token"].encode("utf-8")
    expected_bom = cfg["expected_bom"]
    expected_header_prefix = cfg["expected_header_prefix"]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            log.info("%s - %s", self.address_string(), format % args)

        def _invalid_csv(self, detail):
            _json_response(self, 422, {
                "ok": False, "error": "invalid_csv", "detail": detail,
            })

        def do_GET(self):
            _json_response(self, 405, {"ok": False, "error": "method_not_allowed"})

        def do_POST(self):
            if self.path != "/upload":
                _json_response(self, 404, {"ok": False, "error": "not_found"})
                return

            # Uniform 401 body for both "missing/malformed header" and "wrong
            # token" — don't disclose which failure mode the caller hit.
            auth = self.headers.get("Authorization", "")
            if not auth.startswith(BEARER_PREFIX):
                _json_response(self, 401, {"ok": False, "error": "auth"})
                return
            presented = auth[len(BEARER_PREFIX):].encode("utf-8")
            if not hmac.compare_digest(presented, token_bytes):
                _json_response(self, 401, {"ok": False, "error": "auth"})
                return

            ctype = self.headers.get("Content-Type", "")
            if not ctype.lower().startswith(CSV_CONTENT_TYPE_PREFIX):
                self._invalid_csv("wrong content type")
                return

            session_id = self.headers.get("X-Ipso-Session-Id", "")
            if not UUID_RE.match(session_id):
                self._invalid_csv("missing or malformed session id")
                return

            client_ip = _client_ip(self)
            if not bucket.take(client_ip):
                _json_response(self, 429, {
                    "ok": False, "error": "rate_limited",
                })
                return

            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                _json_response(self, 413, {"ok": False, "error": "too_large"})
                return
            body = self.rfile.read(length) if length else b""

            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                self._invalid_csv("body is not UTF-8")
                return

            if not text.startswith(expected_bom):
                self._invalid_csv("missing BOM")
                return
            after_bom = text[len(expected_bom):]
            if not after_bom.startswith(expected_header_prefix):
                self._invalid_csv("unexpected header")
                return

            dest = os.path.join(upload_dir, session_id + ".csv")
            tmp = dest + ".part"

            # Idempotency check first: if the canonical file exists, we
            # are a re-send. Compare bytes; same → 200 duplicate; different
            # → 409. This avoids the O_EXCL-on-.part race entirely.
            if os.path.exists(dest):
                with open(dest, "rb") as fh:
                    existing = fh.read()
                if hmac.compare_digest(existing, body):
                    # No meta rewrite on duplicate: preserves the
                    # original received_at / remote_addr.
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
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, UPLOAD_FILE_MODE)
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

            try:
                meta = _meta_from_csv(
                    after_bom, self.headers.get("X-Ipso-Schema-Version", ""),
                    client_ip,
                )
                meta_path = os.path.join(upload_dir, session_id + ".meta.json")
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh)
            except Exception as exc:  # best-effort
                log.warning("meta sidecar write failed for %s: %s",
                            session_id, exc)

            _json_response(self, 200, {
                "ok": True,
                "stored_as": session_id + ".csv",
                "duplicate": False,
            })

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

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
