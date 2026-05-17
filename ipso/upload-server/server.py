"""ipso-upload — receive CSVs from ipso and store them on disk.

Single-file stdlib HTTP server. Listens on 127.0.0.1; Apache fronts it
with HTTPS + ProxyPass. See ipso/upload-server/README.md for the
deploy story; docs/superpowers/specs/2026-05-17-ipso-upload-design.md
for the spec.
"""

import http
import http.server
import json
import logging
import os
import socketserver
import sys

# owner rw, group r, world none — matches the systemd unit's intended umask
UPLOAD_FILE_MODE = 0o640

log = logging.getLogger("ipso-upload")


def _json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(cfg):
    upload_dir = cfg["upload_dir"]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            log.info("%s - %s", self.address_string(), format % args)

        def do_POST(self):
            if self.path != "/upload":
                _json_response(self, 404, {"ok": False, "error": "not_found"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

            session_id = self.headers.get("X-Ipso-Session-Id", "")
            dest = os.path.join(upload_dir, session_id + ".csv")

            tmp = dest + ".part"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, UPLOAD_FILE_MODE)
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
