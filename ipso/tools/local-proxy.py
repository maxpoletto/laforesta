#!/usr/bin/env python3
"""Local-dev reverse proxy: serve `build/` + proxy /upload to the upload-server.

Mimics the production Apache topology (single origin, /upload proxied
to the upload-server backend). This eliminates the cross-origin / CORS
question in local dev — the PWA and /upload share `http://localhost:8000/`.

Usage:
    python3 tools/local-proxy.py [PORT] [DIRECTORY]

Defaults: PORT=8000, DIRECTORY=build. Used by `make local-test`. Stdlib only.
"""

import http.client
import http.server
import os
import socketserver
import sys


UPLOAD_PATH = "/upload"
UPLOAD_HOST = "127.0.0.1"
UPLOAD_PORT = 8765
PROXY_TIMEOUT_S = 10
# Hop-by-hop headers per RFC 7230 + Content-Length (we recompute from
# the proxied body).
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve static files from cwd; proxy /upload to the upload-server."""

    def _is_upload(self):
        # Exact match or query-string variant.
        return self.path == UPLOAD_PATH or self.path.startswith(UPLOAD_PATH + "?")

    def do_POST(self):
        if self._is_upload():
            self._proxy("POST")
        else:
            self.send_error(405, "Method Not Allowed")

    def do_OPTIONS(self):
        if self._is_upload():
            self._proxy("OPTIONS")
        else:
            self.send_response(204)
            self.end_headers()

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        # Forward all headers except hop-by-hop; set X-Forwarded-For so
        # the upload-server's rate limit + meta sidecar see the real client.
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
        headers.setdefault("X-Forwarded-For", self.client_address[0])

        conn = http.client.HTTPConnection(
            UPLOAD_HOST, UPLOAD_PORT, timeout=PROXY_TIMEOUT_S,
        )
        try:
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in HOP_BY_HOP:
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)
        except (ConnectionRefusedError, OSError) as e:
            self.send_error(
                502, f"upload-server unreachable at "
                     f"{UPLOAD_HOST}:{UPLOAD_PORT}: {e}",
            )
        finally:
            conn.close()


class _ThreadingHTTPServer(socketserver.ThreadingMixIn,
                           http.server.HTTPServer):
    daemon_threads = True


def main(argv):
    port = int(argv[1]) if len(argv) > 1 else 8000
    directory = argv[2] if len(argv) > 2 else "build"
    os.chdir(directory)
    with _ThreadingHTTPServer(("127.0.0.1", port), Handler) as srv:
        print(
            f"local-proxy: serving {directory}/ on http://127.0.0.1:{port}/ "
            f"(proxying {UPLOAD_PATH} -> {UPLOAD_HOST}:{UPLOAD_PORT})",
            flush=True,
        )
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main(sys.argv)
