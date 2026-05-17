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
BOM = "\ufeff"
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
        else:
            raise RuntimeError("ipso-upload test server did not start within 1s")

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
        status, _, body = self.h.request(
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
        self.assertEqual(json.loads(body), {"ok": False, "error": "auth"})


if __name__ == "__main__":
    unittest.main()
