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

    def __init__(self, rate_limit_per_minute=1000):
        self.tmp = tempfile.mkdtemp(prefix="ipso-upload-test-")
        self.port = _pick_port()
        cfg = {
            "bind_host": "127.0.0.1",
            "bind_port": self.port,
            "token": VALID_TOKEN,
            "upload_dir": os.path.join(self.tmp, "uploads"),
            "rate_limit_per_minute": rate_limit_per_minute,
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
        self.assertEqual(json.loads(body)["error"], "invalid_csv")

    def test_wrong_content_type_returns_422(self):
        h = dict(VALID_HEADERS, **{"Content-Type": "application/json"})
        status, _, body = self.h.request("POST", "/upload", SAMPLE_CSV, h)
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "invalid_csv")

    def test_body_without_bom_returns_422(self):
        no_bom = SAMPLE_CSV[1:]  # strip BOM
        status, _, body = self.h.request("POST", "/upload", no_bom, VALID_HEADERS)
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "invalid_csv")

    def test_body_without_expected_header_returns_422(self):
        bad = BOM + "wrong;header;line\r\n"
        status, _, body = self.h.request("POST", "/upload", bad, VALID_HEADERS)
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error"], "invalid_csv")

    def test_wrong_method_returns_405(self):
        status, _, _ = self.h.request("GET", "/upload", None, VALID_HEADERS)
        self.assertEqual(status, 405)

    def test_wrong_path_returns_404(self):
        status, _, _ = self.h.request("POST", "/other", SAMPLE_CSV, VALID_HEADERS)
        self.assertEqual(status, 404)

    def test_get_on_unknown_path_returns_405_not_404(self):
        # do_GET runs before path inspection: method-not-allowed wins over
        # path-not-found. Avoids leaking which paths exist via 404 vs. 405.
        status, _, _ = self.h.request("GET", "/other", None, VALID_HEADERS)
        self.assertEqual(status, 405)


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
        self.assertEqual(json.loads(resp.read())["error"], "too_large")
        conn.close()


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
        self.h = ServerHarness(rate_limit_per_minute=3)

    def tearDown(self):
        self.h.close()

    def test_too_many_requests_returns_429(self):
        # First 3 requests use the same UUID so they hit the idempotency path
        # (no rate-limit cost from server work, but the rate-limit decrement
        # still happens). The 4th gets 429.
        for _ in range(3):
            status, _, _ = self.h.request(
                "POST", "/upload", SAMPLE_CSV, VALID_HEADERS,
            )
            self.assertEqual(status, 200)
        status, _, body = self.h.request(
            "POST", "/upload", SAMPLE_CSV, VALID_HEADERS
        )
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(body), {"ok": False, "error": "rate_limited"})


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


if __name__ == "__main__":
    unittest.main()
