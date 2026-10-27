import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from api import API_PATH, make_handler


class InspectionReportApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.report_path = self.directory / "inspection.json"
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.report_path, self.directory),
        )
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}{API_PATH}"

    def _stop_server(self):
        self.server.shutdown()
        self.thread.join()

    def test_serves_valid_report_as_uncached_json(self):
        report = {"schema": "casting-inspection/1", "castings": []}
        self.report_path.write_text(json.dumps(report), encoding="utf-8")

        with urlopen(self.url) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                response.headers.get_content_type(),
                "application/json",
            )
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(json.load(response), report)

    def test_missing_report_returns_clear_error(self):
        with self.assertRaises(HTTPError) as raised:
            urlopen(self.url)

        self.assertEqual(raised.exception.code, 404)
        self.assertIn("Run export_results.py first", json.load(raised.exception)["error"])

    def test_rejects_report_with_wrong_schema(self):
        self.report_path.write_text(json.dumps({"schema": "unknown"}), encoding="utf-8")

        with self.assertRaises(HTTPError) as raised:
            urlopen(self.url)

        self.assertEqual(raised.exception.code, 503)
        self.assertIn("casting-inspection/1", json.load(raised.exception)["error"])

    def test_serves_frontend_from_same_origin(self):
        (self.directory / "index.html").write_text("<h1>Dashboard</h1>", encoding="utf-8")

        with urlopen(f"http://127.0.0.1:{self.server.server_port}/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"Dashboard", response.read())


if __name__ == "__main__":
    unittest.main()
