"""Serve the frontend and the latest exported inspection report.

    python api.py
    python api.py --report results/inspection.json --port 8000
"""

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


API_PATH = "/api/inspection-report"
SCHEMA = "casting-inspection/1"
DEFAULT_FRONTEND_DIR = (
    Path(__file__).resolve().parent.parent / "AI-Manufacturing-UI---Frontend"
)


def make_handler(report_path, frontend_dir):
    class InspectionReportHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(frontend_dir), **kwargs)

        def do_GET(self):
            if urlsplit(self.path).path != API_PATH:
                super().do_GET()
                return

            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._send_json(
                    404,
                    {"error": "Inspection report not found. Run export_results.py first."},
                )
                return
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self.log_error("Unable to read %s: %s", report_path, error)
                self._send_json(
                    503,
                    {"error": "Inspection report is unavailable or invalid. Regenerate it."},
                )
                return

            if not isinstance(report, dict) or report.get("schema") != SCHEMA:
                self._send_json(
                    503,
                    {"error": f"Inspection report must use schema {SCHEMA}."},
                )
                return

            self._send_json(200, report)

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    return InspectionReportHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--report", type=Path, default=Path("results/inspection.json"))
    parser.add_argument("--frontend-dir", type=Path, default=DEFAULT_FRONTEND_DIR)
    args = parser.parse_args()

    if not args.frontend_dir.is_dir():
        parser.error(f"Frontend directory does not exist: {args.frontend_dir}")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(args.report, args.frontend_dir),
    )
    print(f"Frontend: http://{args.host}:{server.server_port}/")
    print(f"Report API: http://{args.host}:{server.server_port}{API_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
