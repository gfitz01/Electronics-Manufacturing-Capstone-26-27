import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manufacturingnet_datasets",
    ROOT / "ManufacturingNet" / "datasets" / "datasets.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Response:
    def __init__(self, cookies=None, headers=None, text="", chunks=None):
        self.cookies = cookies or {}
        self.headers = headers or {}
        self.text = text
        self._chunks = chunks or []

    def iter_content(self, chunk_size):
        for chunk in self._chunks:
            yield chunk


class GoogleDriveDownloadTests(unittest.TestCase):
    def test_get_confirm_token_from_cookie_and_html(self):
        cookie_response = Response({"download_warning_123": "abc"}, {"Content-Type": "application/octet-stream"})
        self.assertEqual(module.get_confirm_token(cookie_response), "abc")

        html_response = Response({}, {"Content-Type": "text/html"}, "confirm=xyz123&export=download")
        self.assertEqual(module.get_confirm_token(html_response), "xyz123")

    def test_save_response_content_creates_file(self):
        destination = ROOT / "tmp_download_test.bin"
        try:
            response = Response(chunks=[b"hello", b" world"])
            module.save_response_content(response, str(destination))
            self.assertEqual(destination.read_bytes(), b"hello world")
        finally:
            if destination.exists():
                destination.unlink()


if __name__ == "__main__":
    unittest.main()
