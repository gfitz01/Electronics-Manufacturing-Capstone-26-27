"""Dataset acquisition helpers.

The original version of this module assumed every dataset would be pulled
from a public Google Drive link at runtime.  That assumption no longer holds:

* Google changed the large-file virus-scan interstitial.  The confirmation
  token moved out of a ``download_warning`` cookie and into a hidden field in
  the returned HTML, so the old cookie-only token scrape always returns
  ``None`` and the "zip" that lands on disk is actually an HTML error page.
* Public Drive files carry a per-file download quota.  Once it is hit the
  download fails for everyone until Google resets it, and no amount of client
  code can work around that.

So the download path here is now best-effort, and it fails *loudly* with an
actionable message instead of writing an HTML page into a ``.zip``.  The
supported path for coursework is :func:`use_local_archive` /
:func:`prepare_image_folder`: fetch the archive once by hand, point these
helpers at it, and never depend on Drive again.
"""

import os
import re
import shutil
import zipfile
from pathlib import Path
from zipfile import ZipFile

import requests

try:  # gdown is optional -- the local-data path does not need it.
    import gdown
except ImportError:  # pragma: no cover - depends on the user's environment
    gdown = None


# --------------------------------------------------------------------------
# Google Drive download (best effort)
# --------------------------------------------------------------------------

# Matches the confirm token both in the modern HTML form field and in the
# older query-string form, e.g. ``confirm=t&uuid=...`` or ``confirm=xyz123&``.
_CONFIRM_RE = re.compile(r"confirm=([0-9A-Za-z_\-]+)")


def get_confirm_token(response):
    """Extract Google Drive's download-confirmation token from a response.

    Checks the legacy ``download_warning`` cookie first, then falls back to
    scraping the token out of the interstitial HTML, which is where Google
    actually puts it now.  Returns ``None`` when no token is present (which
    means the response is the file itself, not an interstitial).
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        match = _CONFIRM_RE.search(response.text or "")
        if match:
            return match.group(1)

    return None


def save_response_content(response, destination):
    """Stream a response body to ``destination``."""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    """Download a public Drive file, handling the confirmation interstitial."""
    URL = "https://drive.usercontent.google.com/download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "export": "download"},
                           stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "export": "download", "confirm": token}
        response = session.get(URL, params=params, stream=True)

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise RuntimeError(
            "Google Drive returned an HTML page instead of the file. This "
            "usually means the per-file download quota has been exceeded or "
            "the file is no longer shared publicly. Download the archive "
            "manually in a browser and load it with "
            "ManufacturingNet.datasets.use_local_archive(path)."
        )

    save_response_content(response, destination)
    _assert_valid_zip(destination)


def download_file_from_google_drive_with_gdown(id, destination):
    """Download via gdown, which tracks Drive's interstitial changes."""
    if gdown is None:
        raise ImportError(
            "gdown is not installed. Either `pip install gdown` or download "
            "the archive manually and use "
            "ManufacturingNet.datasets.use_local_archive(path)."
        )

    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, destination, quiet=False, fuzzy=True)

    if not os.path.exists(destination):
        raise RuntimeError(
            f"gdown did not produce {destination}. Google Drive downloads are "
            "unreliable for these shared files (quota limits and changing "
            "interstitials). Download the archive manually in a browser and "
            "load it with ManufacturingNet.datasets.use_local_archive(path)."
        )

    _assert_valid_zip(destination)


def _assert_valid_zip(path):
    """Fail immediately if the download is not actually a zip archive."""
    if not zipfile.is_zipfile(path):
        size = os.path.getsize(path) if os.path.exists(path) else 0
        raise RuntimeError(
            f"{path} is not a valid zip archive ({size} bytes). Google Drive "
            "almost certainly returned an error page instead of the data. "
            "Download the archive manually in a browser and load it with "
            "ManufacturingNet.datasets.use_local_archive(path)."
        )


def extract_files(name, destination="."):
    Zip = ZipFile(name)
    Zip.extractall(destination)


def remove_zip(name):
    os.remove(name)


# --------------------------------------------------------------------------
# Local data path (recommended)
# --------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def use_local_archive(archive_path, destination="."):
    """Extract a manually downloaded dataset archive.

    This is the dependency-free path: download the zip once in a browser (or
    from the dataset's primary source), then point this at it.  No network
    call, no Drive quota, reproducible for every member of the team.

    Returns the directory the archive was extracted into.
    """
    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"No such archive: {archive_path}")

    _assert_valid_zip(archive_path)

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extract_files(archive_path, destination)
    return destination


def discover_classes(root):
    """Return the sorted class names ImageFolder would infer from ``root``."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def count_images(root):
    """Return ``{class_name: n_images}`` for an ImageFolder-style directory."""
    root = Path(root).expanduser().resolve()
    counts = {}
    for class_name in discover_classes(root):
        counts[class_name] = sum(
            1 for p in (root / class_name).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


def prepare_image_folder(source, output="data", val_split=0.2, seed=42,
                         move=False):
    """Split a flat class-folder dataset into train/ and val/ subdirectories.

    ``source`` is expected to look like::

        source/ok/*.jpg
        source/defective/*.jpg

    and the result is the layout ``torchvision.datasets.ImageFolder`` (and
    therefore every image model in this library) expects::

        output/train/ok/...      output/val/ok/...
        output/train/defective/  output/val/defective/

    Returns ``(train_dir, val_dir)`` as strings, ready to hand straight to
    ``AlexNet(train_dir, val_dir)``.
    """
    import random

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

    classes = discover_classes(source)
    if not classes:
        raise ValueError(
            f"No class subdirectories found in {source}. Expected one folder "
            "per class, e.g. source/ok/ and source/defective/."
        )

    train_dir = output / "train"
    val_dir = output / "val"
    rng = random.Random(seed)
    transfer = shutil.move if move else shutil.copy2

    for class_name in classes:
        images = sorted(
            p for p in (source / class_name).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"No images found in {source / class_name}")

        rng.shuffle(images)
        n_val = max(1, int(round(len(images) * val_split)))
        splits = {"val": images[:n_val], "train": images[n_val:]}

        for split_name, split_images in splits.items():
            target = output / split_name / class_name
            target.mkdir(parents=True, exist_ok=True)
            for image in split_images:
                transfer(str(image), str(target / image.name))

    return str(train_dir), str(val_dir)


# --------------------------------------------------------------------------
# Bundled dataset shortcuts
# --------------------------------------------------------------------------

DATASET_IDS = {
    "PaderbornBearingData": "15v1fwFxfrntTE1FVdNvZXzMF5xufMa3Z",
    "MotorTempData": "11Q5emmsc1dMMiMGe8niho5x4-3I5Oc_K",
    "ChatterData": "1z_2ceidvHmE5p7XCD4PaGn4ezvcZMxdD",
    "ThreeDPrintingData": "1VhZcOgNOEw_Sciuww25XZdIuaqO90Nkj",
    "MercedesData": "1D7eQDV4h6lEXnNE1Cbk1kRU62Dn9xMnb",
    "LithographyData": "1XY4fbNtzKrXXPtfiPGpwunWvyIDH_V57",
    "GearboxData": "1aTFu-M8V5CxbDY4e-nRBLNuPmbsbAbgk",
    "CastingData": "1qNnLCcq1HlzS0WmOCRlJfNC9ZF26j_6f",
    "CWRUBearingData": "1nUjYdpJkEmjJTzG0j8EBZ9vQul0sedqk",
}

# Where each dataset can be obtained if (when) the Drive link fails.
PRIMARY_SOURCES = {
    "CastingData":
        "Kaggle: 'casting product image data for quality inspection'",
    "CWRUBearingData":
        "Case Western Reserve University Bearing Data Center",
    "PaderbornBearingData":
        "Paderborn University KAt-DataCenter bearing dataset",
}


def _fetch(name, local_archive=None):
    """Shared implementation for the named dataset helpers."""
    destination = f"{name}.zip"

    if local_archive is not None:
        return use_local_archive(local_archive)

    try:
        download_file_from_google_drive_with_gdown(DATASET_IDS[name],
                                                   destination)
    except Exception as exc:
        hint = PRIMARY_SOURCES.get(name)
        message = (
            f"Could not download {name} from Google Drive: {exc}\n"
            f"Download the archive manually and call "
            f"{name}(local_archive='path/to/{destination}')."
        )
        if hint:
            message += f"\nPrimary source: {hint}"
        raise RuntimeError(message) from exc

    extract_files(destination)
    remove_zip(destination)
    return os.getcwd()


def PaderbornBearingData(local_archive=None):
    return _fetch("PaderbornBearingData", local_archive)


def MotorTempData(local_archive=None):
    return _fetch("MotorTempData", local_archive)


def ChatterData(local_archive=None):
    return _fetch("ChatterData", local_archive)


def ThreeDPrintingData(local_archive=None):
    return _fetch("ThreeDPrintingData", local_archive)


def MercedesData(local_archive=None):
    return _fetch("MercedesData", local_archive)


def LithographyData(local_archive=None):
    return _fetch("LithographyData", local_archive)


def GearboxData(local_archive=None):
    return _fetch("GearboxData", local_archive)


def CastingData(local_archive=None):
    return _fetch("CastingData", local_archive)


def CWRUBearingData(local_archive=None):
    return _fetch("CWRUBearingData", local_archive)
