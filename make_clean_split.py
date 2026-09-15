"""Build a leak-free copy of the dataset.

    python make_clean_split.py

Reads data/train and data/val, and writes data_clean/ containing:

    data_clean/train/   every training image EXCEPT those that are
                        byte-identical to a validation image
    data_clean/val/     the validation set, complete and untouched

Why remove from training rather than from validation: it leaves the published
test split intact at its full size, so results are reported on the standard
715-image benchmark rather than on a subset you carved yourself. It also costs
almost nothing -- the duplicates are under 1% of the training data.

Originals are never modified; this only copies.
"""

import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def index(root):
    table = defaultdict(list)
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            table[sha256(path)].append(path)
    return table


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data_clean")

    train_dir, val_dir = source / "train", source / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise SystemExit(f"Missing {d}")
    if out.exists():
        raise SystemExit(
            f"{out} already exists. Delete or rename it first so an old "
            "build cannot be mixed with a new one.")

    print(f"Hashing {train_dir} ...")
    train = index(train_dir)
    print(f"Hashing {val_dir} ...")
    val = index(val_dir)

    val_hashes = set(val)
    n_train = sum(len(v) for v in train.values())
    n_val = sum(len(v) for v in val.values())

    # -- copy validation through untouched --------------------------------
    copied_val = 0
    for paths in val.values():
        for p in paths:
            target = out / "val" / p.relative_to(val_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied_val += 1

    # -- copy training, skipping anything that also lives in validation ----
    copied_train = 0
    dropped = []
    for h, paths in train.items():
        if h in val_hashes:
            dropped.extend(paths)
            continue
        for p in paths:
            target = out / "train" / p.relative_to(train_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied_train += 1

    per_class = defaultdict(int)
    for p in dropped:
        per_class[p.parent.name] += 1

    print()
    print("=" * 58)
    print(f"  training   {n_train} -> {copied_train}   "
          f"({len(dropped)} removed)")
    print(f"  validation {n_val} -> {copied_val}   (unchanged)")
    print()
    for cls, n in sorted(per_class.items()):
        print(f"    removed from {cls}: {n}")
    print("=" * 58)
    print()
    print(f"Written to {out.resolve()}")
    print()
    print("Verify, then train on it:")
    print(f"    python check_leakage.py {out}")
    print("    python run.py clean")
    print("    python run.py clean augment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
