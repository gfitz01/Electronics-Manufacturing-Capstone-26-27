"""Verify no image appears in both the training and validation sets.

    python check_leakage.py            checks data/train against data/val
    python check_leakage.py mydata     checks mydata/train against mydata/val

A perfect or near-perfect score is exactly when this check earns its keep. If
any validation image also appears in training, the model has seen the answer
and the score is meaningless. That is the first question a reviewer should ask
about a 100% result, and "we checked" is a much better answer than "it should
be fine."

Compares by SHA-256 of the file contents, not by filename -- the casting
dataset reuses the same naming scheme in both splits, so identical names do
not imply identical images, and different names do not imply different ones.
"""

import hashlib
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
    """{content hash: [paths]} for every image under root."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    table = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            table[sha256(path)].append(path)
    return table


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    train_dir, val_dir = base / "train", base / "val"

    print(f"Hashing {train_dir} ...")
    train = index(train_dir)
    print(f"Hashing {val_dir} ...")
    val = index(val_dir)

    n_train = sum(len(v) for v in train.values())
    n_val = sum(len(v) for v in val.values())
    print()
    print(f"  training images   : {n_train}")
    print(f"  validation images : {n_val}")
    print()

    # -- duplicates WITHIN each split ------------------------------------
    for name, table in (("training", train), ("validation", val)):
        dupes = {h: paths for h, paths in table.items() if len(paths) > 1}
        if dupes:
            extra = sum(len(p) - 1 for p in dupes.values())
            print(f"  NOTE: {name} contains {extra} duplicate image(s) "
                  f"({len(dupes)} repeated file(s)).")
            for paths in list(dupes.values())[:3]:
                print("        " + " == ".join(p.name for p in paths[:3]))
    print()

    # -- the one that matters: overlap BETWEEN splits ---------------------
    overlap = set(train) & set(val)

    print("=" * 62)
    if not overlap:
        print("PASS - no validation image appears in the training set.")
        print("The held-out score is measured on genuinely unseen data.")
    else:
        leaked = sum(len(val[h]) for h in overlap)
        pct = 100.0 * leaked / n_val if n_val else 0
        print(f"FAIL - {leaked} validation image(s) ({pct:.2f}%) also appear "
              "in training.")
        print("The reported accuracy is inflated by that much and cannot be")
        print("used as-is. Rebuild the split before reporting any figure.")
        print()
        for h in list(overlap)[:10]:
            print(f"    val: {val[h][0]}")
            print(f"  train: {train[h][0]}")
    print("=" * 62)
    return 0 if not overlap else 1


if __name__ == "__main__":
    sys.exit(main())
