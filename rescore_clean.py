"""Re-score existing runs on the leak-free subset of the validation set.

    python rescore_clean.py                  re-scores every run under results/
    python rescore_clean.py results/a        re-scores specific runs

check_leakage.py found validation images that are byte-identical to training
images. Scoring on those is scoring on data the model has already memorised,
so every accuracy figure that included them is inflated.

This does NOT require retraining. Each run already wrote predictions.csv with
one row per validation image, so the fix is to recompute the metrics over only
the images the model genuinely had not seen.

Caveat worth stating in any writeup: the leaked images were still in the
TRAINING set, so the weights saw them. Excluding them from scoring removes the
inflated measurement, not their (small, ~1% of training data) influence on the
model. A fully clean result needs them removed from training and a retrain.
"""

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFECT_HINTS = ("def", "damag", "bad", "fail", "reject", "crack", "flaw")


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hashes_under(root):
    table = defaultdict(list)
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            table[sha256(path)].append(path)
    return table


def leaked_val_files(base="data"):
    """Absolute paths of validation images that also appear in training."""
    train = hashes_under(Path(base) / "train")
    val = hashes_under(Path(base) / "val")
    leaked = set()
    for h in set(train) & set(val):
        for p in val[h]:
            leaked.add(p.resolve())
    return leaked, sum(len(v) for v in val.values())


def is_defect(label):
    return any(hint in label.lower() for hint in DEFECT_HINTS)


def score(rows):
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    tp = sum(1 for r in rows if is_defect(r["true"]) and is_defect(r["predicted"]))
    fn = sum(1 for r in rows if is_defect(r["true"]) and not is_defect(r["predicted"]))
    fp = sum(1 for r in rows if not is_defect(r["true"]) and is_defect(r["predicted"]))
    tn = total - tp - fn - fp
    actual_def = tp + fn
    return {
        "n": total,
        "correct": correct,
        "errors": total - correct,
        "accuracy": round(100.0 * correct / total, 2) if total else 0.0,
        "missed": fn,
        "false_alarms": fp,
        "missed_pct": round(100.0 * fn / actual_def, 2) if actual_def else 0.0,
        "false_alarm_pct": round(100.0 * fp / (tn + fp), 2) if (tn + fp) else 0.0,
    }


def load_predictions(run_dir):
    path = Path(run_dir) / "predictions.csv"
    if not path.is_file():
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "file": Path(row["file"]).resolve(),
                "true": row["true"],
                "predicted": row["predicted"],
                "correct": row["correct"].strip().lower() == "true",
            })
    return rows


def main():
    args = sys.argv[1:]
    runs = [Path(a) for a in args] if args else sorted(
        p for p in Path("results").iterdir()
        if (p / "predictions.csv").is_file())

    if not runs:
        raise SystemExit("No runs with predictions.csv found under results/.")

    print("Hashing data/train and data/val to identify leaked images ...")
    leaked, n_val = leaked_val_files()
    print(f"  {len(leaked)} of {n_val} validation images are duplicates of "
          f"training images ({100.0*len(leaked)/n_val:.2f}%)")
    print(f"  clean validation set: {n_val - len(leaked)} images")
    print()

    header = (f"  {'run':<34}{'as reported':>14}{'leak-free':>12}"
              f"{'errors':>9}{'missed':>8}{'false al':>10}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for run in runs:
        rows = load_predictions(run)
        if not rows:
            continue
        reported = score(rows)
        clean_rows = [r for r in rows if r["file"] not in leaked]
        clean = score(clean_rows)
        print(f"  {run.name:<34}{reported['accuracy']:>13.2f}%"
              f"{clean['accuracy']:>11.2f}%"
              f"{clean['errors']:>9}{clean['missed']:>8}"
              f"{clean['false_alarms']:>10}")

    print("=" * len(header))
    print()
    print("'leak-free' is the number to report. The leaked images were still")
    print("in the training set, so a fully clean result needs them removed")
    print("from training and a retrain -- but the inflated MEASUREMENT is")
    print("corrected here.")


if __name__ == "__main__":
    main()
