"""Check a split directly: does any validation casting have a twin in training?

    python verify_split.py                      checks data_grouped
    python verify_split.py --data data_clean    checks the old split
    python verify_split.py --neighbours 40

Why this exists separately from check_rotations.py
--------------------------------------------------
check_rotations.py builds the duplicate list that make_grouped_split.py uses.
If that list is incomplete, the split inherits the gap and nothing downstream
notices. And it IS incomplete: it verified only each image's 6 nearest
neighbours, and 845 images turned out to have 6 or more confirmed partners --
so 12.8% of the duplicated images had relationships that were never examined.

The error biases toward missing duplicates, which is the direction that leaves
twins straddling a split. A split built on that list cannot be called clean
just because the list says so.

This does not trust the list at all. It takes the split as it exists on disk
and asks the question directly, for every validation casting:

    how similar is this to the MOST similar training image, at any rotation?

Nothing is assumed about which images were supposed to be duplicates. The
answer is computed from the pixels.

Reading the result
------------------
There is no single cut that separates "duplicate" from "different casting", so
this does not pretend there is. An earlier version thresholded on the MINIMUM
of the sampled null, which is a terrible statistic: being the extreme of a
random draw, it came back 8.8, 11.2, 10.4 and 4.7 on four runs of the same
data, and the verdict moved with it.

Instead it reports how many validation castings fall below several cuts. Read
it the way you would read any sensitivity table -- if the conclusion is the
same at every cut, it is safe; if it flips, the answer is "we cannot tell".

For calibration on this dataset: below about 3 is visually identical, and the
difference image is black. Above about 10 is an ordinary pair of different
castings of the same part.

The number that matters is not the leak count on its own. It is what the model
scored on the castings that are NOT leaked. If a model got everything right and
only 11% of the split could possibly be leaked, the leak cannot be what
produced the score.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ("def_front", "ok_front")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_grouped")
    ap.add_argument("--neighbours", type=int, default=25,
                    help="training images checked per validation casting")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from check_rotations import (polar_of, descriptor, rotation_between,
                                 residual, sample_null, NR, NA, KEEP)

    root = Path(args.data)
    files = []
    for split in ("train", "val"):
        for cls in CLASSES:
            d = root / split / cls
            if d.is_dir():
                files.extend((p, split, cls) for p in sorted(d.rglob("*"))
                             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"No images under {root}/")
    tr = [i for i, (_, s, _) in enumerate(files) if s == "train"]
    va = [i for i, (_, s, _) in enumerate(files) if s == "val"]
    if not tr or not va:
        raise SystemExit(f"{root} needs both train/ and val/.")

    print(f"Split   : {root}")
    print(f"  train : {len(tr)}")
    print(f"  val   : {len(va)}")
    print()

    print("Fingerprinting")
    n = len(files)
    polars = np.empty((n, NR, NA), dtype=np.float32)
    desc = np.empty((n, NR * KEEP), dtype=np.float32)
    for i, (p, _, _) in enumerate(files):
        polars[i] = polar_of(p)
        desc[i] = descriptor(polars[i])
        if (i + 1) % 500 == 0:
            print(f"\r  {i+1}/{n}", end="", flush=True)
    print(f"\r  {n}/{n}")
    ffts = np.fft.rfft(polars, axis=2)
    del polars

    rng = np.random.default_rng(0)
    print("\nMeasuring the null (random pairs of different castings)")
    floors = {}
    for cls in CLASSES:
        idx = [i for i in tr if files[i][2] == cls]
        if len(idx) < 2:
            continue
        v = sample_null(files, ffts, idx, 300, rng)
        floors[cls] = float(v.min())
        print(f"  {cls:<10} min {v.min():5.1f}   1st pct "
              f"{np.percentile(v,1):5.1f}   median {np.percentile(v,50):5.1f}")

    print(f"\nFor each of the {len(va)} validation castings, finding its closest")
    print(f"training image out of {len(tr)} (top {args.neighbours} checked at "
          "their true rotation)")
    trd = desc[tr]
    best = np.empty(len(va)); best_j = np.empty(len(va), dtype=int)
    k = min(args.neighbours, len(tr))
    for c, i in enumerate(va):
        sims = trd @ desc[i]
        cand = np.argpartition(sims, -k)[-k:]
        bb, bj = 1e9, -1
        for t in cand:
            j = tr[int(t)]
            r = residual(files[i][0], files[j][0],
                         rotation_between(ffts[i], ffts[j]))
            if r < bb:
                bb, bj = r, j
        best[c], best_j[c] = bb, bj
        if (c + 1) % 100 == 0:
            print(f"\r  {c+1}/{len(va)}", end="", flush=True)
    print(f"\r  {len(va)}/{len(va)}")

    floor = min(floors.values()) if floors else 9.0
    print("\n  NOTE: that floor is the minimum of a random sample, so it moves")
    print("  between runs. It is printed for reference, not used as a verdict.")
    print("\n  Closest-training-match residual, per validation casting:")
    edges = np.linspace(0, max(best.max(), floor * 1.6), 17)
    hist, _ = np.histogram(best, bins=edges)
    peak = max(hist.max(), 1)
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        mark = "  <- null starts here" if lo <= floor < hi else ""
        print(f"    {lo:5.1f}-{hi:5.1f}  {'#'*int(30*h/peak):<30} {h:>5}{mark}")

    print()
    print("=" * 70)
    print(f"  HOW MUCH OF {root.name} COULD BE LEAKED")
    print("=" * 70)
    print(f"    {'cut':>6}{'below':>8}{'% of val':>10}{'castings left':>15}")
    for cut in (2.0, 3.0, 5.0, 7.0, 9.0):
        below = int((best < cut).sum())
        print(f"    {cut:>6.1f}{below:>8}{100*below/len(va):>9.1f}%"
              f"{len(va)-below:>15}")
    print()
    print("    Below ~3 is visually identical. Above ~10 is an ordinary pair")
    print("    of different castings. In between is a judgement call, which is")
    print("    why the whole range is shown instead of one number.")

    strict = [(best[c], files[va[c]][0].name, files[int(best_j[c])][0].name)
              for c in range(len(va)) if best[c] < 3.0]
    print()
    if strict:
        print(f"  {len(strict)} validation castings have a training image at "
              "residual < 3 --")
        print("  visually identical. Those are certainly leaked:")
        for r, a, b in sorted(strict)[:8]:
            print(f"    {r:5.2f}   {a}  <->  {b}")
    else:
        print("  No validation casting has a training image below residual 3.")
        print("  Nothing here is visually identical to something in training.")

    print()
    print("  WHAT THIS DOES AND DOES NOT MEAN")
    print("  A leak count is only half the story. Take the accuracy the model")
    print("  scored on this split, subtract the castings above, and ask what")
    print("  it got on the rest. If it was right on everything, then even the")
    print("  most paranoid cut leaves a large clean subset it also got right,")
    print("  and the leak cannot be what produced the score.")
    print("=" * 70)

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["val_file", "closest_train_file", "residual"])
            for c in np.argsort(best):
                w.writerow([files[va[c]][0].name,
                            files[int(best_j[c])][0].name, f"{best[c]:.3f}"])
        print(f"\n  {args.out}")
    return 0 if not strict else 1


if __name__ == "__main__":
    sys.exit(main())
