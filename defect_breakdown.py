"""Defect-type breakdown for a set of castings.

    python defect_breakdown.py
    python defect_breakdown.py --residuals clean.csv --min-residual 5.7
    python defect_breakdown.py --data data_clean      the original 715-image test

Joins defect_types/predicted.csv (from apply_defect_types.py) to whichever
castings are sitting in a split's validation folder, and reports what the
stage-two model thinks is wrong with them.

These are PREDICTIONS, not labels
---------------------------------
Every number here comes from a model with a measured accuracy, not from
inspection. Cross-validated AUC on the 200 hand-labelled castings:

    fray  1.00      spot  0.92      slash 0.92      chip  0.86

So the fray column is close to reliable and the chip column is the shakiest.
Quote these as estimates with that caveat attached, never as counts.

Two known biases, both upward, both on the same two flags
--------------------------------------------------------
Training used pos_weight to stop the rarer defects being ignored, which pushes
the model toward calling them present. Measured against the hand labels
reweighted to the full defect set, chip came out about 8 points high and slash
about 5. fray and spot landed within 3 points.

So read chip and slash as upper bounds. The direction of the error is known
even though the exact size is not.

Restricting to genuinely unseen castings
----------------------------------------
    python verify_split.py --out clean.csv
    python defect_breakdown.py --residuals clean.csv --min-residual 5.7

verify_split.py's --out lists every validation casting with how close its
nearest training image is. Passing it here drops the ones that might be
rotated duplicates, so the breakdown covers only castings the model had
genuinely never seen. 5.7 is the paranoid cut; 3.0 is the "visually
identical" cut.

Images are not castings
-----------------------
The grouped split keeps each rotation family together, which is what stops
leakage -- but it means the validation folder holds several rotations of the
same casting side by side. On the 20 Sept data_grouped, the 434 clean
defective images it could match are between 126 and 223 distinct castings.

Counting each image as independent makes the intervals too narrow. So when
rotation_duplicates.csv is present this reports the range of distinct
castings and computes the 95% interval at the LOWER end of it -- the cautious
choice. The range comes from two groupings of the same duplicate list:
union-find (over-merges, so a floor) and complete-linkage cliques (over-splits
where the neighbour cap left pairs unverified, so a ceiling).
"""

import argparse
import csv
import sys
from collections import Counter
from math import sqrt
import re
from pathlib import Path

FLAGS = ["fray", "chip", "spot", "slash"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# share predicted present over all 4,211 def_front (p >= 0.5) minus the hand
# labels' rate reweighted to the population (group sizes in blind_meta_r2.json):
#   fray 34.0 - 36.8   chip 20.3 - 12.5   spot 52.7 - 51.6   slash 36.3 - 31.3
KNOWN_BIAS = {"fray": -2.8, "chip": +7.8, "spot": +1.1, "slash": +5.0}
AUC = {"fray": 1.00, "chip": 0.86, "spot": 0.92, "slash": 0.92}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tail(p):
    """split/class/filename. The source dataset reuses filenames across its
    train and test folders for different images, so a bare name is not a key."""
    return "/".join(Path(str(p).replace("\\", "/")).parts[-3:])


PREFIX = re.compile(r"^from-(train|val)__(.+)$")


def source_of(disk_name, class_name, by_name):
    """Map a file in data_grouped/ back to its source image in data_clean/.

    make_grouped_split.py writes a collided filename as from-<split>__<name>,
    which is unambiguous. An unprefixed name is looked up directly, and is
    ambiguous only in a split made by the older script, which did not prefix;
    those return None and are left out rather than guessed."""
    m = PREFIX.match(disk_name)
    if m:
        return f"{m.group(1)}/{class_name}/{m.group(2)}"
    cands = by_name.get(disk_name, [])
    return cands[0] if len(cands) == 1 else None


def distinct_castings(names, dup_path):
    """(floor, ceiling) on how many distinct castings `names` contains.
    `names` are source keys (split/class/filename)."""
    from collections import defaultdict
    names = set(names)
    adj = defaultdict(set)
    parent = {x: x for x in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with open(dup_path, newline="") as f:
        for row in csv.DictReader(f):
            a = tail(row["file_a"])
            b = tail(row["file_b"])
            if a in names and b in names and a != b:
                adj[a].add(b)
                adj[b].add(a)
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    floor = len({find(x) for x in names})
    used, cliques = set(), 0
    for seed in sorted(adj, key=lambda k: (-len(adj[k]), k)):
        if seed in used:
            continue
        clique = [seed]
        for c in sorted(adj[seed], key=lambda k: (-len(adj[k]), k)):
            if c not in used and all(c in adj[m] for m in clique):
                clique.append(c)
        used.update(clique)
        cliques += 1
    return floor, cliques + len(names - used)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_grouped")
    ap.add_argument("--split", default="val")
    ap.add_argument("--class-name", default="def_front")
    ap.add_argument("--predicted", default="defect_types/predicted.csv")
    ap.add_argument("--residuals", default=None,
                    help="verify_split.py --out CSV, to restrict to castings "
                         "with no near-duplicate in training")
    ap.add_argument("--min-residual", type=float, default=5.7)
    ap.add_argument("--duplicates", default="rotation_duplicates.csv",
                    help="used to count distinct castings; skipped if absent")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    d = Path(args.data) / args.split / args.class_name
    if not d.is_dir():
        raise SystemExit(f"Missing {d}")
    here = {p.name for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if not here:
        raise SystemExit(f"No images in {d}")

    pred_path = Path(args.predicted)
    if not pred_path.is_file():
        raise SystemExit(f"Missing {pred_path}\n"
                         "Run:  python apply_defect_types.py")
    P, by_name, roots = {}, {}, set()
    with open(pred_path, newline="") as f:
        for row in csv.DictReader(f):
            k = tail(row["file"])
            P[k] = {fl: float(row[f"p_{fl}"]) for fl in FLAGS}
            by_name.setdefault(k.rsplit("/", 1)[-1], []).append(k)
            parts = Path(row["file"].replace("\\", "/")).parts
            if len(parts) >= 4:
                roots.add(parts[-4])
    # Pointed at the folder predicted.csv was built from (data_clean), the
    # split folder IS the source split, so split/class/name is exact and no
    # filename is ambiguous. Pointed anywhere else (data_grouped), it is not.
    direct = Path(args.data).resolve().name in roots

    keep = set(here)
    dropped = 0
    if args.residuals:
        rp = Path(args.residuals)
        if not rp.is_file():
            raise SystemExit(f"Missing {rp}\nRun: python verify_split.py --out {rp}")
        near = set()
        with open(rp, newline="") as f:
            for row in csv.DictReader(f):
                if float(row["residual"]) < args.min_residual:
                    near.add(Path(row["val_file"].replace("\\", "/")).name)
        before = len(keep)
        keep -= near
        dropped = before - len(keep)

    rows, ambiguous = [], 0
    for nm in sorted(keep):
        src = (f"{args.split}/{args.class_name}/{nm}" if direct
               else source_of(nm, args.class_name, by_name))
        if src is None and len(by_name.get(nm, [])) > 1:
            ambiguous += 1
        elif src in P:
            rows.append((src, P[src]))
    missing = len(keep) - len(rows) - ambiguous
    n = len(rows)
    if not n:
        raise SystemExit("No castings matched between the folder and "
                         f"{pred_path}. Was predicted.csv built from a "
                         "different dataset?")

    print(f"Castings : {len(here)} in {d}")
    if args.residuals:
        print(f"           -{dropped} with a training image closer than "
              f"{args.min_residual} (possible duplicates)")
    if ambiguous:
        print(f"           -{ambiguous} whose filename belongs to two different source")
        print("            images (split made before the collision fix), skipped")
    if missing:
        print(f"           -{missing} not present in {pred_path.name}")
    print(f"           {n} images in the breakdown")

    # intervals are computed at n_eff; the counts and rates stay per image
    n_eff = n
    if Path(args.duplicates).is_file():
        lo_c, hi_c = distinct_castings([nm for nm, _ in rows], args.duplicates)
        n_eff = lo_c
        print(f"           = between {lo_c} and {hi_c} distinct castings "
              "(rotations of one")
        print("             casting sit together in val). Intervals below use "
              f"{lo_c},")
        print("             the cautious end, not the image count.")
    else:
        print(f"           ({args.duplicates} not found: intervals treat every "
              "image as")
        print("            independent, which makes them too narrow)")
    print()

    t = args.threshold
    print("=" * 66)
    print(f"  PREDICTED defect types  (p >= {t})")
    print("=" * 66)
    print(f"    {'flag':<8}{'images':>8}{'rate':>9}{'95% CI':>16}{'AUC':>7}"
          f"{'known bias':>12}")
    for fl in FLAGS:
        c = sum(1 for _, p in rows if p[fl] >= t)
        lo, hi = wilson(round(c * n_eff / n), n_eff)
        b = KNOWN_BIAS[fl]
        print(f"    {fl:<8}{c:>8}{100*c/n:>8.1f}%"
              f"{f'{100*lo:.0f} to {100*hi:.0f}':>16}{AUC[fl]:>7.2f}"
              f"{b:>+11.1f}")
    print()
    print("    'known bias' is how far this flag ran high or low against the")
    print("    hand labels, in points. chip and slash read high; subtract")
    print("    roughly that much before quoting them.")

    counts = Counter(sum(1 for fl in FLAGS if p[fl] >= t) for _, p in rows)
    print()
    print("=" * 66)
    print("  HOW MANY DEFECTS PER CASTING")
    print("=" * 66)
    for k in sorted(counts):
        lo, hi = wilson(round(counts[k] * n_eff / n), n_eff)
        print(f"    {k} of 4{counts[k]:>8}{100*counts[k]/n:>8.1f}%"
              f"{f'   [{100*lo:.0f} to {100*hi:.0f}]':>18}")
    multi = sum(v for k, v in counts.items() if k >= 2)
    none = counts.get(0, 0)
    print()
    print(f"    {100*multi/n:.0f}% carry two or more at once.")
    print(f"    {100*none/n:.0f}% are predicted to have none of the four, "
          "though every")
    print("    one of them is labelled defective in the dataset.")

    print()
    print("=" * 66)
    print("  MOST COMMON COMBINATIONS")
    print("=" * 66)
    combos = Counter(tuple(fl for fl in FLAGS if p[fl] >= t) for _, p in rows)
    for combo, c in combos.most_common(8):
        name = " + ".join(combo) if combo else "(none predicted)"
        print(f"    {name:<28}{c:>6}{100*c/n:>8.1f}%")

    print()
    print("  Every figure above is a model estimate, not an inspection count.")
    print("  fray is the one to trust (AUC 1.00); chip the least (0.86 on 29")
    print("  positive training examples).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
