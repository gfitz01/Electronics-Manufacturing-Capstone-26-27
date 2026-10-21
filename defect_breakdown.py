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

    fray  1.00      slash 0.94      spot  0.92      chip  0.86

So the fray column is close to reliable and the chip column is the shakiest.
Quote these as estimates with that caveat attached, never as counts.

Known biases, and why slash needs watching
------------------------------------------
Training used pos_weight to stop the rarer defects being ignored, which pushes
the model toward calling them present. Measured against the hand labels
reweighted to the full defect set (the 28-epoch model of 28 Sept):

    slash  +16.4      chip  +4.4      spot  +2.8      fray  -2.7

Slash is the one to distrust as a rate: the model calls it on 47.7% of
defective castings where the hand labels imply about 31%. Its AUC is high
(0.94), so it RANKS castings well -- it is the 0.5 cut that sits too low, not
the model's judgement. Quote slash with the lean removed, or not as a rate at
all.

These numbers move when stage two is retrained. Recompute them from
apply_defect_types.py's "ESTIMATED defect rates" table whenever you do.

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
import json
import sys
from collections import Counter
from math import sqrt
import re
from pathlib import Path

FLAGS = ["fray", "chip", "spot", "slash"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# share predicted present over all 4,211 def_front (p >= 0.5) minus the hand
# labels' rate reweighted to the population (group sizes in blind_meta_r2.json).
# From the 28 Sept stage-two model, as apply_defect_types.py printed them:
#   fray 34.1 - 36.8   chip 16.9 - 12.5   spot 54.4 - 51.6   slash 47.7 - 31.3
KNOWN_BIAS = {"fray": -2.7, "chip": +4.4, "spot": +2.8, "slash": +16.4}
AUC = {"fray": 1.00, "chip": 0.86, "spot": 0.92, "slash": 0.94}


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
    ap.add_argument("--include-labelled", action="store_true",
                    help="keep the castings you hand-labelled. They are left "
                         "out by default: stage two trained on them, so their "
                         "predictions are memory rather than judgement, and "
                         "they flatter every rate by about a point.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="one cutoff for every flag, overriding the tuned "
                         "ones (0.5 is what the old runs used)")
    ap.add_argument("--thresholds",
                    default="defect_types/stage_two/thresholds.json",
                    help="per-flag cutoffs measured by train_defect_types.py")
    ap.add_argument("--cut", choices=["rate", "f1", "half"], default="rate",
                    help="which measured cutoff to use: 'rate' matches the "
                         "labelled rate (right for reporting rates, the "
                         "default), 'f1' balances catching defects against "
                         "false alarms (right for judging one casting), "
                         "'half' is the old flat 0.5")
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
    P, by_name, roots, hand = {}, {}, set(), set()
    with open(pred_path, newline="") as f:
        for row in csv.DictReader(f):
            k = tail(row["file"])
            P[k] = {fl: float(row[f"p_{fl}"]) for fl in FLAGS}
            if row.get("hand_labelled") == "1":
                hand.add(k)
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

    rows, ambiguous, trained_on = [], 0, 0
    for nm in sorted(keep):
        src = (f"{args.split}/{args.class_name}/{nm}" if direct
               else source_of(nm, args.class_name, by_name))
        if src is None and len(by_name.get(nm, [])) > 1:
            ambiguous += 1
        elif src in P:
            if src in hand and not args.include_labelled:
                trained_on += 1
                continue
            rows.append((src, P[src]))
    missing = len(keep) - len(rows) - ambiguous - trained_on
    n = len(rows)
    if not n:
        raise SystemExit("No castings matched between the folder and "
                         f"{pred_path}. Was predicted.csv built from a "
                         "different dataset?")

    print(f"Castings : {len(here)} in {d}")
    if args.residuals:
        print(f"           -{dropped} with a training image closer than "
              f"{args.min_residual} (possible duplicates)")
    if trained_on:
        print(f"           -{trained_on} you hand-labelled, so stage two "
              "trained on them")
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

    # Cutoffs: the ones train_defect_types.py measured, unless told otherwise.
    tuned = None
    if args.threshold is not None:
        T = {fl: args.threshold for fl in FLAGS}
        source = f"one cutoff of {args.threshold} for every flag, as you asked"
    else:
        tp = Path(args.thresholds)
        try:
            tuned = json.loads(tp.read_text())["flags"]
            key = {"rate": "threshold_rate", "f1": "threshold_f1",
                   "half": None}[args.cut]
            if key is None:
                T = {fl: 0.5 for fl in FLAGS}
            else:
                T = {fl: float(tuned[fl].get(key, tuned[fl]["threshold"]))
                     for fl in FLAGS}
            source = (f"'{args.cut}' cutoffs from {tp}" if key else
                      f"0.5 for every flag, with gaps from {tp}")
        except (OSError, ValueError, KeyError):
            T = {fl: 0.5 for fl in FLAGS}
            tuned = None
            source = (f"0.5 for every flag -- no {tp} yet, so the 'lean' "
                      "column is the stored correction")

    print("=" * 66)
    print("  PREDICTED defect types")
    print("=" * 66)
    print(f"    {source}")
    print()
    # The held-out score, if a labelling round has been scored. AUC and the
    # gap column both come from the 200 the model trained on; this column is
    # the only one measured on castings it never saw.
    held = {}
    try:
        hs = json.loads((Path(args.predicted).parent
                         / "holdout_score.json").read_text())
        held = hs.get("flags", {})
        held_n = hs.get("castings")
    except (OSError, ValueError):
        pass

    head = "gap on 200" if tuned else "known bias"
    hcol = f"{'held out':>11}" if held else ""
    print(f"    {'flag':<8}{'cut':>5}{'images':>8}{'rate':>9}{'95% CI':>16}"
          f"{'AUC':>7}{head:>12}{hcol}")
    for fl in FLAGS:
        c = sum(1 for _, p in rows if p[fl] >= T[fl])
        lo, hi = wilson(round(c * n_eff / n), n_eff)
        if tuned:
            b = {"rate": tuned[fl].get("gap_at_threshold"),
                 "f1": (tuned[fl].get("at_f1_cut") or {}).get("gap"),
                 "half": tuned[fl].get("gap_at_half")}[args.cut]
            b = 0.0 if b is None else b
        else:
            b = KNOWN_BIAS[fl]
        h = ""
        if held:
            a = held.get(fl, {}).get("agreement")
            h = f"{a:>10.1f}%" if a is not None else f"{'--':>11}"
        print(f"    {fl:<8}{T[fl]:>5.2f}{c:>8}{100*c/n:>8.1f}%"
              f"{f'{100*lo:.0f} to {100*hi:.0f}':>16}{AUC[fl]:>7.2f}"
              f"{b:>+11.1f}{h}")
    print()
    if held:
        print(f"    'held out' is how often the model agreed with your labels")
        print(f"    on {held_n} castings it never trained on -- the only column")
        print("    here not measured on the 200 it learned from. Run")
        print("    validate_defect_types.py to refresh it.")
        weak = [fl for fl in FLAGS
                if (held.get(fl, {}).get("precision") or 1) < 0.6]
        if weak:
            print()
            for fl in weak:
                d = held[fl]
                print(f"    {fl}: {d['caught']} of {d['caught']+d['false_alarm']}"
                      f" calls were real on the held-out set, so most of the "
                      f"{sum(1 for _, p in rows if p[fl] >= T[fl])} above")
                print(f"    {' '*len(fl)}  are false alarms. Quote it as a "
                      f"ranking, not a count.")
        print()
    if tuned:
        print("    Each cutoff was chosen on castings the model did not train")
        print("    on, and the last column is what the rate still misses the")
        print("    hand labels by, in points, at that cutoff. Small numbers")
        print("    there mean these rates can be quoted as they stand.")
    else:
        print("    'known bias' is how far this flag ran high or low against")
        print("    the hand labels, in points, at a cutoff of 0.5. Subtract it")
        print("    before quoting a rate -- or run train_defect_types.py to")
        print("    replace the guesswork with measured cutoffs.")

    counts = Counter(sum(1 for fl in FLAGS if p[fl] >= T[fl]) for _, p in rows)
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
    combos = Counter(tuple(fl for fl in FLAGS if p[fl] >= T[fl])
                     for _, p in rows)
    for combo, c in combos.most_common(8):
        name = " + ".join(combo) if combo else "(none predicted)"
        print(f"    {name:<28}{c:>6}{100*c/n:>8.1f}%")

    print()
    print("  Every figure above is a model estimate, not an inspection count.")
    print("  fray is the one to trust (AUC 1.00); chip the least (0.86 on 29")
    print("  positive training examples). slash ranks well but over-calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
