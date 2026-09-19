"""Build stage-two training folders from the hand labels.

    python make_defect_type_folders.py
    python make_defect_type_folders.py --dry-run
    python make_defect_type_folders.py --val-frac 0.25

The cascade this feeds
----------------------
    stage one   good  ->  done, nothing else to say about it
                bad   ->  pass it to stage two
    stage two   which of the four defects does this bad casting have

Only defective castings reach stage two, so only defective castings are in
here. That is not an oversight -- asking "what type of defect" about a good
casting has no answer.

Why four folders would be wrong, and what this builds instead
-------------------------------------------------------------
The obvious layout is one folder per defect type:

    defect_type_front/fray/  chip/  spot/  slash/

ImageFolder gives each image exactly one folder, so that layout says every
casting has exactly one defect. 37% of yours have two or more. Casting 5772
has a frayed edge AND a face spot; filing it under "fray" throws away the spot,
and filing it under both makes the model believe two contradictory things about
the same picture.

So this builds FOUR SEPARATE binary datasets instead:

    defect_type_front/
        fray/train/has_fray/      fray/train/no_fray/
        fray/val/has_fray/        fray/val/no_fray/
        chip/train/has_chip/      ... and so on for spot and slash

Each one is an ordinary two-class ImageFolder the existing AlexNet wrapper can
train with no code changes. Four models, one per question. A casting with
fraying and a spot lands in has_fray AND has_spot, which is the truth, and no
model is asked to choose.

The 15 castings where you saw nothing land in all four "no_" folders. That is
also the truth, and they are useful negatives.

The split keeps rotation families together
------------------------------------------
The dataset contains the same physical casting at several rotations. Put one
rotation in train and another in val and the val score is inflated -- that is
the flaw that made the original 715-image result meaningless. This uses
rotation_duplicates.csv to keep every family whole on one side, and uses the
SAME split for all four flags so a casting is never in train for one question
and val for another.

A warning about what this can measure
-------------------------------------
200 labelled castings split 80/20 leaves about 40 for validation. For spot
(48.5% positive) that is ~19 positives; for chip (14.5%) it is about 6. An
accuracy measured on 6 positives has an error bar of roughly +/- 35 points.

train_defect_types.py is the better measurement on this much data: it
cross-validates, so every casting is used for both training and testing across
folds, and it reports AUC rather than accuracy. Use these folders to train a
model you want to keep; use that script to find out whether it works.
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

FLAGS = ["fray", "chip", "spot", "slash"]


def norm(p):
    """The label CSVs store Windows paths. Backslashes work on Windows and
    forward slashes work everywhere, so normalise once and the scripts run
    on either."""
    return Path(str(p).replace("\\", "/"))


def load(key_path, labels_path):
    key = {}
    with open(key_path, newline="") as f:
        for row in csv.DictReader(f):
            key[int(row["id"])] = row["file"]
    items = []
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not set(FLAGS) <= cols:
            raise SystemExit(
                f"{labels_path} has columns {sorted(cols)}; this needs "
                f"id,{','.join(FLAGS)}. That is probably an earlier round.")
        for row in reader:
            i = int(row["id"])
            if i not in key:
                continue
            v = [row[f].strip() for f in FLAGS]
            if all(x in ("0", "1") for x in v):
                items.append((str(norm(key[i])), [int(x) for x in v]))
            elif all(x == "?" for x in v):
                items.append((str(norm(key[i])), [0, 0, 0, 0]))
    return items


def families(dup_path, files):
    parent = {f: f for f in files}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    if not Path(dup_path).is_file():
        return {f: f for f in files}, False
    byname = defaultdict(list)
    for f in files:
        byname[Path(f).name].append(f)
    with open(dup_path, newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = norm(row["file_a"]).name, norm(row["file_b"]).name
            if a in byname and b in byname:
                ra, rb = find(byname[a][0]), find(byname[b][0])
                if ra != rb:
                    parent[rb] = ra
    return {f: find(f) for f in files}, True


def choose_split(items, fam, val_frac, tries, seed):
    """Same split for all four flags, families kept whole, picked from a few
    random draws for the one that balances every flag best."""
    groups = defaultdict(list)
    for f, y in items:
        groups[fam[f]].append((f, y))
    keys = list(groups)
    target = val_frac * len(items)
    rng = random.Random(seed)
    best = None
    for _ in range(tries):
        rng.shuffle(keys)
        val, n = set(), 0
        for k in keys:
            if n >= target:
                break
            val.add(k); n += len(groups[k])
        # score: how far each flag's val rate drifts from its overall rate
        cost = abs(n - target) / max(target, 1)
        for k2 in range(len(FLAGS)):
            allr = sum(y[k2] for _, y in items) / len(items)
            vitems = [y for g in val for _, y in groups[g]]
            if vitems:
                cost += abs(sum(y[k2] for y in vitems) / len(vitems) - allr)
        if best is None or cost < best[0]:
            best = (cost, set(val))
    return best[1], groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default="defect_types/blind_key_r2.csv")
    ap.add_argument("--labels", default="defect_types/labels_r2.csv")
    ap.add_argument("--duplicates", default="rotation_duplicates.csv")
    ap.add_argument("--out", default="defect_type_front")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--tries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.labels).is_file():
        cands = sorted(Path("defect_types").glob("my_labels*.csv"))
        raise SystemExit(f"Missing {args.labels}\nCandidates: "
                         f"{[str(c) for c in cands] or 'none'}\nPass --labels")

    items = load(args.key, args.labels)
    if not items:
        raise SystemExit("No usable labels.")
    files = [f for f, _ in items]
    fam, have_dupes = families(args.duplicates, files)
    nfam = len(set(fam.values()))

    print(f"Labelled castings : {len(items)}")
    for k, f in enumerate(FLAGS):
        pos = sum(y[k] for _, y in items)
        print(f"    {f:<7} {pos:>4} with it   {len(items)-pos:>4} without"
              f"   ({100*pos/len(items):4.1f}%)")
    print(f"    none    {sum(1 for _, y in items if sum(y) == 0):>4} "
          "with no defect visible")
    print(f"    2+      {sum(1 for _, y in items if sum(y) >= 2):>4} "
          "with several at once")
    print()
    if have_dupes:
        print(f"Rotation families : {nfam} across {len(items)} castings"
              f"  ({len(items)-nfam} share one)")
        print("  Families are kept whole, so no casting is validated against")
        print("  a rotated copy of itself.")
    else:
        print(f"WARNING: no {args.duplicates}. The split cannot keep rotation")
        print("  families together, so val scores will be inflated. Run")
        print("  check_rotations.py first.")
    print()

    val_fams, groups = choose_split(items, fam, args.val_frac, args.tries, args.seed)
    split_of = {}
    for g, members in groups.items():
        s = "val" if g in val_fams else "train"
        for f, _ in members:
            split_of[f] = s

    ntr = sum(1 for f in files if split_of[f] == "train")
    nva = len(files) - ntr
    print(f"Split : {ntr} train / {nva} val\n")
    print(f"    {'flag':<8}{'train has':>11}{'train no':>10}"
          f"{'val has':>9}{'val no':>8}")
    thin = []
    for k, f in enumerate(FLAGS):
        th = sum(1 for p, y in items if split_of[p] == "train" and y[k])
        tn = sum(1 for p, y in items if split_of[p] == "train" and not y[k])
        vh = sum(1 for p, y in items if split_of[p] == "val" and y[k])
        vn = sum(1 for p, y in items if split_of[p] == "val" and not y[k])
        print(f"    {f:<8}{th:>11}{tn:>10}{vh:>9}{vn:>8}")
        if min(vh, vn) < 10:
            thin.append((f, min(vh, vn)))
    if thin:
        print()
        for f, c in thin:
            print(f"    WARNING: {f} has only {c} in its smaller val class.")
        print("    Accuracy on that few is worth roughly +/- 30 points. Treat")
        print("    these as something to train on, not something to measure on.")
        print("    train_defect_types.py cross-validates and is the measurement.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    copied = 0
    for k, f in enumerate(FLAGS):
        for split in ("train", "val"):
            for cls in (f"has_{f}", f"no_{f}"):
                (out / f / split / cls).mkdir(parents=True, exist_ok=True)
        for p, y in items:
            cls = f"has_{f}" if y[k] else f"no_{f}"
            dest = out / f / split_of[p] / cls / Path(p).name
            shutil.copy2(p, dest)
            copied += 1
    print(f"\nCopied {copied} files ({len(items)} castings x {len(FLAGS)} questions)")

    print()
    print("=" * 66)
    print("  Train one model per question, using the existing wrapper:")
    print("=" * 66)
    for f in FLAGS:
        print(f"    python run.py --data {out}\\{f} augment")
    print()
    print("  Each is an ordinary two-class problem, so nothing in the wrapper")
    print("  needs changing. Four models, four questions, and a casting with")
    print("  two defects is correctly a yes to two of them.")
    print()
    print("  Before trusting any of their val numbers, run:")
    print("      python train_defect_types.py")
    print("  On 200 castings, cross-validation over all of them beats a single")
    print("  40-image holdout by a wide margin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
