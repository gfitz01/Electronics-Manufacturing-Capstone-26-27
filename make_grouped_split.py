"""Build a train/val split where no casting appears on both sides.

    python make_grouped_split.py
    python make_grouped_split.py --dry-run
    python make_grouped_split.py --drop-contradictory

Why the existing split cannot be used
-------------------------------------
data_clean/ was built by removing images whose SHA-256 hash matched a training
image. That caught 64. It could not catch the same casting photographed at a
different angle, because rotating an image changes every byte.

check_rotations.py found the rest: 636 of the 715 validation castings have a
rotated twin in training. So the reported 99.72% / 100.00% describes a test set
the model had largely already seen. Splitting the 715 by whether a twin existed:

    had a twin in training    636    100.00%  /  100.00%
    genuinely unseen           79     97.47%  /  100.00%

Only the second row is a generalization estimate, and 79 castings is a thin one.

What this builds
----------------
data_grouped/, where every rotation family sits entirely on one side of the
split. A casting in validation has no copy of itself in training at any angle,
so the val score means what a val score is supposed to mean.

The split targets the original proportions -- about 10% validation, with the
same defective/good ratio -- but families are indivisible, so the counts land
near the targets rather than on them. That is the cost of a correct split and
it is small.

The contradictory pairs
----------------------
46 duplicate pairs span the two classes: filed as def_front in one image and
ok_front in the other. Their residuals run from 1.9 to 7.0 (median 4.3), and
that spread matters:

    residual < 3     7 pairs    visually identical -- certainly one casting
    residual < 5.7  37 pairs    very similar; probably, not certainly, one
    all             46 pairs    everything check_rotations.py passed

An earlier version of this note said all 46 sat at 1.9 to 2.9. That was wrong:
it described the lowest few, not the set. Quote "7 certain, up to 46" rather
than "46".

None of them land in data_grouped/val (checked against the run's
predictions.csv), so they cannot affect the validation score.

Counted at the PAIR level deliberately. A linkage group of 1,385 that happens
to contain one cross-class pair is not 1,385 contradictory castings, and
reporting it that way would badly overstate the problem.

These are not a splitting problem -- groups still go to one side -- but they
are a labelling problem. A model trained on both is shown one casting and told
two different answers, and it caps what any accuracy figure can mean: on those
castings, being "right" is undefined. It also explains part of the earlier
observation that 15 of 200 castings labelled defective showed no visible
defect -- some of them are also sitting in ok_front.

--drop-contradictory removes the images in those pairs. Off by default, because
deleting data to make a number look better is its own kind of dishonesty. Run
it both ways and report the difference.

Filename collisions
-------------------
The source dataset reuses filenames across its train and test folders for
DIFFERENT images: cast_ok_0_5815.jpeg in train is not the cast_ok_0_5815.jpeg
in test. The first version of this script keyed everything on the bare
filename, which went wrong twice:

  * duplicate pairs were attached to whichever image owned the name first,
    so some rotation links landed on the wrong image and twins went unlinked;
  * when two same-named images were assigned to the same folder, the second
    copy silently overwrote the first -- 65 training images vanished.

verify_split.py (which reads pixels, not names) found 9 validation castings
visually identical to a training image in that split, and 7 of the 9 involve
a collided filename. So most of the residual leakage was this bug.

Everything is now keyed on split/class/filename, and a collided name is
written as from-<split>__<name> so both images survive. The 20 Sept runs
(716/716 for both) used the old split; the result stands because both models
were right on all 707 castings that are not visually identical to training.

After this
----------
    python run.py --data data_grouped augment
    python run.py --data data_grouped

Two 100-epoch runs, same settings as before, so the numbers are comparable to
the originals. On the 20 Sept split both came out 716/716 -- not lower, as
first expected. The task is easy for this model once the leak is gone.
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ("def_front", "ok_front")


def norm(p):
    return Path(str(p).replace("\\", "/"))


def tail(p):
    """split/class/filename -- the only safe key. See 'Filename collisions'."""
    return "/".join(norm(p).parts[-3:])


def out_name(p, split, collided):
    """Filename inside data_grouped/. Unchanged unless another source image
    shares it, in which case the source split is prefixed so neither copy
    overwrites the other. defect_breakdown.py understands the prefix."""
    return f"from-{split}__{p.name}" if p.name in collided else p.name


def collect(data, splits, classes):
    out = []
    for split in splits:
        for cls in classes:
            d = Path(data) / split / cls
            if d.is_dir():
                out.extend((p, split, cls) for p in sorted(d.rglob("*"))
                           if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return out


def build_families(dup_path, files):
    """Union every confirmed duplicate pair into a linkage group.

    These are NOT one physical casting each. Union-find takes the transitive
    closure, so A-B and B-C merge into one blob even when A and C are not a
    pair; on the real data that produces groups of 400 and 1,385. The better
    estimate of distinct castings is check_rotations.py's complete-linkage
    count (2,092 groups, none larger than 7).

    Over-merging is deliberate here. For a split, the only thing that matters
    is that no duplicate relationship crosses the boundary, and a group that is
    too big can only ever be too cautious. Under-merging would let a twin
    through, which is the failure this whole exercise exists to fix.
    """
    bykey = {tail(p): i for i, (p, _, _) in enumerate(files)}
    parent = list(range(len(files)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    linked = 0
    if Path(dup_path).is_file():
        with open(dup_path, newline="") as f:
            for row in csv.DictReader(f):
                a = bykey.get(tail(row["file_a"]))
                b = bykey.get(tail(row["file_b"]))
                if a is None or b is None:
                    continue
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                    linked += 1
    fam = defaultdict(list)
    for i in range(len(files)):
        fam[find(i)].append(i)
    return fam, linked


def pick_val(fams, files, target_total, target_def, tries, seed, slack=1.06):
    """Whole groups into validation, aiming at the original proportions.

    Groups are indivisible and some are enormous (chaining), so a group that
    would overshoot the target is skipped rather than accepted -- otherwise one
    1,385-image blob lands in validation and the split is ruined. Several
    arrangements are sampled and the one closest on both targets is kept."""
    keys = list(fams)
    rng = random.Random(seed)
    best = None
    for _ in range(tries):
        rng.shuffle(keys)
        chosen, n, ndef = set(), 0, 0
        for k in keys:
            members = fams[k]
            if n + len(members) > target_total * slack:
                continue                      # too big to fit; leave in train
            chosen.add(k)
            n += len(members)
            ndef += sum(1 for i in members if files[i][2] == "def_front")
            if n >= target_total:
                break
        if n == 0:
            continue
        cost = abs(n - target_total) / target_total \
            + 2.0 * abs(ndef - target_def) / max(target_def, 1)
        if best is None or cost < best[0]:
            best = (cost, set(chosen), n, ndef)
    if best is None:
        raise SystemExit("Could not assemble a validation set: every group is "
                         "larger than the target. Lower the target or check "
                         "rotation_duplicates.csv.")
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_clean",
                    help="source of the images (their labels come from here)")
    ap.add_argument("--duplicates", default="rotation_duplicates.csv")
    ap.add_argument("--out", default="data_grouped")
    ap.add_argument("--drop-contradictory", action="store_true",
                    help="exclude the images in cross-class pairs -- the same "
                         "casting labelled both defective and good")
    ap.add_argument("--tries", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = collect(args.data, ("train", "val"), CLASSES)
    if not files:
        raise SystemExit(f"No images under {args.data}/")
    orig = Counter((s, c) for _, s, c in files)
    n = len(files)
    n_val = sum(v for (s, _), v in orig.items() if s == "val")
    n_val_def = orig[("val", "def_front")]

    print(f"Source : {args.data}  ({n} images)")
    for s in ("train", "val"):
        for c in CLASSES:
            print(f"    {s}/{c}: {orig[(s, c)]}")
    print()

    if not Path(args.duplicates).is_file():
        raise SystemExit(f"Missing {args.duplicates}\n"
                         "Run:  python check_rotations.py")
    fams, linked = build_families(args.duplicates, files)
    sizes = Counter(len(v) for v in fams.values())
    biggest = max(sizes) if sizes else 0
    print(f"Linkage groups : {len(fams)} groups over {n} images "
          f"({linked} linking pairs), largest {biggest}")
    print("    sizes : " + ", ".join(f"{k}x{v}" for k, v in
                                     sorted(sizes.items())[:8])
          + (" ..." if len(sizes) > 8 else ""))
    print("    These are linkage groups, not distinct castings: union-find")
    print("    chains A-B-C together even when A and C are not a pair. That is")
    print("    the safe direction for a split -- too cautious, never leaky.")
    print()

    # Contradictions are counted at the PAIR level. A chained group of 1,385
    # containing one cross-class pair is not 1,385 contradictory castings.
    bykey = {tail(p): i for i, (p, _, _) in enumerate(files)}
    contra_imgs = set()
    n_contra_pairs = 0
    contra_res = []
    with open(args.duplicates, newline="") as f:
        for row in csv.DictReader(f):
            a = bykey.get(tail(row["file_a"]))
            b = bykey.get(tail(row["file_b"]))
            if a is None or b is None:
                continue
            if files[a][2] != files[b][2]:
                n_contra_pairs += 1
                contra_imgs.update((a, b))
                contra_res.append(float(row["residual"]))
    if n_contra_pairs:
        certain = sum(1 for r in contra_res if r < 3.0)
        print(f"CONTRADICTORY : {n_contra_pairs} duplicate pairs "
              f"({len(contra_imgs)} images) are labelled")
        print("    def_front on one side and ok_front on the other. Residuals "
              f"{min(contra_res):.1f} to {max(contra_res):.1f}:")
        print(f"    {certain} below 3 (visually identical, certainly one casting),")
        print(f"    the other {n_contra_pairs - certain} very similar but not "
              "certainly the same.")
        print("    Where they are the same casting, the dataset's good/bad labels")
        print("    disagree with themselves.")
        if args.drop_contradictory:
            print(f"    --drop-contradictory: excluding those {len(contra_imgs)} "
                  "images.")
            for k in list(fams):
                fams[k] = [i for i in fams[k] if i not in contra_imgs]
                if not fams[k]:
                    del fams[k]
        else:
            print("    Keeping them. Re-run with --drop-contradictory to see")
            print("    what they cost -- and report both numbers.")
        print()

    kept = sorted({i for m in fams.values() for i in m})
    keptset = set(kept)
    target_total = round(n_val * len(keptset) / n)
    target_def = round(n_val_def * len(keptset) / n)

    val_keys = pick_val(fams, files, target_total, target_def, args.tries, args.seed)
    val_idx = {i for k in val_keys for i in fams[k]}

    counts = Counter()
    for i in kept:
        counts[("val" if i in val_idx else "train", files[i][2])] += 1
    nv = sum(v for (s, _), v in counts.items() if s == "val")

    print("New split (families kept whole):")
    print(f"    {'':<6}{'def_front':>12}{'ok_front':>11}{'total':>9}")
    for s in ("train", "val"):
        t = counts[(s, "def_front")] + counts[(s, "ok_front")]
        print(f"    {s:<6}{counts[(s,'def_front')]:>12}"
              f"{counts[(s,'ok_front')]:>11}{t:>9}")
    print(f"\n    validation is {100*nv/len(keptset):.1f}% of the data "
          f"(target {100*target_total/len(keptset):.1f}%), "
          f"{100*counts[('val','def_front')]/max(nv,1):.1f}% defective "
          f"(original {100*n_val_def/max(n_val,1):.1f}%)")

    # the guarantee, checked rather than asserted
    bad = 0
    for k, m in fams.items():
        sides = {"val" if i in val_idx else "train" for i in m}
        if len(sides) > 1:
            bad += 1
    print(f"\n    families split across train and val: {bad}  "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    if bad:
        raise SystemExit("Split is not group-aware. Not writing anything.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        for c in CLASSES:
            (out / s / c).mkdir(parents=True, exist_ok=True)
    name_count = Counter(files[i][0].name for i in kept)
    collided = {nm for nm, k in name_count.items() if k > 1}
    if collided:
        print(f"  {len(collided)} filenames are shared by different source images;"
              " those copies get a from-<split>__ prefix")
    for j, i in enumerate(kept):
        p, src_split, c = files[i]
        dest = out / ("val" if i in val_idx else "train") / c / \
            out_name(p, src_split, collided)
        if dest.exists():
            raise SystemExit(f"Refusing to overwrite {dest}")
        shutil.copy2(p, dest)
        if (j + 1) % 1000 == 0:
            print(f"\r  copied {j+1}/{len(kept)}", end="", flush=True)
    print(f"\r  copied {len(kept)}/{len(kept)}")

    print()
    print("=" * 70)
    print("  Retrain both runs on this split:")
    print("=" * 70)
    print(f"    python run.py --data {args.out} augment")
    print(f"    python run.py --data {args.out}")
    print()
    print("  Same settings as the originals, so the numbers are comparable.")
    print("  These are measured against castings the model has genuinely not")
    print("  seen. Then check the split directly:  python verify_split.py")
    print()
    print("  Then compare:")
    print("    python compare_runs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
