"""Find castings that are the same physical part photographed at a different
rotation.

    python check_rotations.py
    python check_rotations.py --data data_clean --threshold 15
    python check_rotations.py --classes def_front        (defects only, faster)

Why this exists
---------------
check_leakage.py compares SHA-256 hashes, so it finds images that are
byte-identical. It found 64 of those. It cannot find the same casting rotated:
rotate an image and every byte changes, so a hash comparison says "different
image" about what is plainly the same part.

That distinction matters three ways:

1. LEAKAGE. If a validation casting is a rotated copy of a training casting,
   the model has effectively seen it. Reported accuracy is inflated by however
   many of those exist, and the hash-based fix did not remove them.

2. EFFECTIVE SAMPLE SIZE. 4,211 defect images is not 4,211 castings if some
   are rotations of each other. Every statistic computed over them -- base
   rates, group comparisons, p-values -- assumes independent parts.

3. WHAT AUGMENTATION MEANS HERE. If the published set already contains
   rotations, then rotation augmentation during training is partly duplicating
   something the data already does, and the augmentation experiment's result
   should be read with that in mind.

How it works
------------
Stage 1, screening. Each image is resampled into polar coordinates about its
centre. Rotating a circular part is a circular SHIFT along the angle axis, and
the magnitude of an FFT is unchanged by a circular shift -- so |FFT| along the
angle axis is a rotation-invariant fingerprint. Comparing fingerprints finds
candidate pairs fast, without trying every angle.

Screening alone is not enough: these are mass-produced identical parts, so two
DIFFERENT castings still score high. On a test set the true pair led the list
by only 0.007. So:

Stage 2, verification. For each candidate, phase-correlate the two polar images
along the angle axis to recover the rotation in one operation, rotate one image
by it, and measure the mean absolute difference. Same part -> the defects land
on top of each other and the residual collapses. Different parts -> both sets
of defects survive and the residual stays high. On the known pair that was
10.5 against 27-30 for controls, which is a wide margin, not a judgement call.

Calibration, and why the first version of this was wrong
--------------------------------------------------------
The first run used a fixed threshold of 15 and grouped with single-linkage
union-find. It reported 358 groups covering 7,194 images, one of them with 474
members, and claimed 696 of 715 validation images were duplicates. All of that
was wrong, in a way worth writing down because it is a general trap.

At threshold 15 the per-pair false-positive rate is about 1.3% -- small. But
~42,000 candidate pairs were verified, so roughly 550 false links appeared. And
union-find takes the TRANSITIVE closure: one false link between two real
clusters merges both. A few hundred false links chained thousands of distinct
castings into mega-groups. The fingerprint was fine and the alignment was fine;
the grouping step was wrong.

This version fixes it three ways:

* It measures its own null. It samples random pairs WITHIN each class and
  reports the residual distribution between castings that are merely the same
  part design. On the test sample that null ran 11.7 at minimum and 34.8 at the
  median, while confirmed duplicates sat at 2.7, 5.3 and 9.4. The threshold has
  to sit below the null's floor, not in the middle of it.
* Separate nulls per class. Two good castings have no distinguishing marks and
  legitimately look near-identical, so ok_front needs a stricter bar than
  def_front. One global threshold cannot serve both.
* COMPLETE linkage, not single. A group forms only when every pair inside it
  verifies. Chaining then cannot happen by construction, which is what stops
  one bad link from swallowing a thousand castings.

Pairs are the primary output; groups are secondary.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NR, NA, KEEP = 48, 180, 24          # polar radii, angles, FFT bins kept
VERIFY_SIZE = 128                    # px for the residual check


def collect(data, splits, classes):
    out = []
    for split in splits:
        for cls in classes:
            d = Path(data) / split / cls
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    out.append((p, split, cls))
    return out


def polar_of(path, nr=NR, na=NA):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    sz = min(a.shape)
    a = a[:sz, :sz]
    c = (sz - 1) / 2.0
    rs = np.linspace(0.08, 0.95, nr) * c
    th = np.linspace(0, 2 * np.pi, na, endpoint=False)
    R, T = np.meshgrid(rs, th, indexing="ij")
    ys = np.clip((c + R * np.sin(T)).round().astype(np.int32), 0, sz - 1)
    xs = np.clip((c + R * np.cos(T)).round().astype(np.int32), 0, sz - 1)
    p = a[ys, xs]
    return p - p.mean(axis=1, keepdims=True)


def descriptor(polar, keep=KEEP):
    mag = np.abs(np.fft.rfft(polar, axis=1))[:, :keep].ravel()
    n = np.linalg.norm(mag)
    return (mag / n).astype(np.float32) if n else mag.astype(np.float32)


def rotation_between(fa, fb, na=NA):
    """Angle, in degrees, that best aligns b onto a -- recovered in one step by
    phase-correlating the polar images along the angle axis, instead of
    rotating through 360 trials."""
    cross = fa * np.conj(fb)
    corr = np.fft.irfft(cross, n=na, axis=1).sum(axis=0)
    return float(np.argmax(corr)) * 360.0 / na


def residual(path_a, path_b, angle, size=VERIFY_SIZE):
    A = Image.open(path_a).convert("L").resize((size, size), Image.BILINEAR)
    B = Image.open(path_b).convert("L").resize((size, size), Image.BILINEAR)
    B = B.rotate(-angle, resample=Image.BICUBIC)
    an = np.asarray(A, np.float32)
    bn = np.asarray(B, np.float32)
    y, x = np.ogrid[:size, :size]
    c = (size - 1) / 2.0
    m = ((x - c) ** 2 + (y - c) ** 2) <= (0.93 * c) ** 2
    return float(np.abs(an[m] - bn[m]).mean())


def sample_null(files, ffts, idx, n_pairs, rng):
    """Residuals between RANDOM pairs of the same class -- castings that share
    a part design but are not the same object. The threshold has to sit below
    this distribution's floor."""
    if len(idx) < 2:
        return np.array([])
    out = []
    for _ in range(n_pairs):
        i, j = rng.choice(idx, 2, replace=False)
        ang = rotation_between(ffts[i], ffts[j])
        out.append(residual(files[i][0], files[j][0], ang))
    return np.array(out)


def complete_link(pairs, n):
    """Groups where EVERY internal pair verified. Single-linkage lets one bad
    edge chain two clusters together; this cannot."""
    adj = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j); adj[j].add(i)
    groups, used = [], set()
    for seed in sorted(adj, key=lambda k: -len(adj[k])):
        if seed in used:
            continue
        clique = [seed]
        for cand in sorted(adj[seed], key=lambda k: -len(adj[k])):
            if cand in used:
                continue
            if all(cand in adj[m] for m in clique):
                clique.append(cand)
        if len(clique) > 1:
            groups.append(sorted(clique))
            used.update(clique)
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_clean")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--classes", default="def_front,ok_front")
    ap.add_argument("--neighbours", type=int, default=6)
    ap.add_argument("--threshold", default="auto",
                    help="residual below which a pair is the same part. "
                         "'auto' puts it under the measured null's floor.")
    ap.add_argument("--null-pairs", type=int, default=400,
                    help="random same-class pairs used to calibrate")
    ap.add_argument("--labels", default="defect_types/blind_key_r2.csv")
    ap.add_argument("--out", default="rotation_duplicates.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    files = collect(args.data, args.splits.split(","), args.classes.split(","))
    if not files:
        raise SystemExit(f"No images under {args.data}/")
    n = len(files)
    print(f"Images   : {n}")
    for cls in args.classes.split(","):
        for split in args.splits.split(","):
            c = sum(1 for _, s, k in files if s == split and k == cls)
            if c: print(f"           {split}/{cls}: {c}")
    print()

    print("Stage 1: rotation-invariant fingerprints")
    polars = np.empty((n, NR, NA), dtype=np.float32)
    desc = np.empty((n, NR * KEEP), dtype=np.float32)
    for i, (p, _, _) in enumerate(files):
        polars[i] = polar_of(p)
        desc[i] = descriptor(polars[i])
        if (i + 1) % 250 == 0:
            print(f"\r  {i+1}/{n}", end="", flush=True)
    print(f"\r  {n}/{n}")
    ffts = np.fft.rfft(polars, axis=2)
    del polars

    # ---------------- calibrate against the data's own null ----------------
    print("\nStage 2: measuring the null -- random pairs of DIFFERENT castings")
    nulls, floors = {}, {}
    for cls in args.classes.split(","):
        idx = [i for i in range(n) if files[i][2] == cls]
        if len(idx) < 2:
            continue
        v = sample_null(files, ffts, idx, min(args.null_pairs, len(idx) * 2), rng)
        nulls[cls] = v
        floors[cls] = float(v.min())
        print(f"  {cls:<10} n={len(v):<5} min {v.min():5.1f}   "
              f"1st pct {np.percentile(v,1):5.1f}   median {np.percentile(v,50):5.1f}")
    if not nulls:
        raise SystemExit("Could not calibrate.")

    if args.threshold == "auto":
        thr = {c: max(2.0, floors[c] * 0.80) for c in floors}
        print("\n  auto threshold = 80% of each class's null floor:")
    else:
        t = float(args.threshold)
        thr = {c: t for c in floors}
        print(f"\n  threshold fixed at {t} for every class:")
    for c, t in thr.items():
        below = 100.0 * float((nulls[c] < t).mean())
        print(f"    {c:<10} {t:5.1f}   flags {below:.2f}% of known-different pairs")
    print("\n  Any non-zero rate here becomes false PAIRS. Complete-linkage")
    print("  grouping keeps them from chaining into false GROUPS.")

    print("\nStage 3: shortlisting candidates")
    k = min(args.neighbours + 1, n)
    cand = set()
    BL = 512
    for s0 in range(0, n, BL):
        block = desc[s0:s0 + BL] @ desc.T
        for r in range(block.shape[0]):
            i = s0 + r
            block[r, i] = -1
            for j in np.argpartition(block[r], -k)[-k:]:
                if i != int(j):
                    cand.add((min(i, int(j)), max(i, int(j))))
        print(f"\r  {min(s0+BL,n)}/{n}", end="", flush=True)
    print(f"\r  {n}/{n}   {len(cand)} candidate pairs")

    print("\nStage 4: verifying each candidate at its recovered rotation")
    confirmed, resids = [], []
    cand = sorted(cand)
    for c, (i, j) in enumerate(cand):
        ang = rotation_between(ffts[i], ffts[j])
        res = residual(files[i][0], files[j][0], ang)
        resids.append(res)
        ci, cj = files[i][2], files[j][2]
        limit = min(thr.get(ci, 1e9), thr.get(cj, 1e9))
        if res < limit:
            confirmed.append((res, ang, i, j))
        if (c + 1) % 500 == 0:
            print(f"\r  {c+1}/{len(cand)}", end="", flush=True)
    print(f"\r  {len(cand)}/{len(cand)}")

    rv = np.array(resids)
    print("\n  Candidate residuals, against the null for comparison:")
    lo = min(rv.min(), min(v.min() for v in nulls.values()))
    hi = max(np.percentile(rv, 99), max(np.percentile(v, 99) for v in nulls.values()))
    edges = np.linspace(lo, hi, 17)
    hc, _ = np.histogram(rv, bins=edges)
    hn, _ = np.histogram(np.concatenate(list(nulls.values())), bins=edges)
    peak = max(hc.max(), 1)
    npeak = max(hn.max(), 1)
    print(f"    {'residual':>13}  {'candidates':<24} {'null':<20}")
    for a, b, c1, c2 in zip(edges[:-1], edges[1:], hc, hn):
        mark = " *" if any(a <= t < b for t in thr.values()) else ""
        print(f"    {a:5.1f}-{b:5.1f}  {'#'*int(22*c1/peak):<24}"
              f"{'.'*int(18*c2/npeak):<20}{mark}")
    print("    (# candidates, . null, * threshold)")
    print("\n  A separate spike of candidates BELOW where the null starts is")
    print("  what a real duplicate population looks like. Candidates sitting")
    print("  inside the null are just the same part design.")

    print(f"\n  Confirmed pairs: {len(confirmed)}")

    groups = complete_link([(i, j) for _, _, i, j in confirmed], n)
    in_group = sum(len(g) for g in groups)
    print()
    print("=" * 66)
    print(f"  images                         {n}")
    print(f"  confirmed duplicate pairs      {len(confirmed)}")
    print(f"  duplicate groups               {len(groups)}")
    print(f"  images that are a repeat       {in_group - len(groups)}"
          f"   ({100*(in_group-len(groups))/n:.2f}%)")
    print(f"  distinct castings (est.)       {n - (in_group - len(groups))}")
    print("=" * 66)
    if groups:
        sz = defaultdict(int)
        for g in groups: sz[len(g)] += 1
        print("  group sizes: " + ", ".join(f"{k}x{v}" for k, v in sorted(sz.items())))
        if max(len(g) for g in groups) > 8:
            print("  NOTE: a group larger than ~8 is worth eyeballing. Real")
            print("  rotated sets are small; a big one suggests the threshold")
            print("  is still too loose for this class.")

    crossing = [g for g in groups if len({files[i][1] for i in g}) > 1]
    print()
    if crossing:
        val_hit = sorted({i for g in crossing for i in g if files[i][1] == "val"})
        nval = sum(1 for _, s, _ in files if s == "val")
        print(f"  *** {len(crossing)} groups span train and val ***")
        print(f"  Validation images affected: {len(val_hit)}/{nval}"
              f" = {100*len(val_hit)/nval:.2f}%")
        print("  These are the same casting on both sides of the split, which a")
        print("  hash comparison cannot see. Held-out accuracy is inflated by")
        print("  whatever they contribute.")
        by = defaultdict(int)
        for i in val_hit: by[files[i][2]] += 1
        print("  by class: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    else:
        print("  No group spans train and val. The split is clean on this axis")
        print("  and reported accuracy needs no adjustment for rotation.")

    lab = Path(args.labels)
    if lab.is_file():
        import csv
        want = {Path(r["file"]).name
                for r in csv.DictReader(open(lab, newline=""))}
        idx = [i for i in range(n) if files[i][0].name in want]
        gof = {}
        for gi, g in enumerate(groups):
            for i in g: gof[i] = gi
        seen, collide = set(), 0
        for i in idx:
            key = gof.get(i, ("solo", i))
            if key in seen: collide += 1
            seen.add(key)
        print()
        print(f"  Hand-labelled castings: {len(idx)}, of which {collide} repeat")
        print(f"  another in the same sample -> effective n = {len(idx)-collide}.")

    with open(args.out, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["group", "residual", "angle", "file_a", "split_a",
                    "file_b", "split_b", "class"])
        gof = {}
        for gi, g in enumerate(groups, 1):
            for i in g: gof[i] = gi
        for res, ang, i, j in sorted(confirmed):
            w.writerow([gof.get(i, ""), f"{res:.2f}", f"{ang:.1f}",
                        files[i][0], files[i][1], files[j][0], files[j][1],
                        files[i][2]])
    print(f"\n  {args.out}   (one row per confirmed pair, closest first)")
    if confirmed:
        print("\n  Closest pairs -- check a few by eye before trusting any of it:")
        for res, ang, i, j in sorted(confirmed)[:5]:
            print(f"    {res:5.1f} at {ang:5.1f} deg   {files[i][0].name} / "
                  f"{files[j][0].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
