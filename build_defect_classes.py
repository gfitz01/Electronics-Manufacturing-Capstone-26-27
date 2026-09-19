"""Turn the two-way defect split into stage-two training data.

    python build_defect_classes.py
    python build_defect_classes.py --names frayed_edge,surface_spot
    python build_defect_classes.py --dry-run

Why two classes and not four
----------------------------
discover_defect_types.py scored k=2 highest (silhouette 0.6068) and every
larger k lower, down to 0.4533 at k=8. Looking at k=4 by eye, only one of the
four groups was describable: clean rim, no fraying, localised nicks and spots
rather than torn edges. The other three were not separable from each other.

That is the same answer the silhouette score gave. There is one separable
defect mode in these features, not four. This script therefore re-clusters at
k=2 and builds the folders for that.

What it does
------------
1. Reloads defect_types/features.npy -- the 4096-d vectors already extracted.
   No GPU pass, no model needed, runs in seconds.
2. Re-runs the identical PCA + k-means at k=2.
3. Cross-tabulates the k=2 result against whatever is currently in
   cluster_assignments.csv, so you can see where the group you identified
   landed. If your group sits almost entirely inside one k=2 cluster, your
   eye and the silhouette score found the same structure.
4. Names the cluster that absorbed your group (--anchor-cluster) with the
   second name, the other one with the first.
5. Copies the images into defect_classes/{train,val}/<name>/, preserving the
   original train/val boundary so stage two cannot leak.
6. Writes defect_types/contact_sheet_k2.html so you can check the naming.

IMPORTANT -- what a stage-two model trained on this would and would not prove
-----------------------------------------------------------------------------
These labels are the model's own opinion, not ground truth. Train a classifier
on them and it will score very high, because it is being asked to reproduce a
grouping that came out of its own features. That number is circular and means
nothing on its own.

The honest version is label_check.py: label a blind sample by hand, then
measure how often your judgement agrees with the cluster. If agreement is
high, the split corresponds to something physically real and stage two is
worth building. If it is near chance, the clusters are an artefact and no
amount of training accuracy will fix that.

Run that before training anything.
"""

import argparse
import base64
import csv
import io
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# small helpers, mirroring discover_defect_types.py deliberately.
# Duplicated rather than imported so this script does not have to pull in
# torch and torchvision, which it has no use for.
# --------------------------------------------------------------------------

def defect_images(base, class_name):
    """Same order discover_defect_types.py used, so row i of features.npy
    still corresponds to paths[i]."""
    base = Path(base)
    found = []
    for split in ("train", "val"):
        directory = base / split / class_name
        if directory.is_dir():
            found.extend(sorted(
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
    return found


def thumbnail_data_uri(path, size=132, quality=72):
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def split_of(path, data_root):
    """'train' or 'val', read off the path rather than guessed."""
    try:
        rel = Path(path).resolve().relative_to(Path(data_root).resolve())
    except ValueError:
        return "train"
    return rel.parts[0] if rel.parts else "train"


# --------------------------------------------------------------------------
# contact sheet, k=2, with names attached
# --------------------------------------------------------------------------

def write_contact_sheet(out_path, clusters, paths, names, per_cluster, crosstab):
    blocks = []
    for label in sorted(clusters):
        members = clusters[label]
        shown = members[:per_cluster]
        thumbs = "\n".join(
            f'<figure><img src="{thumbnail_data_uri(paths[i])}" '
            f'alt="{paths[i].name}" loading="lazy">'
            f'<figcaption>{paths[i].name}</figcaption></figure>'
            for i in shown)
        blocks.append(f"""<section>
  <h2>{names[label]}</h2>
  <p class="count">cluster {label} &middot; {len(members)} castings &middot;
     showing the {len(shown)} closest to the centre (most typical)</p>
  <div class="grid">{thumbs}</div>
</section>""")

    rows = "\n".join(
        "<tr><td>" + str(old) + "</td>" +
        "".join(f"<td>{crosstab[old].get(new, 0)}</td>" for new in sorted(clusters)) +
        "</tr>"
        for old in sorted(crosstab))
    head = "".join(f"<th>{names[new]}</th>" for new in sorted(clusters))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Two defect groups</title>
<style>
  :root {{ color-scheme: light dark;
    --bg:#F1F2F4; --card:#fff; --ink:#15181C; --muted:#6B747D;
    --rule:#D3D7DC; --accent:#1F5E8C; }}
  @media (prefers-color-scheme: dark) {{ :root {{
    --bg:#131619; --card:#1B1F24; --ink:#E9EBEE; --muted:#8B949D;
    --rule:#2E343B; --accent:#6FAFD8; }} }}
  * {{ box-sizing:border-box }}
  body {{ background:var(--bg); color:var(--ink); margin:0;
    padding:32px 20px 72px; font:15px/1.6 system-ui, sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto }}
  h1 {{ font-size:30px; margin:0 0 6px; letter-spacing:-.02em }}
  .lede {{ color:var(--muted); margin:0 0 28px; max-width:70ch }}
  section {{ background:var(--card); border:1px solid var(--rule);
    border-radius:4px; padding:18px 18px 22px; margin-bottom:22px }}
  h2 {{ font-size:19px; margin:0 0 3px; color:var(--accent);
    font-family:ui-monospace,monospace }}
  .count {{ color:var(--muted); font-size:13px; margin:0 0 14px }}
  .grid {{ display:grid; gap:10px;
    grid-template-columns:repeat(auto-fill,minmax(118px,1fr)) }}
  figure {{ margin:0 }}
  figure img {{ width:100%; display:block; border-radius:3px;
    border:1px solid var(--rule); background:#000 }}
  figcaption {{ font:10px/1.4 ui-monospace,monospace; color:var(--muted);
    margin-top:4px; word-break:break-all }}
  table {{ border-collapse:collapse; font:13px ui-monospace,monospace }}
  td, th {{ padding:5px 16px 5px 0; text-align:left;
    border-bottom:1px solid var(--rule) }}
  .how {{ background:var(--card); border-left:3px solid var(--accent);
    padding:16px 18px; border-radius:0 4px 4px 0; margin-bottom:26px }}
  .how p {{ margin:0 0 10px; max-width:74ch }} .how p:last-child {{ margin:0 }}
</style></head><body><div class="wrap">

<h1>Two defect groups</h1>
<p class="lede">The split the features actually support. Silhouette score
preferred k=2 over every larger k, and inspecting k=4 by eye found only one
describable group &mdash; which is this same split seen through four labels.</p>

<div class="how">
  <p><strong>These are not verified labels.</strong> They are the model's own
  grouping of its internal features. Before training anything on them, run
  <code>label_check.py</code>: it asks you to sort a blind sample by hand and
  reports how often you agree with the cluster.</p>
  <p>High agreement means the grouping tracks something physically real.
  Agreement near chance means it does not, and no training accuracy would
  tell you otherwise, because a stage-two model is only reproducing the
  clustering it was trained on.</p>
</div>

<section>
  <h2>where the previous clusters went</h2>
  <p class="count">rows: the clusters in cluster_assignments.csv before this
  run &middot; columns: the two groups now</p>
  <table><thead><tr><th>was</th>{head}</tr></thead><tbody>{rows}</tbody></table>
</section>

{"".join(blocks)}

</div></body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data_clean")
    ap.add_argument("--class-name", default="def_front")
    ap.add_argument("--features", default="defect_types/features.npy")
    ap.add_argument("--assignments", default="defect_types/cluster_assignments.csv")
    ap.add_argument("--out", default="defect_classes")
    ap.add_argument("--sheet", default="defect_types/contact_sheet_k2.html")
    ap.add_argument("--names", default="frayed_edge,surface_spot",
                    help="two names, comma separated. The SECOND is given to "
                         "whichever group absorbed --anchor-cluster.")
    ap.add_argument("--anchor-cluster", type=int, default=2,
                    help="the cluster from the previous run that you could "
                         "describe. Defaults to 2, the clean-rim/spots group.")
    ap.add_argument("--per-cluster", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split and write the sheet, copy nothing")
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(names) != 2:
        raise SystemExit("--names needs exactly two comma-separated names")

    features_path = Path(args.features)
    if not features_path.is_file():
        raise SystemExit(
            f"Missing {features_path}\n"
            "Run:  python discover_defect_types.py")

    paths = defect_images(args.data, args.class_name)
    features = np.load(features_path)
    if len(paths) != features.shape[0]:
        raise SystemExit(
            f"{features_path} has {features.shape[0]} rows but "
            f"{args.data} has {len(paths)} defect images.\n"
            "The dataset changed since the features were extracted. "
            "Re-run:  python discover_defect_types.py")

    print(f"Defects : {len(paths)} images")
    print(f"Features: {features.shape}")
    print()

    # identical to discover_defect_types.py, so k=2 here is the same k=2 there
    n_components = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(features)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(reduced)
    labels = km.labels_

    clusters = {}
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        d = np.linalg.norm(reduced[idx] - km.cluster_centers_[label], axis=1)
        clusters[label] = list(idx[np.argsort(d)])

    # ---------------- where did the previous clustering go ----------------
    crosstab = defaultdict(Counter)
    previous = {}
    assignments = Path(args.assignments)
    if assignments.is_file():
        with open(assignments, newline="") as f:
            for row in csv.DictReader(f):
                previous[str(Path(row["file"]))] = int(row["cluster"])
        for i, p in enumerate(paths):
            old = previous.get(str(p))
            if old is not None:
                crosstab[old][int(labels[i])] += 1

    anchor_target = None
    if crosstab:
        print("Previous clusters vs the two groups:")
        print(f"  {'was':>5}  {'-> group 0':>11}  {'-> group 1':>11}   purity")
        for old in sorted(crosstab):
            c = crosstab[old]
            total = sum(c.values())
            top = c.most_common(1)[0]
            print(f"  {old:>5}  {c.get(0,0):>11}  {c.get(1,0):>11}   "
                  f"{100*top[1]/total:5.1f}% into group {top[0]}")
        if args.anchor_cluster in crosstab:
            anchor_target = crosstab[args.anchor_cluster].most_common(1)[0][0]
            share = (crosstab[args.anchor_cluster][anchor_target]
                     / sum(crosstab[args.anchor_cluster].values()))
            print()
            print(f"  Your cluster {args.anchor_cluster} is {100*share:.1f}% "
                  f"inside group {anchor_target}.")
            if share < 0.80:
                print("  That is a weak correspondence. The group you could "
                      "describe does not\n  survive the two-way split "
                      "cleanly -- check the sheet before training.")
        else:
            print(f"\n  Cluster {args.anchor_cluster} not present in "
                  f"{assignments}; naming by size instead.")
        print()

    if anchor_target is None:
        # fall back: the smaller group takes the first name
        anchor_target = 0 if len(clusters[0]) > len(clusters[1]) else 1

    naming = {anchor_target: names[1],
              1 - anchor_target: names[0]}

    print("Naming:")
    for label in sorted(clusters):
        print(f"  group {label}  ->  {naming[label]:<16} "
              f"{len(clusters[label]):>5} castings")
    print()

    # ---------------- build the folders ----------------
    counts = defaultdict(Counter)
    for i, p in enumerate(paths):
        counts[naming[int(labels[i])]][split_of(p, args.data)] += 1

    print("Stage-two dataset (original train/val boundary preserved):")
    for name in names:
        c = counts[name]
        print(f"  {name:<16} train {c['train']:>5}   val {c['val']:>4}")
    thin = [n for n in names if counts[n]["val"] < 30]
    if thin:
        print(f"\n  WARNING: {', '.join(thin)} has under 30 validation images. "
              "Accuracy on\n  a set that small has an error bar of roughly "
              "+/- 10 points.")
    print()

    if args.dry_run:
        print("--dry-run: no files copied.")
    else:
        out = Path(args.out)
        if out.exists():
            shutil.rmtree(out)
        for name in names:
            for split in ("train", "val"):
                (out / split / name).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(paths):
            dest = out / split_of(p, args.data) / naming[int(labels[i])] / p.name
            shutil.copy2(p, dest)
            if (i + 1) % 500 == 0:
                print(f"\r  copied {i+1}/{len(paths)}", end="", flush=True)
        print(f"\r  copied {len(paths)}/{len(paths)}")

    print("\nBuilding contact sheet ...")
    Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
    write_contact_sheet(args.sheet, clusters, paths, naming,
                        args.per_cluster, crosstab)

    print()
    print("=" * 64)
    print(f"  {args.sheet}   <- check the naming here")
    if not args.dry_run:
        print(f"  {args.out}/   <- stage-two ImageFolder layout")
    print("=" * 64)
    print()
    print("Next, and do this BEFORE training anything on these folders:")
    print()
    print("    python label_check.py --build")
    print()
    print("It makes a blind sample for you to sort by hand. Training a model")
    print("on clusters it produced itself proves nothing; agreeing with your")
    print("own eye on images you sorted blind is the result worth reporting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
