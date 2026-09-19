"""Discover defect types the binary model learned implicitly.

    python discover_defect_types.py
    python discover_defect_types.py --k 5
    python discover_defect_types.py --weights results/alexnet-aug-.../model_parameters.pth

Your model outputs two numbers: defective or good. But the layer BEFORE that
output is 4096 numbers wide, and a crack and a misshapen rim almost certainly
light up different parts of it. The final layer crushes that detail away; this
script reads it before it is lost.

What it does:

1. Rebuilds the trained model and strips off the final 2-way classifier, so it
   emits the 4096-dimensional feature vector instead of a verdict.
2. Runs every defective casting through it.
3. Reduces those vectors with PCA and clusters them with k-means.
4. Writes defect_types/contact_sheet.html -- each cluster shown as a grid of
   its most typical castings, so you can see at a glance whether the groups
   are visually coherent.
5. Writes defect_types/cluster_assignments.csv -- every image with its cluster,
   ready to become training labels for a second-stage model.

NOTHING HERE NEEDS LABELS. It is the cheap way to find out whether a
type-classifier is worth the labelling effort before anyone starts labelling.

Reading the output:

* Clusters that are visually coherent (all rim damage, all surface porosity)
  mean the model learned separable defect modes. Name them, correct the
  strays, and you have stage-two training data for a fraction of the work.
* Clusters that look like random mixtures mean the model found one generic
  "wrongness" feature rather than type-specific ones. That is a real finding
  too, and it saves you from labelling 3,700 images for nothing.
"""

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def find_latest_weights(results="results"):
    """Newest run directory that actually saved weights."""
    root = Path(results)
    if not root.is_dir():
        raise SystemExit(f"No {root}/ directory. Train a model first.")
    candidates = sorted(root.glob("*/model_parameters.pth"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No model_parameters.pth under {root}/. Train a model and answer "
            "'y' when it asks to save the weights.")
    return candidates[-1]


def load_feature_extractor(weights_path, num_classes=2, device="cpu"):
    """Rebuild the trained AlexNet, then cut off its final classifier layer.

    The remaining network outputs the 4096-dim vector that fed the verdict.
    """
    model = alexnet(weights=None)
    model.classifier[-1] = nn.Linear(4096, num_classes)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # everything except the last Linear
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    return model.to(device)


def build_transform(size=224, channels=3):
    # identical to the VALIDATION transform -- no augmentation, or the
    # features would describe the random transform rather than the casting
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize((size, size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def defect_images(base, class_name):
    base = Path(base)
    found = []
    for split in ("train", "val"):
        directory = base / split / class_name
        if directory.is_dir():
            found.extend(sorted(
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
    return found


@torch.no_grad()
def extract_features(model, paths, transform, device, batch_size=32):
    vectors, batch, done = [], [], 0
    for path in paths:
        with Image.open(path) as im:
            batch.append(transform(im.convert("RGB")))
        if len(batch) == batch_size:
            out = model(torch.stack(batch).to(device))
            vectors.append(out.cpu().numpy())
            done += len(batch)
            batch = []
            print(f"\r  {done}/{len(paths)} images", end="", flush=True)
    if batch:
        out = model(torch.stack(batch).to(device))
        vectors.append(out.cpu().numpy())
        done += len(batch)
    print(f"\r  {done}/{len(paths)} images")
    return np.vstack(vectors)


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------

def thumbnail_data_uri(path, size=132, quality=72):
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_contact_sheet(out_path, clusters, paths, per_cluster, scores):
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
  <h2>Cluster {label}</h2>
  <p class="count">{len(members)} castings &middot; showing the {len(shown)} closest to the cluster centre (most typical)</p>
  <label class="namer">Name this group:
    <input type="text" id="name-{label}" placeholder="e.g. rim damage, porosity, misshapen&hellip;">
  </label>
  <div class="grid">{thumbs}</div>
</section>""")

    score_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in scores)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Defect type clusters</title>
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
  h2 {{ font-size:19px; margin:0 0 3px; color:var(--accent) }}
  .count {{ color:var(--muted); font-size:13px; margin:0 0 14px }}
  .namer {{ display:block; font-size:13px; color:var(--muted);
    margin-bottom:16px }}
  .namer input {{ display:block; margin-top:5px; width:min(340px,100%);
    padding:7px 10px; font-size:14px; border:1px solid var(--rule);
    border-radius:3px; background:var(--bg); color:var(--ink) }}
  .grid {{ display:grid; gap:10px;
    grid-template-columns:repeat(auto-fill,minmax(118px,1fr)) }}
  figure {{ margin:0 }}
  figure img {{ width:100%; display:block; border-radius:3px;
    border:1px solid var(--rule); background:#000 }}
  figcaption {{ font:10px/1.4 ui-monospace,monospace; color:var(--muted);
    margin-top:4px; word-break:break-all }}
  table {{ border-collapse:collapse; font:13px ui-monospace,monospace }}
  td, th {{ padding:5px 14px 5px 0; text-align:left;
    border-bottom:1px solid var(--rule) }}
  .how {{ background:var(--card); border-left:3px solid var(--accent);
    padding:16px 18px; border-radius:0 4px 4px 0; margin-bottom:26px }}
  .how p {{ margin:0 0 10px; max-width:74ch }} .how p:last-child {{ margin:0 }}
</style></head><body><div class="wrap">

<h1>Defect type clusters</h1>
<p class="lede">Castings grouped by what the trained model sees, not by any label. These groupings came from the 4096-dimensional feature vector behind the model's defective/good verdict.</p>

<div class="how">
  <p><strong>What to look for.</strong> Scroll each cluster and ask one question: do these castings share a visible characteristic? A cluster that is clearly all rim damage, or clearly all surface porosity, means the model learned that defect mode on its own.</p>
  <p><strong>If the clusters are coherent</strong>, name them in the boxes, then use <code>cluster_assignments.csv</code> as the starting point for stage-two labels. You are correcting a rough sort rather than labelling from scratch.</p>
  <p><strong>If they look like random mixtures</strong>, the model found one generic "wrongness" signal rather than distinct types. That is worth knowing before anyone labels several thousand images.</p>
  <p>The images shown are the ones closest to each cluster centre &mdash; the most typical members. Outliers sit in the CSV but not on this page.</p>
</div>

<section>
  <h2>How many clusters?</h2>
  <p class="count">Silhouette score by k &mdash; higher means better-separated groups. Below about 0.1 there is little real structure.</p>
  <table><thead><tr><th>k</th><th>silhouette</th></tr></thead>
  <tbody>{score_rows}</tbody></table>
</section>

{"".join(blocks)}

</div></body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data_clean")
    ap.add_argument("--class-name", default="def_front",
                    help="folder name of the defect class")
    ap.add_argument("--weights", default=None,
                    help="path to model_parameters.pth (default: newest)")
    ap.add_argument("--k", type=int, default=None,
                    help="number of clusters (default: best silhouette)")
    ap.add_argument("--per-cluster", type=int, default=24,
                    help="images shown per cluster on the contact sheet")
    ap.add_argument("--out", default="defect_types")
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Path(args.weights) if args.weights else find_latest_weights()
    print(f"Model   : {weights}")
    print(f"Device  : {device}")

    paths = defect_images(args.data, args.class_name)
    if not paths:
        raise SystemExit(
            f"No images under {args.data}/*/{args.class_name}/. "
            "Run make_clean_split.py first.")
    print(f"Defects : {len(paths)} images")
    print()

    model = load_feature_extractor(weights, device=device)
    transform = build_transform()

    print("Extracting features (4096-d, the layer before the verdict):")
    features = extract_features(model, paths, transform, device)
    print(f"  feature matrix {features.shape}")
    print()

    # PCA first: k-means on 4096 raw dimensions is slow and noisier
    n_components = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(features)

    print("Choosing k (silhouette score, higher is better):")
    scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(reduced)
        s = silhouette_score(reduced, km.labels_)
        scores.append((k, s))
        print(f"  k={k}  {s:.4f}")

    best_k = args.k or max(scores, key=lambda kv: kv[1])[0]
    print(f"\nUsing k={best_k}"
          f"{' (best silhouette)' if not args.k else ' (you specified it)'}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(reduced)
    labels = km.labels_

    # order each cluster's members by distance to its centre, so the contact
    # sheet leads with the most typical examples
    clusters = {}
    for label in range(best_k):
        idx = np.flatnonzero(labels == label)
        d = np.linalg.norm(reduced[idx] - km.cluster_centers_[label], axis=1)
        clusters[label] = list(idx[np.argsort(d)])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "cluster_assignments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "cluster", "distance_to_centre"])
        for label in sorted(clusters):
            for rank, i in enumerate(clusters[label]):
                d = np.linalg.norm(reduced[i] - km.cluster_centers_[label])
                w.writerow([str(paths[i]), label, f"{d:.4f}"])

    np.save(out / "features.npy", features)

    print("\nBuilding contact sheet ...")
    write_contact_sheet(out / "contact_sheet.html", clusters, paths,
                        args.per_cluster, scores)

    print()
    print("=" * 60)
    for label in sorted(clusters):
        print(f"  cluster {label}: {len(clusters[label]):>5} castings")
    print("=" * 60)
    print(f"\n  {out/'contact_sheet.html'}      <- open this")
    print(f"  {out/'cluster_assignments.csv'}")
    print(f"  {out/'features.npy'}")
    print("\nOpen the contact sheet and judge whether the groups are")
    print("visually coherent. That decides whether stage two is worth it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
