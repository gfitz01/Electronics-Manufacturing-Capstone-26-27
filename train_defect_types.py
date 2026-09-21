"""Stage two: train a model to say WHICH defects a casting has.

    python train_defect_types.py                      pilot + learning curve
    python train_defect_types.py --full               train one model on everything
    python train_defect_types.py --arch resnet18      stronger backbone
    python train_defect_types.py --epochs 60

What this is
------------
Stage one answers good or bad. This is stage two: given a casting stage one
flagged as defective, which of the four defects does it carry --

    fray    a rim ragged or torn along its length
    chip    material missing from the edge, one discrete bite
    spot    a blemish on the flat face, away from the rim
    slash   a linear scratch, gouge or drag line

Four independent yes/no outputs, NOT a four-way choice. 37% of defective
castings carry two or more at once, so a model forced to pick one label is
wrong by construction on more than a third of them. That means a sigmoid on
each output and BCE loss, not a softmax and cross-entropy.

Why it trains on YOUR labels and not the clusters
-------------------------------------------------
The unsupervised grouping is tempting -- it labels all 4,211 defect images for
free. Don't use it. It is about a 65%-accurate stand-in for real defect type,
only 12 points better than guessing. A model trained on it would score ~99%
against the clusters and ~65% against reality, and only the first number would
ever appear in its own output. Hand labels are the only non-circular source.

Why the folds are group-aware
-----------------------------
The dataset contains large-scale rotational duplication: the same physical
casting appears several times at different angles. Split those across a fold
boundary and the model is tested on rotations of what it trained on, which is
exactly the flaw that made the original 715-image result meaningless. This
script reads rotation_duplicates.csv and keeps every duplicate family whole
inside one fold. Without that file it warns and proceeds, and the numbers
should then be treated as optimistic.

The learning curve is the point of the pilot
--------------------------------------------
Training on 60, then 100, then 140, then everything, shows whether performance
is still climbing when the labels run out. That answers the only question worth
answering before committing hours to labelling:

  still climbing steeply  -> more labels will pay off, and roughly how many
  flattening              -> more labels will not help much; change the
                             approach instead (stronger backbone, more
                             augmentation, or a different flag definition)

Each flag gets its own verdict, because they differ enormously. spot is nearly
balanced at 48.5% and has the most to work with; chip at 14.5% has about 29
positive examples in 200 and will almost certainly need far more, or to be
folded into a broader rim-damage flag.

Read AUC, not accuracy
----------------------
With chip at 14.5%, a model that answers "no chip" every time scores 85.5%
accuracy and is useless. The script prints that majority-class baseline beside
every accuracy so the comparison is unavoidable, and leads with AUC, which is
unaffected by how rare a flag is. AUC 0.5 is chance, 1.0 is perfect.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FLAGS = ["fray", "chip", "spot", "slash"]


def norm(p):
    """The label CSVs store Windows paths. Backslashes work on Windows and
    forward slashes work everywhere, so normalise once and the scripts run
    on either."""
    return Path(str(p).replace("\\", "/"))


def tail(p):
    """split/class/filename. The source dataset reuses filenames across its
    train and test folders for DIFFERENT images (44 among def_front alone), so
    a bare filename is not a unique key. Always join on this instead."""
    return "/".join(norm(p).parts[-3:])


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_labels(key_path, labels_path):
    """Join the blind key (id -> file) to the answers (id -> flags)."""
    key = {}
    with open(key_path, newline="") as f:
        for row in csv.DictReader(f):
            key[int(row["id"])] = row["file"]

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not set(FLAGS) <= cols:
            raise SystemExit(
                f"{labels_path} has columns {sorted(cols)}; this needs "
                f"id,{','.join(FLAGS)}.\nThat is probably an earlier round's "
                "file -- check which my_labels*.csv you passed.")
        out = []
        for row in reader:
            i = int(row["id"])
            if i not in key:
                continue
            vals = [row[f].strip() for f in FLAGS]
            if all(v in ("0", "1") for v in vals):
                y = [int(v) for v in vals]
            elif all(v == "?" for v in vals):
                y = [0, 0, 0, 0]      # "nothing wrong with it" -- a real answer
            else:
                continue
            out.append((str(norm(key[i])), y))
    return out


def load_rounds(rounds, out_dir="defect_types"):
    """Merge several labelling rounds into one training set.

    Each round is a (blind_key_<tag>.csv, labels_<tag>.csv) pair. Ids restart
    at 1 every round, so they are joined per round and keyed by file path --
    joining on id across rounds would silently mislabel everything.

    If a casting somehow appears twice, the later round wins: the shortlist
    excludes already-labelled castings, so an overlap means something was
    relabelled deliberately.
    """
    seen, per_round = {}, []
    for tag in rounds:
        k = Path(out_dir) / f"blind_key_{tag}.csv"
        l = Path(out_dir) / f"labels_{tag}.csv"
        if not k.is_file() or not l.is_file():
            have = sorted(p.name for p in Path(out_dir).glob("labels_*.csv"))
            raise SystemExit(
                f"Round '{tag}' needs both {k.name} and {l.name}.\n"
                f"Label files present: {have or 'none'}\n"
                f"After labelling, rename the download to {l.name}.")
        rows = load_labels(k, l)
        before = len(seen)
        for f, y in rows:
            seen[f] = y
        per_round.append((tag, len(rows), len(seen) - before))
    for tag, n, added in per_round:
        dup = n - added
        print(f"    round {tag}: {n} labelled"
              + (f"  ({dup} already seen, later round kept)" if dup else ""))
    return list(seen.items())


def duplicate_groups(dup_path, files):
    """file -> family id, so rotations of one casting stay in one fold."""
    parent = {f: f for f in files}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    if not Path(dup_path).is_file():
        return None, 0
    byname = {}
    for f in files:
        byname.setdefault(tail(f), []).append(f)
    merged = 0
    with open(dup_path, newline="") as f:
        for row in csv.DictReader(f):
            a = tail(row["file_a"])
            b = tail(row["file_b"])
            if a in byname and b in byname:
                join(byname[a][0], byname[b][0]); merged += 1
    return {f: find(f) for f in files}, merged


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(arch, n_out, freeze, device, init_weights=None):
    """init_weights: path to the trained BINARY model. Loading it means stage
    two starts from a network that already learned what a casting looks like
    from all 6,569 training images, instead of from ImageNet cats and cars.
    Those images carry no defect-type labels so they cannot be trained on
    directly -- but everything they taught the backbone carries over."""
    import torch
    import torch.nn as nn
    from torchvision import models
    if arch == "alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        if init_weights:
            m.classifier[-1] = nn.Linear(4096, 2)      # the binary head
            state = torch.load(init_weights, map_location="cpu")
            missing = m.load_state_dict(state, strict=False)
            if getattr(missing, "unexpected_keys", None):
                raise SystemExit(
                    f"{init_weights} does not look like this model's weights "
                    f"(unexpected: {list(missing.unexpected_keys)[:3]})")
        if freeze:
            for p in m.features.parameters():
                p.requires_grad = False
        m.classifier[-1] = nn.Linear(4096, n_out)
    elif arch == "resnet18":
        if init_weights:
            raise SystemExit(
                "--init-binary only works with --arch alexnet: the binary "
                "model IS an AlexNet, so its weights do not fit a ResNet.")
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze:
            for name, p in m.named_parameters():
                if not name.startswith(("layer4", "fc")):
                    p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, n_out)
    else:
        raise SystemExit(f"unknown --arch {arch}")
    return m.to(device)


def transforms_for(size=224):
    from torchvision import transforms
    train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),      # the parts have no canonical up
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
    ])
    val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return train, val


class DS:
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        import torch
        from PIL import Image
        path, y = self.items[i]
        with Image.open(path) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor(y, dtype=torch.float32)


def run_fold(train_items, val_items, args, device, pos_weight):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    ttf, vtf = transforms_for()
    tl = DataLoader(DS(train_items, ttf), batch_size=args.batch_size,
                    shuffle=True, num_workers=0)
    vl = DataLoader(DS(val_items, vtf), batch_size=args.batch_size,
                    shuffle=False, num_workers=0)

    model = build_model(args.arch, len(FLAGS), args.freeze, device,
                        args.init_binary)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for _ in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y in vl:
            P.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            Y.append(y.numpy())
    return np.vstack(P), np.vstack(Y)


# --------------------------------------------------------------------------

def evaluate(P, Y):
    from sklearn.metrics import roc_auc_score
    out = {}
    for k, f in enumerate(FLAGS):
        y, p = Y[:, k], P[:, k]
        pos = int(y.sum()); n = len(y)
        base = max(pos, n - pos) / n if n else 0.0
        acc = float(((p >= 0.5).astype(int) == y).mean()) if n else 0.0
        try:
            auc = float(roc_auc_score(y, p)) if 0 < pos < n else float("nan")
        except ValueError:
            auc = float("nan")
        tp = int(((p >= 0.5) & (y == 1)).sum())
        fp = int(((p >= 0.5) & (y == 0)).sum())
        fn = int(((p < 0.5) & (y == 1)).sum())
        out[f] = {"n": n, "pos": pos, "auc": auc, "acc": acc, "baseline": base,
                  "precision": tp / (tp + fp) if tp + fp else float("nan"),
                  "recall": tp / (tp + fn) if tp + fn else float("nan")}
    return out


def cv(items, groups, args, device, sizes):
    import torch
    rng = random.Random(args.seed)

    if args.permute:
        # Permutation test. Shuffle one flag's labels so the picture and the
        # answer no longer correspond, then train exactly as before. A model
        # that still scores well is not reading the casting -- it is exploiting
        # something structural, and the honest AUC for that flag is whatever
        # this prints, not what the real run printed.
        k = FLAGS.index(args.permute)
        col = [y[k] for _, y in items]
        random.Random(args.seed + 991).shuffle(col)
        items = [(f, [c if j != k else col[i] for j, c in enumerate(y)])
                 for i, (f, y) in enumerate(items)]
        print(f"  PERMUTED '{args.permute}': its labels are now shuffled.")
        print("  Expect AUC near 0.50. Anything much higher means leakage.\n")

    keys = sorted({groups[f] for f, _ in items}) if groups else \
        [f for f, _ in items]
    rng.shuffle(keys)
    folds = [keys[i::args.folds] for i in range(args.folds)]
    gkey = (lambda f: groups[f]) if groups else (lambda f: f)

    results = {}
    for size in sizes:
        P_all, Y_all = [], []
        for fi in range(args.folds):
            held = set(folds[fi])
            tr = [it for it in items if gkey(it[0]) not in held]
            va = [it for it in items if gkey(it[0]) in held]
            if not va or not tr:
                continue
            rng.shuffle(tr)
            tr = tr[:size] if size else tr
            if len(tr) < 8:
                continue
            Y = np.array([y for _, y in tr], dtype=np.float32)
            pw = torch.tensor([(len(Y) - Y[:, k].sum()) / max(Y[:, k].sum(), 1)
                               for k in range(len(FLAGS))], dtype=torch.float32)
            pw = pw.clamp(1.0, 8.0)
            P, Yv = run_fold(tr, va, args, device, pw)
            P_all.append(P); Y_all.append(Yv)
            print(f"\r  n={size or len(tr):<4} fold {fi+1}/{args.folds}",
                  end="", flush=True)
        if P_all:
            results[size or "all"] = evaluate(np.vstack(P_all), np.vstack(Y_all))
        print()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", default=None,
                    help="comma-separated labelling rounds to train on, e.g. "
                         "r2,r3 -- reads defect_types/blind_key_<tag>.csv and "
                         "labels_<tag>.csv for each and merges them.")
    ap.add_argument("--key", default="defect_types/blind_key_r2.csv")
    ap.add_argument("--labels", default="defect_types/labels_r2.csv")
    ap.add_argument("--duplicates", default="rotation_duplicates.csv")
    ap.add_argument("--arch", default="alexnet", choices=["alexnet", "resnet18"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--init-binary", metavar="PTH", default=None,
                    help="start from the trained good/bad model's weights "
                         "instead of ImageNet. This is how the 6,569 training "
                         "images contribute: they cannot supply defect-type "
                         "labels, but they taught the backbone what castings "
                         "look like.")
    ap.add_argument("--freeze", action="store_true",
                    help="train only the head. With ~200 images this often "
                         "beats fine-tuning everything.")
    ap.add_argument("--permute", metavar="FLAG", default=None,
                    choices=FLAGS,
                    help="shuffle this flag's labels before training. A real "
                         "result collapses to AUC 0.50; a leak does not. Run "
                         "this on any flag that scores suspiciously well.")
    ap.add_argument("--full", action="store_true",
                    help="skip the curve, train one model on all labels")
    ap.add_argument("--out", default="defect_types/stage_two")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(args.labels).is_file():
        cands = sorted(Path("defect_types").glob("my_labels*.csv"))
        raise SystemExit(f"Missing {args.labels}\n"
                         f"Candidates: {[str(c) for c in cands] or 'none'}\n"
                         "Pass one with --labels")
    if args.rounds:
        tags = [t.strip() for t in args.rounds.split(",") if t.strip()]
        print(f"Rounds        : {', '.join(tags)}")
        items = load_rounds(tags, str(Path(args.key).parent))
    else:
        items = load_labels(args.key, args.labels)
    if not items:
        raise SystemExit("No usable labelled castings.")

    files = [f for f, _ in items]
    groups, merged = duplicate_groups(args.duplicates, files)
    Y = np.array([y for _, y in items])

    print(f"Device        : {device}")
    src = "the trained good/bad model" if args.init_binary else "ImageNet"
    print(f"Backbone      : {args.arch}{'  (head only)' if args.freeze else ''}"
          f"  starting from {src}")
    if args.init_binary and not Path(args.init_binary).is_file():
        raise SystemExit(f"Missing {args.init_binary}\n"
                         "Find it with:  dir /s /b results\\*.pth")
    print(f"Labelled      : {len(items)} castings")
    for k, f in enumerate(FLAGS):
        pos = int(Y[:, k].sum())
        print(f"    {f:<7} {pos:>4} positive  ({100*pos/len(items):4.1f}%)")
    print(f"    none     {int((Y.sum(axis=1) == 0).sum()):>4} with no defect visible")
    print(f"    2+       {int((Y.sum(axis=1) >= 2).sum()):>4} with several at once")

    if groups is None:
        print("\n  WARNING: no rotation_duplicates.csv. Folds cannot be made")
        print("  group-aware, so a casting's rotated twin may sit on the other")
        print("  side of a fold boundary. Treat every number below as")
        print("  optimistic. Run check_rotations.py first.")
    else:
        nfam = len({groups[f] for f in files})
        print(f"\nDuplicate-aware folds: {len(files)} castings -> {nfam} families"
              f"  ({merged} linking pairs)")
        if nfam < len(files):
            print(f"  {len(files)-nfam} castings share a family with another in")
            print("  this sample; they will never be split across a fold.")

    sizes = [None] if args.full else [60, 100, 140, None]
    if args.permute:
        print(f"\n*** PERMUTATION TEST on '{args.permute}' ***")
    print(f"\nTraining {args.folds}-fold, {args.epochs} epochs per fold"
          f"{'' if args.full else ', at four training sizes'}")
    results = cv(items, groups, args, device, sizes)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # A permutation run is a diagnostic, not a result. Writing it to
    # results.json would let downstream tools read a deliberately broken score
    # as the real one -- which is exactly what happened: apply_defect_types.py
    # picked its shortlist flag by "weakest AUC" and found fray at 0.456.
    name = f"results_permuted_{args.permute}.json" if args.permute else "results.json"
    (out / name).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out/name}")

    print("\n" + "=" * 70)
    print("  AUC by flag and training-set size   (0.50 = chance, 1.00 = perfect)")
    print("=" * 70)
    keys = [k for k in results]
    print(f"    {'flag':<8}" + "".join(f"{str(k):>10}" for k in keys) + "    trend")
    for f in FLAGS:
        row = [results[k][f]["auc"] for k in keys]
        cells = "".join(f"{v:>10.3f}" if v == v else f"{'--':>10}" for v in row)
        good = [v for v in row if v == v]
        if len(good) >= 2:
            d = good[-1] - good[0]
            trend = ("still climbing" if d > 0.05 else
                     "flattening" if d > -0.02 else "noisy/flat")
        else:
            trend = "--"
        print(f"    {f:<8}{cells}    {trend}")

    last = results[keys[-1]]
    print("\n" + "=" * 70)
    print("  At full label count")
    print("=" * 70)
    print(f"    {'flag':<8}{'AUC':>7}{'acc':>8}{'baseline':>10}"
          f"{'precision':>11}{'recall':>9}{'pos':>6}")
    for f in FLAGS:
        r = last[f]
        print(f"    {f:<8}{r['auc']:>7.3f}{100*r['acc']:>7.1f}%"
              f"{100*r['baseline']:>9.1f}%{r['precision']:>11.2f}"
              f"{r['recall']:>9.2f}{r['pos']:>6}")
    print("\n  'baseline' is what you get by always answering no. An accuracy")
    print("  that only matches it means the model learned nothing, however")
    print("  high the number looks.")

    print("\n" + "=" * 70)
    print("  What to do next")
    print("=" * 70)
    for f in FLAGS:
        row = [results[k][f]["auc"] for k in keys if results[k][f]["auc"] == results[k][f]["auc"]]
        auc = row[-1] if row else float("nan")
        climb = (row[-1] - row[0]) if len(row) >= 2 else 0.0
        pos = last[f]["pos"]
        if auc != auc:
            msg = "too few positives to score at all"
        elif auc < 0.60:
            msg = ("not learnable from this many labels; more of the same may "
                   "not help") if climb <= 0.05 else "weak but still climbing"
        elif auc < 0.75:
            msg = ("promising -- more labels should help"
                   if climb > 0.05 else "modest and flattening; try --arch resnet18")
        else:
            msg = ("working well" + ("; still climbing, so more labels will help"
                                     if climb > 0.05 else "; near its ceiling here"))
        print(f"    {f:<8} AUC {auc:.2f}  {msg}")
    print("\n  Rough labelling cost to reach ~500 positives per flag, at the")
    print("  rates measured in this sample:")
    for k, f in enumerate(FLAGS):
        rate = Y[:, k].mean()
        if rate > 0:
            print(f"    {f:<8} {int(500/rate):>6,} castings to label")
    print("\n  Label the flags that are climbing. A flag that is flat at 200")
    print("  will usually still be flat at 2,000 -- that is a definition")
    print("  problem, not a data-volume problem.")
    print(f"\n  {out/'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
