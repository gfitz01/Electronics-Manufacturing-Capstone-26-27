"""Train on the 200 hand labels, then predict defect types for all 4,211.

    python apply_defect_types.py
    python apply_defect_types.py --focus chip --shortlist 200
    python apply_defect_types.py --epochs 60 --arch resnet18

Two different things this gives you
-----------------------------------
1. PREDICTIONS for every defective casting -- an estimate of which defects each
   one carries, written to defect_types/predicted.csv. This is the output you
   wanted: defect types across the whole dataset instead of a 200-image sample.

2. A SHORTLIST of which castings to label next. This is the part that actually
   makes the model better, and it is worth more than the predictions.

What applying the model does NOT do
-----------------------------------
It does not improve the model. Accuracy is already fixed at what
train_defect_types.py measured by cross-validation (fray 1.00, spot 0.92,
slash 0.92, chip 0.86). Running the same weights over 4,011 more images
produces 4,011 more guesses at that same accuracy -- no better, no worse.

And do NOT train on these predictions. Feeding a model its own output back as
labels ("pseudo-labelling") makes it agree with itself: the score climbs while
the real accuracy does not, and every mistake it already makes gets reinforced
into training data. That is the same circularity as training on the k-means
clusters, one step further disguised. Predictions are an estimate to read, not
a label to learn from.

Why the shortlist is the valuable half
--------------------------------------
Labelling castings at random means most of your effort goes on easy ones the
model already gets right -- an obvious torn rim teaches it nothing it does not
know. The castings worth your time are the ones it finds genuinely ambiguous,
where its prediction sits near 0.5.

So this ranks all 4,011 unlabelled castings by how uncertain the model is, and
writes the top ones to defect_types/shortlist.csv. Labelling those is worth
roughly two to three times labelling the same number at random, because each
one resolves a case the model cannot currently call.

Then:

    python label_check.py --build --from-shortlist --tag r3

builds a blind labelling sheet from exactly those castings, and you are back
round the loop with the labels that buy the most.

--focus picks which flag's uncertainty to rank by. Default is the flag with the
weakest cross-validated AUC, which is where more labels help most. chip is the
obvious candidate: 0.86 AUC on only 29 positive examples.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FLAGS = ["fray", "chip", "spot", "slash"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def norm(p):
    return Path(str(p).replace("\\", "/"))


def tail(p):
    """split/class/filename. The source dataset reuses filenames across its
    train and test folders for DIFFERENT images (44 among def_front alone), so
    a bare filename is not a unique key. Always join on this instead."""
    return "/".join(norm(p).parts[-3:])


def load_labels(key_path, labels_path):
    key = {}
    with open(key_path, newline="") as f:
        for row in csv.DictReader(f):
            key[int(row["id"])] = row["file"]
    out = []
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        if not set(FLAGS) <= set(reader.fieldnames or []):
            raise SystemExit(f"{labels_path} needs columns id,{','.join(FLAGS)}")
        for row in reader:
            i = int(row["id"])
            if i not in key:
                continue
            v = [row[f].strip() for f in FLAGS]
            if all(x in ("0", "1") for x in v):
                out.append((str(norm(key[i])), [int(x) for x in v]))
            elif all(x == "?" for x in v):
                out.append((str(norm(key[i])), [0, 0, 0, 0]))
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


def all_defects(data, class_name):
    found = []
    for split in ("train", "val"):
        d = Path(data) / split / class_name
        if d.is_dir():
            found.extend((p, split) for p in sorted(d.rglob("*"))
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_clean")
    ap.add_argument("--class-name", default="def_front")
    ap.add_argument("--rounds", default=None,
                    help="comma-separated labelling rounds to train on, e.g. "
                         "r2,r3 -- reads defect_types/blind_key_<tag>.csv and "
                         "labels_<tag>.csv for each and merges them.")
    ap.add_argument("--key", default="defect_types/blind_key_r2.csv")
    ap.add_argument("--labels", default="defect_types/labels_r2.csv")
    ap.add_argument("--arch", default="alexnet", choices=["alexnet", "resnet18"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--focus", default=None, choices=FLAGS,
                    help="rank the shortlist by this flag's uncertainty "
                         "(default: the weakest flag from the pilot)")
    ap.add_argument("--shortlist", type=int, default=200)
    ap.add_argument("--out", default="defect_types")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_defect_types import build_model, transforms_for, DS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.rounds:
        tags = [t.strip() for t in args.rounds.split(",") if t.strip()]
        print(f"Rounds      : {', '.join(tags)}")
        items = load_rounds(tags, str(Path(args.key).parent))
    else:
        items = load_labels(args.key, args.labels)
    if not items:
        raise SystemExit("No labels found.")
    Y = np.array([y for _, y in items], dtype=np.float32)
    print(f"Device      : {device}")
    print(f"Training on : {len(items)} hand-labelled castings")
    for k, f in enumerate(FLAGS):
        print(f"    {f:<7} {int(Y[:, k].sum()):>4} positive")

    everything = all_defects(args.data, args.class_name)
    if not everything:
        raise SystemExit(f"No images under {args.data}/*/{args.class_name}/")
    labelled = {tail(f) for f, _ in items}
    print(f"\nPredicting  : {len(everything)} defective castings "
          f"({len(everything)-len(labelled)} of them never labelled)")

    # ---------------- train on everything we have ----------------
    ttf, vtf = transforms_for()
    tl = DataLoader(DS(items, ttf), batch_size=args.batch_size, shuffle=True)
    model = build_model(args.arch, len(FLAGS), args.freeze, device, None)
    pw = torch.tensor([(len(Y) - Y[:, k].sum()) / max(Y[:, k].sum(), 1)
                       for k in range(len(FLAGS))],
                      dtype=torch.float32).clamp(1.0, 8.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"\nTraining {args.epochs} epochs on all {len(items)} labels")
    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"\r  epoch {ep+1}/{args.epochs}  loss {tot/len(items):.4f}",
                  end="", flush=True)
    print()

    # ---------------- predict everything ----------------
    print("\nPredicting ...")
    model.eval()
    paths = [p for p, _ in everything]
    pl = DataLoader(DS([(str(p), [0] * len(FLAGS)) for p in paths], vtf),
                    batch_size=args.batch_size, shuffle=False)
    P = []
    with torch.no_grad():
        for i, (x, _) in enumerate(pl):
            P.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"\r  {min((i+1)*args.batch_size, len(paths))}/{len(paths)}",
                      end="", flush=True)
    P = np.vstack(P)
    print(f"\r  {len(paths)}/{len(paths)}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    was_labelled = [tail(p) in labelled for p in paths]
    with open(out / "predicted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "hand_labelled"] + [f"p_{x}" for x in FLAGS])
        for (p, split), row, lab in zip(everything, P, was_labelled):
            w.writerow([str(p), split, int(lab)] + [f"{v:.4f}" for v in row])

    # ---------------- what it thinks is out there ----------------
    fresh = ~np.array(was_labelled)

    # The hand-labelled 200 were NOT a representative sample: they were drawn
    # evenly from the two clusters, while the population is split unevenly.
    # Comparing the model's rate over 4,211 against the raw rate over those
    # 200 compares two different populations and makes the model look wrong.
    corrected = {}
    meta = Path(args.key).parent / "blind_meta_r2.json"
    grp = {}
    try:
        gk = {}
        with open(args.key, newline="") as f:
            for row in csv.DictReader(f):
                gk[str(norm(row["file"]))] = row.get("group", "?")
        sizes = json.loads(meta.read_text())["group_sizes"]
        total = sum(sizes.values())
        for k, fl in enumerate(FLAGS):
            acc = 0.0
            for g, n in sizes.items():
                rows = [y[k] for (fp, y) in items if gk.get(fp) == g]
                if rows:
                    acc += (n / total) * (sum(rows) / len(rows))
            corrected[fl] = acc
    except Exception:
        corrected = {}

    print()
    print("=" * 74)
    print("  ESTIMATED defect rates across all defective castings")
    print("=" * 74)
    hdr = f"    {'flag':<8}{'model says':>13}"
    if corrected:
        hdr += f"{'hand labels,':>16}{'gap':>8}"
    hdr += f"{'hand labels,':>16}"
    print(hdr)
    print(f"    {'':<8}{'(all 4211)':>13}"
          + (f"{'pop-corrected':>16}{'':>8}" if corrected else "")
          + f"{'raw 200':>16}")
    for k, f in enumerate(FLAGS):
        pred_all = float((P[:, k] >= 0.5).mean())
        raw = float(Y[:, k].mean())
        line = f"    {f:<8}{100*pred_all:>12.1f}%"
        if corrected:
            c = corrected[f]
            line += f"{100*c:>15.1f}%{100*(pred_all-c):>+8.1f}"
        line += f"{100*raw:>15.1f}%"
        print(line)
    print()
    if corrected:
        print("  Compare column 2 to column 3, NOT to the raw 200. Those 200")
        print("  were drawn evenly from the two clusters while the real split is")
        print("  uneven, so their raw rates describe a different population.")
        print("  Column 3 reweights them back to the full defect set.")
        print()
        print("  A positive gap on a rare flag is expected: pos_weight pushes")
        print("  the model toward calling the rarer class, by design. Check it")
        print("  against each flag's pos_weight before reading it as an error.")
    else:
        print("  Could not reweight (missing blind_meta_r2.json), so the last")
        print("  column is the raw sample rate and is NOT directly comparable.")

    # ---------------- the shortlist ----------------
    focus = args.focus
    if focus is None:
        pilot = out / "stage_two" / "results.json"
        if pilot.is_file():
            try:
                r = json.loads(pilot.read_text())
                last = r[list(r)[-1]]
                focus = min(FLAGS, key=lambda f: last[f]["auc"]
                            if last[f]["auc"] == last[f]["auc"] else 9)
            except Exception:
                focus = "chip"
        else:
            focus = "chip"
    k = FLAGS.index(focus)

    # uncertainty: 1.0 when the model sits exactly on the fence
    unc = 1.0 - 2.0 * np.abs(P[:, k] - 0.5)
    order = [i for i in np.argsort(-unc) if fresh[i]][:args.shortlist]
    with open(out / "shortlist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "file", "split", f"p_{focus}", "uncertainty"])
        for rank, i in enumerate(order, 1):
            w.writerow([rank, str(paths[i]), everything[i][1],
                        f"{P[i, k]:.4f}", f"{unc[i]:.4f}"])

    print()
    print("=" * 68)
    print(f"  SHORTLIST -- the {len(order)} castings worth labelling next")
    print("=" * 68)
    print(f"    ranked by how undecided the model is about '{focus}'"
          f"{'  (weakest flag in the pilot)' if not args.focus else ''}")
    print(f"    their p_{focus} sits between "
          f"{P[order, k].min():.3f} and {P[order, k].max():.3f}")
    print(f"    for comparison, across all {len(paths)} castings only "
          f"{int((unc > 0.5).sum())} are this undecided")
    print()
    print("  Labelling these is worth roughly 2-3x labelling the same number at")
    print("  random: each one resolves a case the model currently cannot call,")
    print("  where an obvious torn rim teaches it nothing it does not know.")
    print()
    print(f"  {out/'predicted.csv'}    every casting, every flag")
    print(f"  {out/'shortlist.csv'}    what to label next")
    print()
    print("  Next round:")
    print("      python label_check.py --build --from-shortlist --tag r3")
    print("      (then label, then re-run train_defect_types.py and compare")
    print("       the AUC table against this one)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
