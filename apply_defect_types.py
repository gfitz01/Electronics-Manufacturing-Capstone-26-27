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
    labelled = {Path(f).name for f, _ in items}
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
            tot += float(loss) * len(x)
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
    was_labelled = [Path(p).name in labelled for p in paths]
    with open(out / "predicted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "hand_labelled"] + [f"p_{x}" for x in FLAGS])
        for (p, split), row, lab in zip(everything, P, was_labelled):
            w.writerow([str(p), split, int(lab)] + [f"{v:.4f}" for v in row])

    # ---------------- what it thinks is out there ----------------
    fresh = ~np.array(was_labelled)
    print()
    print("=" * 68)
    print("  ESTIMATED defect rates across all defective castings")
    print("=" * 68)
    print(f"    {'flag':<8}{'predicted':>11}{'hand-labelled':>16}"
          f"{'agreement check':>18}")
    for k, f in enumerate(FLAGS):
        pred_all = float((P[:, k] >= 0.5).mean())
        hand = float(Y[:, k].mean())
        on_hand = float((P[np.array(was_labelled), k] >= 0.5).mean()) \
            if any(was_labelled) else float("nan")
        print(f"    {f:<8}{100*pred_all:>10.1f}%{100*hand:>15.1f}%"
              f"{100*on_hand:>17.1f}%")
    print()
    print("  Column 2 is the model's guess over all 4,211. Column 3 is the")
    print("  truth on the 200 you labelled. Column 4 is what the model says")
    print("  about those same 200 -- it trained on them, so it should match")
    print("  column 3 closely. If it does not, something is wrong; if it does,")
    print("  that only proves it memorised its training set, not that column 2")
    print("  is right. Trust column 2 to about the accuracy the pilot measured.")

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
