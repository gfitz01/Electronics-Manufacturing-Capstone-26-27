"""Score stage two against castings it never trained on.

    python label_check.py --build --holdout data_grouped/val/def_front --n 50 --tag v1
    ...label them, save the CSV as defect_types/labels_v1.csv...
    python validate_defect_types.py --tag v1

The number this produces
------------------------
Stage one has a held-out score: 715 castings it never saw, 99.2% right. Stage
two has never had one. Its AUCs come from cross-validation inside the 200
labels, which is a fair measurement but a measurement on the same pool the
model was taught from.

This closes that gap. You label a fresh sample drawn only from the validation
split -- castings neither stage trained on -- and this compares your answers
to what the model already predicted for them, in predicted.csv, written before
you labelled anything. Nothing here can leak: the predictions were fixed first.

What it reports, per defect type
--------------------------------
    agreement    how often the model's yes/no matched yours
    blind        what you would score by always giving the same answer
    AUC          how well its score ranks your yeses above your noes
    precision    of the castings it called present, how many really were
    recall       of the ones that really were, how many it found
    rate         your rate against its rate -- the calibration check

Agreement alone flatters a rare defect: calling chip "no" every time scores
about 87% when chip is in one casting in eight. That is what the blind column
is for -- agreement has to beat it to mean anything. Read precision and recall
alongside, and AUC when you want a number the cutoff cannot flatter.

A small round is still worth running. At 50 castings the intervals are wide:
about plus or minus 6 points on a flag scoring 95%, 9 at 90%, 13 at 50%. That
is enough to tell "this works on new castings" from "this does not", and not
enough to argue about a two-point difference between flags. Going to 100 buys
roughly 3 points of precision -- worth doing if the first round looks good and
the number is going in the report.
"""

import argparse
import csv
import json
import sys
from math import sqrt
from pathlib import Path

FLAGS = ["fray", "chip", "spot", "slash"]


def norm(p):
    return str(p).replace("\\", "/")


def tail(p):
    return "/".join(Path(norm(p)).parts[-3:])


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc(scores, truth):
    """Probability a random positive outranks a random negative."""
    pos = [s for s, y in zip(scores, truth) if y]
    neg = [s for s, y in zip(scores, truth) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def load_round(out_dir, tag, zeros_are_clean=False):
    """Your answers for one labelling round, keyed by source path.

    The labelling page writes '?' in every column for a casting you pressed 0
    on (unreadable) AND for one you never reached. It writes real zeros for a
    casting you judged clean, because toggling nothing is an answer. So an
    all-'?' row is a missing value, not a clean casting, and counting it as
    clean would hand the model four free correct 'no' calls per row -- which on
    a rare flag is most of its agreement. Dropped by default, the same way
    label_check.py --score drops them; --zeros-are-clean reads them as clean if
    that is what you meant by pressing 0."""
    key = Path(out_dir) / f"blind_key_{tag}.csv"
    lab = Path(out_dir) / f"labels_{tag}.csv"
    for p in (key, lab):
        if not p.is_file():
            raise SystemExit(
                f"Missing {p}\n"
                f"A round needs both blind_key_{tag}.csv (written by "
                f"label_check.py --build) and labels_{tag}.csv (the download "
                f"from the labelling page, renamed).")
    files = {}
    with open(key, newline="") as f:
        for row in csv.DictReader(f):
            files[row["id"]] = tail(row["file"])
    out, blank, partial, converted = {}, 0, 0, 0
    with open(lab, newline="") as f:
        reader = csv.DictReader(f)
        if not set(FLAGS) <= set(reader.fieldnames or []):
            raise SystemExit(
                f"{lab} has columns {sorted(reader.fieldnames or [])}, but a "
                f"round expects id,{','.join(FLAGS)}. It is probably from an "
                f"earlier round -- check --tag.")
        for row in reader:
            if row["id"] not in files:
                continue
            vals = {fl: (row.get(fl) or "?").strip() for fl in FLAGS}
            if all(v in ("0", "1") for v in vals.values()):
                out[files[row["id"]]] = {fl: int(vals[fl]) for fl in FLAGS}
            elif all(v == "?" for v in vals.values()):
                if zeros_are_clean:
                    out[files[row["id"]]] = {fl: 0 for fl in FLAGS}
                    converted += 1
                else:
                    blank += 1
            else:
                partial += 1
    return out, blank, partial, converted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v1", help="the round to score")
    ap.add_argument("--out", default="defect_types")
    ap.add_argument("--predicted", default="defect_types/predicted.csv")
    ap.add_argument("--thresholds", default=None,
                    help="cutoffs file; defaults to the one matching "
                         "--no-pos-weight")
    ap.add_argument("--no-pos-weight", action="store_true",
                    help="score against the cutoffs from the "
                         "--no-pos-weight training run. Pass this whenever "
                         "predicted.csv came from apply_defect_types.py "
                         "--no-pos-weight -- the cutoffs have to come from "
                         "the same model that made the predictions.")
    ap.add_argument("--zeros-are-clean", action="store_true",
                    help="read a row of all '?' as 'no visible defect' rather "
                         "than dropping it. Only if pressing 0 was how you "
                         "meant 'nothing wrong with this one' -- otherwise it "
                         "scores unreadable castings as four free correct "
                         "'no' calls each.")
    ap.add_argument("--cut", choices=["rate", "f1", "half"], default="f1",
                    help="which cutoff decides the model's yes/no. 'f1' is "
                         "the per-casting cutoff and the right one here; "
                         "'rate' matches population rates; 'half' is a flat 0.5")
    args = ap.parse_args()
    if args.thresholds is None:
        args.thresholds = ("defect_types/stage_two/thresholds_nopw.json"
                           if args.no_pos_weight
                           else "defect_types/stage_two/thresholds.json")

    truth, blank, partial, converted = load_round(args.out, args.tag,
                                                  args.zeros_are_clean)
    if not truth:
        raise SystemExit(f"No usable rows in labels_{args.tag}.csv.")

    pred_path = Path(args.predicted)
    if not pred_path.is_file():
        raise SystemExit(f"Missing {pred_path}\nRun:  python apply_defect_types.py")
    P, hand = {}, set()
    with open(pred_path, newline="") as f:
        for row in csv.DictReader(f):
            k = tail(row["file"])
            P[k] = {fl: float(row[f"p_{fl}"]) for fl in FLAGS}
            if row.get("hand_labelled") == "1":
                hand.add(k)

    cuts, cut_src = {fl: 0.5 for fl in FLAGS}, "a flat 0.5 for every flag"
    if args.cut != "half":
        key = "threshold_f1" if args.cut == "f1" else "threshold_rate"
        try:
            flags = json.loads(Path(args.thresholds).read_text())["flags"]
            cuts = {fl: float(flags[fl].get(key, flags[fl]["threshold"]))
                    for fl in FLAGS}
            # A thresholds.json written before the two-cutoff change carries a
            # single "threshold", the rate-matching one. Say so rather than
            # calling it the f1 cut, because the numbers below depend on it.
            old = all(key not in flags[fl] for fl in FLAGS)
            cut_src = (f"the single 'threshold' from {args.thresholds} -- that "
                       f"file predates the two-cutoff change, so it is the "
                       f"rate-matching cut, not '{args.cut}'" if old else
                       f"'{args.cut}' cutoffs from {args.thresholds}")
        except (OSError, ValueError, KeyError) as exc:
            # Falling back silently would report a different model's numbers
            # under the flag you asked for.
            print(f"Could not read cutoffs from {args.thresholds} ({exc}); "
                  f"using 0.5 for every flag.\nIf predicted.csv came from a "
                  f"--no-pos-weight run, pass --no-pos-weight here too.\n")

    rows, missing, trained = [], 0, 0
    for k, y in truth.items():
        if k not in P:
            missing += 1
        elif k in hand:
            trained += 1          # labelled in an earlier round; not held out
        else:
            rows.append((P[k], y))
    n = len(rows)
    if not n:
        raise SystemExit("None of these castings have predictions that were "
                         "made without training on them.")

    print(f"Round '{args.tag}': {len(truth)} castings you labelled")
    if converted:
        print(f"  ={converted} marked with 0 read as 'no visible defect' "
              f"(--zeros-are-clean)")
    if blank:
        print(f"  -{blank} skipped as unreadable or never reached")
    if partial:
        print(f"  -{partial} with a box left half-answered")
    if missing:
        print(f"  -{missing} with no prediction in {pred_path.name}")
    if trained:
        print(f"  -{trained} stage two had trained on (earlier round)")
    print(f"  {n} genuinely held out, scored below")
    print(f"  Yes/no decided by {cut_src}")
    print()

    print("=" * 74)
    print("  STAGE TWO AGAINST CASTINGS IT NEVER TRAINED ON")
    print("=" * 74)
    print(f"    {'flag':<7}{'cut':>5}{'right':>9}{'agree':>8}{'95% CI':>13}"
          f"{'blind':>8}{'AUC':>7}{'prec':>7}{'recall':>8}")
    overall, exact, confusion = [], [0] * n, []
    for fl in FLAGS:
        y = [t[fl] for _, t in rows]
        s = [p[fl] for p, _ in rows]
        hit = [sc >= cuts[fl] for sc in s]
        agree = sum(h == bool(t) for h, t in zip(hit, y))
        tp = sum(h and t for h, t in zip(hit, y))
        fp = sum(h and not t for h, t in zip(hit, y))
        fn = sum((not h) and t for h, t in zip(hit, y))
        lo, hi = wilson(agree, n)
        a = auc(s, y)
        # The no-skill score: answer the same way every time, whichever of yes
        # and no is more common in your labels. For a rare flag that is "no
        # every time" and it is high -- chip sits near one casting in eight, so
        # a model that never flags chip still scores about 87% agreement.
        base = max(sum(y), n - sum(y)) / n
        overall.append(agree / n)
        confusion.append((fl, tp, fp, fn, n - tp - fp - fn))
        for j, (h, t) in enumerate(zip(hit, y)):
            exact[j] += (h == bool(t))
        print(f"    {fl:<7}{cuts[fl]:>5.2f}{f'{agree}/{n}':>9}{100*agree/n:>7.1f}%"
              f"{f'{100*lo:.0f} to {100*hi:.0f}':>13}{100*base:>7.1f}%"
              f"{(f'{a:.2f}' if a is not None else '  --'):>7}"
              f"{(f'{tp/(tp+fp):.2f}' if tp + fp else '  --'):>7}"
              f"{(f'{tp/(tp+fn):.2f}' if tp + fn else '  --'):>8}")
    print()
    print("    'blind' is what you would score by giving the same answer every")
    print("    time, whichever of yes and no is more common in your labels.")
    print("    Agreement near that column means the flag learned nothing, no")
    print("    matter how high it looks -- read AUC, precision and recall with")
    print("    it. AUC ignores the cutoff entirely: 0.5 is a coin flip.")

    print()
    print("=" * 74)
    print("  WHERE THE CALLS LANDED, CASTING BY CASTING")
    print("=" * 74)
    print(f"    {'flag':<7}{'caught':>9}{'false alarm':>13}{'missed':>9}"
          f"{'correct no':>13}")
    for fl, tp, fp, fn, tn in confusion:
        print(f"    {fl:<7}{tp:>9}{fp:>13}{fn:>9}{tn:>13}")
    print()
    print("    caught      you said yes, it said yes")
    print("    false alarm it said yes, you said no")
    print("    missed      you said yes, it said no")
    print(f"    Each row adds to {n}.")

    print()
    print("=" * 74)
    print("  RATES: YOURS AGAINST THE MODEL'S")
    print("=" * 74)
    print(f"    {'flag':<7}{'you':>12}{'model':>14}{'gap':>8}")
    for fl in FLAGS:
        ny = sum(t[fl] for _, t in rows)
        nm = sum(p[fl] >= cuts[fl] for p, _ in rows)
        print(f"    {fl:<7}{f'{ny}/{n}':>7}{100*ny/n:>5.0f}%"
              f"{f'{nm}/{n}':>9}{100*nm/n:>5.0f}%{100*(nm-ny)/n:>+8.1f}")
    print()
    print("    A small gap means the model's rates can be quoted as they are.")
    print("    A large one means quote it as a ranking, not a count.")

    full = sum(e == len(FLAGS) for e in exact)
    lo, hi = wilson(full, n)

    # Written so defect_breakdown.py can print the held-out accuracy beside
    # each flag's count, instead of the two numbers living in separate runs.
    score_path = Path(args.out) / "holdout_score.json"
    score_path.write_text(json.dumps({
        "tag": args.tag,
        "castings": n,
        "cut": args.cut,
        "pos_weight": not args.no_pos_weight,
        "thresholds": str(args.thresholds),
        "flags": {fl: {"cut": cuts[fl],
                       "right": int(round(a * n)),
                       "agreement": round(100 * a, 1),
                       "caught": tp, "false_alarm": fp, "missed": fn,
                       "correct_no": tn,
                       "precision": (round(tp / (tp + fp), 3) if tp + fp
                                     else None),
                       "recall": (round(tp / (tp + fn), 3) if tp + fn
                                  else None)}
                  for a, (fl, tp, fp, fn, tn) in zip(overall, confusion)},
        "all_four_right": full,
    }, indent=1), encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  Across all four flags the model agreed with you on "
          f"{100*sum(overall)/len(overall):.1f}% of calls,")
    print(f"  and got all four right on {full} of {n} castings "
          f"({100*full/n:.1f}%, 95% CI {100*lo:.0f} to {100*hi:.0f}).")
    print(f"  This is stage two's held-out score, the counterpart of stage")
    print(f"  one's accuracy -- measured on {n} castings neither stage trained")
    print("  on, against predictions written before you labelled them.")
    print()
    print("  Quote the per-flag numbers, not the 'across all four' figure: one")
    print("  casting contributes four correlated calls, so an interval built")
    print(f"  on {4*n} independent calls would be about twice too narrow.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
