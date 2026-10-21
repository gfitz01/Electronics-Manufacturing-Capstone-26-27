"""Write the inspection results as one JSON file a front end can read.

    python export_results.py
    python export_results.py --scope all --out docs/results.json
    python export_results.py --run results/alexnet-aug-20260928-193733

Everything else in this project prints tables for people. This writes one file
for code: one record per casting, no prose, stable field names.

The file
--------
    {
      "schema": "casting-inspection/1",
      "generated_at": "2026-09-30T14:02:11Z",
      "source": { ...which runs these numbers came from... },
      "summary": { ...the totals, counted from the records below... },
      "castings": [ ...one object each... ]
    }

The summary is there so a dashboard does not have to recount 4,000 records on
every page load, and so both of you are reading the same arithmetic. Every
figure in it is derived from "castings", so a front end that prefers to filter
and count for itself will always agree with it.

    summary.castings            how many records
    summary.verdict             good / defective / not_inspected
    summary.label               what the dataset says, same three
    summary.accuracy            correct, incorrect, pct, and the two kinds of
                                mistake: missed_defects and false_alarms
    summary.near_duplicates     records with a near-copy in training
    summary.defect_types
        assessed                castings stage two scored
        counts / rates_pct      per defect type, over those castings
        cutoffs                 the cutoff each type was read at
        per_casting             how many carry 0, 1, 2, 3 or 4 types
        combinations            the commonest sets, biggest first

A casting record:

    id            "cast_def_0_1021"   filename without the extension, unique
                                      across the file. The dataset reuses a
                                      few filenames across its own train and
                                      test folders; those get the source
                                      folder in front ("train-cast_def_0_15").
    file          "cast_def_0_1021.jpeg"
    split         "val" | "train" | "dev" | null
    label         "defective" | "good" | null      the dataset's own label
    verdict       "defective" | "good" | null      what stage one decided
    confidence    0.0-1.0 | null                   how sure stage one was
    correct       true | false | null              verdict vs label
    defects       [ {type, score, present, cutoff} ] | null
    defect_count  0-4 | null                       how many scored present
    hand_labelled true when this is one of the castings you labelled by hand,
                  so stage two trained on it and its scores are memory
    duplicate     { "of": "cast_def_0_339.jpeg", "residual": 1.97 } | null

Three fields are null rather than absent when they do not apply, so the front
end can tell "no" from "not measured":

  * verdict/confidence/correct are null for castings stage one never
    inspected -- it is scored on the validation split only.
  * defects/defect_count are null for castings stage two never saw: it runs
    on defective castings, so a good casting has no defect scores rather than
    four zeros.
  * duplicate is null unless --residuals says this casting has a near-copy in
    the training set. A front end showing accuracy should let people exclude
    those, since the model had effectively seen them.

Scores are the model's raw output, 0 to 1. 'present' applies the cutoff that
train_defect_types.py measured for that flag (see --cut); the cutoff travels
with each score so the front end can re-decide without knowing our conventions.
"""

import argparse
import csv
import re
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FLAGS = ["fray", "chip", "spot", "slash"]
SCHEMA = "casting-inspection/1"
CLASS_LABEL = {"def_front": "defective", "ok_front": "good"}


def norm(p):
    return str(p).replace("\\", "/")


def stem(name):
    return name.rsplit(".", 1)[0]


def newest_run(results_dir):
    """The most recent stage-one run, by the timestamp it recorded itself."""
    runs = []
    for d in Path(results_dir).iterdir() if Path(results_dir).is_dir() else []:
        rj = d / "results.json"
        if rj.is_file():
            try:
                started = json.loads(rj.read_text())["run"]["started_at"]
            except (OSError, ValueError, KeyError):
                started = ""
            runs.append((started, d))
    if not runs:
        raise SystemExit(f"No runs with a results.json under {results_dir}/.\n"
                         "Train one first:  python run.py --data data_grouped")
    return sorted(runs)[-1][1]


def read_stage_one(run_dir):
    """Per-image verdicts, plus what produced them."""
    run_dir = Path(run_dir)
    preds, meta = {}, {}
    pj = run_dir / "predictions.csv"
    if pj.is_file():
        with open(pj, newline="") as f:
            for row in csv.DictReader(f):
                name = norm(row["file"]).rsplit("/", 1)[-1]
                conf = row.get("confidence")
                preds[name] = {
                    "label": CLASS_LABEL.get(row["true"], row["true"]),
                    "verdict": CLASS_LABEL.get(row["predicted"], row["predicted"]),
                    "correct": row["correct"] == "True",
                    "confidence": round(float(conf), 4) if conf else None,
                }
    rj = run_dir / "results.json"
    if rj.is_file():
        d = json.loads(rj.read_text())
        meta = {
            "run": run_dir.name,
            "model": d.get("run", {}).get("model"),
            "trained_at": d.get("run", {}).get("started_at"),
            "augmented": d.get("hyperparameters", {}).get("augment"),
            "training_images": d.get("hyperparameters", {}).get("train_images"),
            "accuracy_pct": d.get("final_metrics", {}).get("accuracy_pct"),
            "epochs_run": (d.get("early_stopping") or {}).get("epochs_run"),
        }
    return preds, meta


PREFIX = re.compile(r"^from-(train|val)__(.+)$")


def source_key(disk_name, class_name, by_name):
    """A file in the split, traced back to its source image.

    The dataset reuses filenames across its own train and test folders for
    different images, so make_grouped_split.py writes a collided name as
    from-<split>__<name>. That prefix makes the lookup exact; an unprefixed
    name is only safe when it belongs to one image."""
    m = PREFIX.match(disk_name)
    if m:
        return f"{m.group(1)}/{class_name}/{m.group(2)}"
    hits = by_name.get(disk_name, [])
    return hits[0] if len(hits) == 1 else None


def read_stage_two(predicted_csv, thresholds_json, cut):
    """Defect scores per casting, and the cutoff each flag is read at."""
    scores, by_name, hand = {}, {}, set()
    p = Path(predicted_csv)
    if p.is_file():
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                parts = norm(row["file"]).split("/")
                key = "/".join(parts[-3:])
                scores[key] = {fl: float(row[f"p_{fl}"]) for fl in FLAGS}
                by_name.setdefault(parts[-1], []).append(key)
                if row.get("hand_labelled") == "1":
                    hand.add(key)

    cuts = {fl: 0.5 for fl in FLAGS}
    meta = {"cutoffs": "0.5 for every flag (none measured yet)"}
    t = Path(thresholds_json)
    if cut == "half":
        meta = {"cutoffs": "0.5 for every flag (asked for)"}
    elif t.is_file():
        try:
            doc = json.loads(t.read_text())
            flags = doc["flags"]
            key = "threshold_f1" if cut == "f1" else "threshold_rate"
            cuts = {fl: float(flags[fl].get(key, flags[fl]["threshold"]))
                    for fl in FLAGS}
            meta = {"cutoffs": f"'{cut}' cutoffs from {t}",
                    "pos_weight": doc.get("pos_weight"),
                    "labels_behind_cutoffs": doc.get("labels")}
        except (ValueError, KeyError):
            pass
    return scores, by_name, hand, cuts, meta


def read_residuals(path):
    """How close each validation casting's nearest training image is."""
    out = {}
    p = Path(path) if path else None
    if p and p.is_file():
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                out[norm(row["val_file"]).rsplit("/", 1)[-1]] = {
                    "of": norm(row["closest_train_file"]).rsplit("/", 1)[-1],
                    "residual": round(float(row["residual"]), 2),
                }
    return out


def split_of(data_dir):
    """filename -> which folder of the split it sits in."""
    out = {}
    root = Path(data_dir)
    for split in ("train", "dev", "val"):
        for cls in CLASS_LABEL:
            d = root / split / cls
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file():
                        out[f.name] = (split, CLASS_LABEL[cls])
    return out


def summarise(castings, cuts, include_labelled=False):
    """Totals, counted from the records themselves so the two always agree.

    Castings stage two trained on are left out of the defect-type totals by
    default -- it knows their answers, so counting them flatters every rate."""
    from collections import Counter

    verdict = Counter(c["verdict"] or "not_inspected" for c in castings)
    label = Counter(c["label"] or "unknown" for c in castings)
    judged = [c for c in castings if c["correct"] is not None]
    correct = sum(1 for c in judged if c["correct"])
    missed = sum(1 for c in judged
                 if c["label"] == "defective" and c["verdict"] == "good")
    false_alarm = sum(1 for c in judged
                      if c["label"] == "good" and c["verdict"] == "defective")

    typed = [c for c in castings if c["defects"]
             and (include_labelled or not c.get("hand_labelled"))]
    present = Counter()
    for c in typed:
        for d in c["defects"]:
            if d["present"]:
                present[d["type"]] += 1
    combos = Counter(tuple(d["type"] for d in c["defects"] if d["present"])
                     for c in typed)
    n = len(typed)

    return {
        "castings": len(castings),
        "verdict": {k: verdict.get(k, 0)
                    for k in ("good", "defective", "not_inspected")},
        "label": {k: label.get(k, 0)
                  for k in ("good", "defective", "unknown")},
        "accuracy": {
            "judged": len(judged),
            "correct": correct,
            "incorrect": len(judged) - correct,
            "pct": round(100 * correct / len(judged), 2) if judged else None,
            "missed_defects": missed,
            "false_alarms": false_alarm,
        },
        "near_duplicates": sum(1 for c in castings if c["duplicate"]),
        "defect_types": {
            "assessed": n,
            "hand_labelled_excluded": (0 if include_labelled else
                                       sum(1 for c in castings if c["defects"]
                                           and c.get("hand_labelled"))),
            "counts": {fl: present.get(fl, 0) for fl in FLAGS},
            "rates_pct": {fl: round(100 * present.get(fl, 0) / n, 1)
                          for fl in FLAGS} if n else {},
            "cutoffs": {fl: round(cuts[fl], 2) for fl in FLAGS},
            "per_casting": {str(k): sum(1 for c in typed
                                        if c["defect_count"] == k)
                            for k in range(len(FLAGS) + 1)},
            "combinations": [
                {"types": list(t), "count": v,
                 "pct": round(100 * v / n, 1)}
                for t, v in combos.most_common(10)
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="a results/<run> directory; default is the newest")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data", default="data_grouped",
                    help="the split, used to say which folder a casting is in")
    ap.add_argument("--predicted", default="defect_types/predicted.csv")
    ap.add_argument("--thresholds",
                    default="defect_types/stage_two/thresholds.json")
    ap.add_argument("--cut", choices=["rate", "f1", "half"], default="rate",
                    help="which measured cutoff decides 'present'")
    ap.add_argument("--residuals", default="clean.csv",
                    help="verify_split.py --out CSV, to mark near-duplicates")
    ap.add_argument("--min-residual", type=float, default=5.7,
                    help="below this, a casting counts as a near-duplicate")
    ap.add_argument("--scope", choices=["inspected", "all"], default="inspected",
                    help="'inspected' is the castings stage one scored (the "
                         "validation split); 'all' adds every other casting "
                         "stage two has scores for, with a null verdict")
    ap.add_argument("--include-labelled", action="store_true",
                    help="count the hand-labelled castings in the summary's "
                         "defect-type totals. They are excluded by default, "
                         "since stage two trained on them; every record still "
                         "carries a hand_labelled field either way.")
    ap.add_argument("--out", default="results/inspection.json")
    ap.add_argument("--indent", type=int, default=1,
                    help="JSON indent; 0 writes one compact line")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else newest_run(args.results_dir)
    preds, one_meta = read_stage_one(run_dir)
    if not preds:
        raise SystemExit(f"No predictions.csv in {run_dir}. Point --run at a "
                         "run directory that has one.")
    scores, by_name, hand, cuts, two_meta = read_stage_two(
        args.predicted, args.thresholds, args.cut)
    dupes = read_residuals(args.residuals)
    splits = split_of(args.data)

    # Identity of a casting is its SOURCE path (split/class/file), because
    # that is what stage two's predictions are keyed by and what survives the
    # renaming the split does for colliding filenames. Disk name is what the
    # file is called inside the split.
    key_of_disk, disk_of_key, ambiguous = {}, {}, 0
    for disk, (split, lbl) in splits.items():
        cls = "def_front" if lbl == "defective" else "ok_front"
        k = source_key(disk, cls, by_name)
        if k:
            key_of_disk[disk] = k
            disk_of_key[k] = disk
        elif len(by_name.get(disk, [])) > 1:
            ambiguous += 1

    # value is the name on disk, or None for castings that are only in
    # stage two's predictions and not in this split
    entries = {}
    for disk in preds:
        entries[key_of_disk.get(disk, disk)] = disk
    if args.scope == "all":
        for k in scores:
            entries.setdefault(k, disk_of_key.get(k))

    castings, idents = [], []
    for ident, disk in sorted(entries.items()):
        from_split = ident.split("/") if "/" in ident else None
        name = disk or (from_split[-1] if from_split else ident)
        p = preds.get(disk) if disk else None
        split, folder_label = splits.get(disk, (None, None)) if disk else (None, None)
        if folder_label is None and from_split:
            folder_label = CLASS_LABEL.get(from_split[1])
        rec = {
            "id": stem(name),
            "file": name,
            "split": split,
            "label": (p or {}).get("label") or folder_label,
            "verdict": (p or {}).get("verdict"),
            "confidence": (p or {}).get("confidence"),
            "correct": (p or {}).get("correct"),
        }
        s2 = scores.get(ident) if "/" in ident else None
        if s2 is None:
            rec["defects"] = None
            rec["defect_count"] = None
        else:
            rec["defects"] = [
                {"type": fl, "score": round(s2[fl], 4),
                 "present": s2[fl] >= cuts[fl], "cutoff": round(cuts[fl], 2)}
                for fl in FLAGS
            ]
            rec["defect_count"] = sum(1 for d in rec["defects"] if d["present"])
        # true when this casting is one of the hand-labelled ones stage two
        # trained on: its defect scores are memory, not judgement
        rec["hand_labelled"] = ident in hand if "/" in ident else False
        d = dupes.get(disk) if disk else None
        rec["duplicate"] = (d if d and d["residual"] < args.min_residual
                            else None)
        castings.append(rec)
        idents.append(ident)

    # ids have to be unique for a front end to key rows by them, and the
    # source dataset reuses filenames across its train and test folders. Where
    # that happens, the source folder goes in front of the name.
    from collections import Counter
    seen = Counter(r["id"] for r in castings)
    for rec, ident in zip(castings, idents):
        if seen[rec["id"]] > 1 and "/" in ident:
            rec["id"] = f"{ident.split('/')[0]}-{rec['id']}"
    still = [k for k, v in Counter(r["id"] for r in castings).items() if v > 1]
    if still:
        raise SystemExit(f"Could not make ids unique: {still[:3]}. "
                         "Tell whoever wrote this script.")

    doc = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "stage_one": one_meta,
            "stage_two": two_meta,
            "split": args.data,
            "scope": args.scope,
            "near_duplicate_below": args.min_residual if dupes else None,
        },
        "summary": summarise(castings, cuts, args.include_labelled),
        "castings": castings,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=args.indent or None))

    scored = sum(1 for c in castings if c["verdict"])
    typed = sum(1 for c in castings if c["defects"] is not None)
    dup = sum(1 for c in castings if c["duplicate"])
    print(f"{out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  {len(castings)} castings")
    print(f"    {scored} with a good/bad verdict from {run_dir.name}")
    print(f"    {typed} with defect-type scores, read at {two_meta['cutoffs']}")
    print(f"    {dup} marked as near-duplicates of a training image")
    if ambiguous:
        print(f"    {ambiguous} left without scores: their filename belongs to "
              "two different\n      source images (a split built before the "
              "collision fix)")
    if not dupes:
        print(f"    (no {args.residuals}, so nothing is marked; run "
              "verify_split.py --out clean.csv)")
    sm = doc["summary"]
    print()
    print("  Summary in the file:")
    print(f"    verdict   good {sm['verdict']['good']}  "
          f"defective {sm['verdict']['defective']}  "
          f"not inspected {sm['verdict']['not_inspected']}")
    if sm["accuracy"]["judged"]:
        a = sm["accuracy"]
        print(f"    accuracy  {a['correct']}/{a['judged']} = {a['pct']}%  "
              f"({a['missed_defects']} defects missed, "
              f"{a['false_alarms']} false alarms)")
    dt = sm["defect_types"]
    if dt["assessed"]:
        print(f"    defects   over {dt['assessed']} castings: " + ", ".join(
            f"{fl} {dt['counts'][fl]} ({dt['rates_pct'][fl]}%)" for fl in FLAGS))
        print("    per part  " + ", ".join(
            f"{k} type(s): {v}" for k, v in dt["per_casting"].items() if v))
    print()
    print("  One record looks like this:")
    sample = next((c for c in castings if c["defects"] and c["verdict"]),
                  castings[0])
    for line in json.dumps(sample, indent=2).split("\n"):
        print("    " + line)
    print()
    print("  Fields are documented at the top of this script. Send your front")
    print("  end developer that docstring -- it is the contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
