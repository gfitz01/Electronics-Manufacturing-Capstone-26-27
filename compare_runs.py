"""Compare two (or more) training runs side by side.

    python compare_runs.py                       compares the two most recent
    python compare_runs.py results/a results/b   compares specific runs

Reads each run's results.json and prints the headline metrics, the per-class
figures and the inspection summary in aligned columns, so the effect of
whatever you changed is visible without opening the files.

Also flags the two things that decide whether a comparison means anything:
whether the runs share their hyperparameters, and whether the difference in
error count is larger than the statistical noise.
"""

import json
import math
import sys
from pathlib import Path

# Hyperparameters that must match for the comparison to isolate one variable.
CONTROLLED = ["epochs", "batch_size", "learning_rate", "optimizer",
              "image_size", "pretrained", "train_images", "val_images"]


def run_started_at(run_dir):
    """When the run actually started.

    Sorting by folder name is wrong: 'alexnet-aug-...' sorts after
    'alexnet-...' alphabetically no matter which ran first, so the two most
    recent runs by name can both be augmented ones. Use the timestamp the run
    itself recorded, falling back to the directory mtime.
    """
    try:
        with open(run_dir / "results.json") as f:
            started = json.load(f).get("run", {}).get("started_at")
        if started:
            return started
    except (OSError, ValueError):
        pass
    return str(run_dir.stat().st_mtime)


def find_recent_runs(n=2, root="results"):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"No {root}/ directory here. Run a training job first.")
    runs = sorted((p for p in root.iterdir() if (p / "results.json").is_file()),
                  key=run_started_at)
    if len(runs) < n:
        raise SystemExit(
            f"Found {len(runs)} run(s) under {root}/, need at least {n}.")
    chosen = runs[-n:]
    print("Comparing the %d most recent runs by start time:" % n)
    for r in chosen:
        print(f"    {r.name}")
    print()
    return chosen


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    if not path.is_file():
        raise SystemExit(f"No results.json at {path}")
    with open(path) as f:
        return path, json.load(f)


def wilson_interval(errors, n, z=1.96):
    """95% interval on the error COUNT, via the Wilson score interval."""
    if n == 0:
        return 0, 0
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, round((centre - half) * n)), round((centre + half) * n)


def row(label, values, width):
    cells = "".join(f"{str(v):>{width}}" for v in values)
    print(f"  {label:<26}{cells}")


def main():
    args = sys.argv[1:]
    paths = args if args else find_recent_runs(2)

    loaded = [load(p) for p in paths]
    names, docs = [], []
    for path, doc in loaded:
        names.append(path.parent.name if path.name == "results.json"
                     else path.name)
        docs.append(doc)

    width = max(18, max(len(n) for n in names) + 2)

    print("=" * (28 + width * len(names)))
    print("RUN COMPARISON")
    print("=" * (28 + width * len(names)))
    row("", names, width)
    print()

    # -- is this a fair comparison? --------------------------------------
    differences = []
    unrecorded = []
    for key in CONTROLLED:
        values = [d.get("hyperparameters", {}).get(key) for d in docs]
        # None means the field was not recorded by that version of the code,
        # not that the setting differed. Older runs predate some fields.
        known = [v for v in values if v is not None]
        if len(known) < len(values):
            if known and len({json.dumps(v, sort_keys=True)
                              for v in known}) == 1:
                unrecorded.append(key)
                continue
        if len({json.dumps(v, sort_keys=True) for v in values}) > 1 and known:
            if len({json.dumps(v, sort_keys=True) for v in known}) > 1:
                differences.append((key, values))
            else:
                unrecorded.append(key)

    aug = [d.get("hyperparameters", {}).get("augment") for d in docs]
    row("augmentation", ["ON" if a else "off" for a in aug], width)
    print()

    # -- headline --------------------------------------------------------
    metrics = [d["final_metrics"] for d in docs]
    row("accuracy %", [m["accuracy_pct"] for m in metrics], width)
    row("macro F1", [m["macro_f1"] for m in metrics], width)
    row("errors", [m["incorrect"] for m in metrics], width)
    row("samples", [m["total_samples"] for m in metrics], width)
    print()

    # -- inspection ------------------------------------------------------
    summaries = [m.get("inspection_summary") for m in metrics]
    if all(summaries):
        row("actual damaged %", [s["actual_damaged_pct"] for s in summaries],
            width)
        row("flagged damaged %",
            [s["model_flagged_damaged_pct"] for s in summaries], width)
        row("missed defects", [s["false_negative"] for s in summaries], width)
        row("missed damaged %", [s["missed_damaged_pct"] for s in summaries],
            width)
        row("false alarms", [s["false_positive"] for s in summaries], width)
        row("false alarm %", [s["false_alarm_pct"] for s in summaries], width)
        print()

    # -- per class -------------------------------------------------------
    class_names = metrics[0]["class_names"]
    for i, cls in enumerate(class_names):
        per = [m["per_class"][i] for m in metrics]
        row(f"{cls} precision", [f"{p['precision']:.4f}" for p in per], width)
        row(f"{cls} recall", [f"{p['recall']:.4f}" for p in per], width)
    print()

    # -- training --------------------------------------------------------
    row("epochs run", [len(d["epochs"]) for d in docs], width)
    row("final train loss",
        [f"{d['epochs'][-1]['train_loss']:.4f}" for d in docs], width)
    row("final val loss",
        [f"{d['epochs'][-1]['val_loss']:.4f}" for d in docs], width)
    best = [min(e["val_loss"] for e in d["epochs"]) for d in docs]
    row("best val loss", [f"{b:.4f}" for b in best], width)
    best_epoch = [min(d["epochs"], key=lambda e: e["val_loss"])["epoch"]
                  for d in docs]
    row("best val loss at epoch", best_epoch, width)
    print()

    # -- how to read it --------------------------------------------------
    print("-" * (28 + width * len(names)))
    if unrecorded:
        print("Note: " + ", ".join(unrecorded) + " was not recorded by the")
        print("code version that produced the older run. Where it is recorded")
        print("the runs agree, so this is a logging gap, not a real")
        print("difference in settings.")
        print()
    if differences:
        print("WARNING: these runs differ in more than one setting, so any")
        print("difference below cannot be attributed to augmentation alone:")
        for key, values in differences:
            print(f"    {key}: {' vs '.join(str(v) for v in values)}")
        print()

    n = metrics[0]["total_samples"]
    if all(m["total_samples"] == n for m in metrics):
        print(f"Noise check (Wilson 95% interval on {n} samples):")
        spans = []
        for name, m in zip(names, metrics):
            lo, hi = wilson_interval(m["incorrect"], n)
            spans.append((lo, hi))
            print(f"    {name}: {m['incorrect']} errors "
                  f"(consistent with {lo}-{hi})")
        if len(spans) == 2 and not (spans[0][1] < spans[1][0] or
                                    spans[1][1] < spans[0][0]):
            print()
            print("    The intervals OVERLAP. Whatever difference you see in")
            print("    accuracy is within noise -- it is not evidence that one")
            print("    run is better. Report robustness, not accuracy.")
        elif len(spans) == 2:
            print()
            print("    The intervals do NOT overlap. That is a real difference.")
    print("-" * (28 + width * len(names)))


if __name__ == "__main__":
    main()
