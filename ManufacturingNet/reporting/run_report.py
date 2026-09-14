"""Structured, machine-readable output for a training run.

The library previously emitted its results as console prints plus two PNG
plots.  That is fine to look at once and useless to build on: nothing
downstream can read a number out of a picture of a loss curve.

``RunReport`` collects the same information the training loop already has and
writes it as ``results.json`` (full detail) and ``summary.csv`` (per-epoch
table).  The JSON is the contract for anything built on top of it.

Typical use inside a training loop::

    report = RunReport(model_name="AlexNet",
                       class_names=train_dataset.classes,
                       hyperparameters={"epochs": 10, "batch_size": 32})

    for epoch in range(epochs):
        ...
        report.log_epoch(epoch, train_loss, train_accuracy,
                         val_loss, val_accuracy)

    report.set_predictions(y_true, y_pred, confidences, file_paths)
    paths = report.save("results")
"""

import csv
import json
import platform
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Substrings that mark a class as a "reject" for the inspection summary.
# Override by passing defect_classes= explicitly.
_DEFECT_HINTS = (
    "def",      # defect, defective
    "damag",    # damaged
    "bad",
    "fail",
    "reject",
    "crack",
    "flaw",
    "scratch",
    "porosity",
    "ng",       # "no good", common in manufacturing datasets
)


def _to_1d_int_array(values, name):
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array.astype(int)


class RunReport:
    """Collects run metadata, per-epoch history and final metrics."""

    SCHEMA_VERSION = "1.0"

    def __init__(self, model_name, class_names=None, hyperparameters=None,
                 defect_classes=None, notes=None):
        self.model_name = model_name
        self.class_names = list(class_names) if class_names else None
        self.hyperparameters = dict(hyperparameters or {})
        self.defect_classes = (
            list(defect_classes) if defect_classes is not None else None
        )
        self.notes = notes

        self.started_at = datetime.now(timezone.utc)
        self.epochs = []
        self.y_true = None
        self.y_pred = None
        self.confidences = None
        self.file_paths = None

    # -- collection ------------------------------------------------------

    def log_epoch(self, epoch, train_loss, train_accuracy=None,
                  val_loss=None, val_accuracy=None, seconds=None,
                  learning_rate=None):
        """Record one epoch of training."""
        entry = OrderedDict()
        entry["epoch"] = int(epoch)
        entry["train_loss"] = _round(train_loss)
        entry["train_accuracy_pct"] = _round(train_accuracy)
        entry["val_loss"] = _round(val_loss)
        entry["val_accuracy_pct"] = _round(val_accuracy)
        entry["seconds"] = _round(seconds, 2)
        entry["learning_rate"] = learning_rate
        self.epochs.append(entry)
        return entry

    def set_predictions(self, y_true, y_pred, confidences=None,
                        file_paths=None):
        """Record the final validation predictions the metrics derive from."""
        self.y_true = _to_1d_int_array(y_true, "y_true")
        self.y_pred = _to_1d_int_array(y_pred, "y_pred")

        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"y_true has {self.y_true.size} entries but y_pred has "
                f"{self.y_pred.size}"
            )

        if confidences is not None:
            self.confidences = np.asarray(confidences).reshape(-1).astype(float)
        if file_paths is not None:
            self.file_paths = [str(p) for p in file_paths]

    # -- derived metrics -------------------------------------------------

    def _resolve_class_names(self):
        if self.class_names:
            return self.class_names
        n = int(max(self.y_true.max(), self.y_pred.max())) + 1
        return [f"class_{i}" for i in range(n)]

    def _resolve_defect_indices(self, class_names):
        if self.defect_classes is not None:
            return [class_names.index(c) for c in self.defect_classes
                    if c in class_names]
        return [
            i for i, name in enumerate(class_names)
            if any(hint in name.lower() for hint in _DEFECT_HINTS)
        ]

    def compute_metrics(self):
        """Return the final evaluation block."""
        if self.y_true is None:
            return None

        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        class_names = self._resolve_class_names()
        labels = list(range(len(class_names)))

        # NOTE: sklearn's signature is (y_true, y_pred) -- the original
        # get_confusion_matrix() passed these the other way round, which
        # transposed the matrix and swapped precision with recall.
        matrix = confusion_matrix(self.y_true, self.y_pred, labels=labels)

        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=labels, zero_division=0
        )

        total = int(self.y_true.size)
        correct = int((self.y_true == self.y_pred).sum())

        per_class = []
        for i, name in enumerate(class_names):
            true_count = int((self.y_true == i).sum())
            pred_count = int((self.y_pred == i).sum())
            per_class.append(OrderedDict([
                ("class_index", i),
                ("class_name", name),
                ("true_count", true_count),
                ("true_pct", _pct(true_count, total)),
                ("predicted_count", pred_count),
                ("predicted_pct", _pct(pred_count, total)),
                ("precision", _round(float(precision[i]), 4)),
                ("recall", _round(float(recall[i]), 4)),
                ("f1", _round(float(f1[i]), 4)),
                ("support", int(support[i])),
            ]))

        metrics = OrderedDict()
        metrics["total_samples"] = total
        metrics["correct"] = correct
        metrics["incorrect"] = total - correct
        metrics["accuracy_pct"] = _pct(correct, total)
        metrics["macro_f1"] = _round(float(np.mean(f1)), 4)
        metrics["class_names"] = class_names
        metrics["confusion_matrix"] = matrix.tolist()
        metrics["confusion_matrix_orientation"] = "rows=true, cols=predicted"
        metrics["per_class"] = per_class
        metrics["inspection_summary"] = self._inspection_summary(
            class_names, total
        )
        return metrics

    def _inspection_summary(self, class_names, total):
        """The headline numbers: how much of the batch is being rejected."""
        defect_indices = self._resolve_defect_indices(class_names)
        if not defect_indices:
            return None

        actual_defects = int(np.isin(self.y_true, defect_indices).sum())
        flagged_defects = int(np.isin(self.y_pred, defect_indices).sum())

        true_is_defect = np.isin(self.y_true, defect_indices)
        pred_is_defect = np.isin(self.y_pred, defect_indices)

        true_positive = int((true_is_defect & pred_is_defect).sum())
        false_positive = int((~true_is_defect & pred_is_defect).sum())
        false_negative = int((true_is_defect & ~pred_is_defect).sum())
        true_negative = int((~true_is_defect & ~pred_is_defect).sum())

        summary = OrderedDict()
        summary["defect_classes"] = [class_names[i] for i in defect_indices]
        summary["pieces_inspected"] = total
        summary["actual_damaged_count"] = actual_defects
        summary["actual_damaged_pct"] = _pct(actual_defects, total)
        summary["model_flagged_damaged_count"] = flagged_defects
        summary["model_flagged_damaged_pct"] = _pct(flagged_defects, total)
        summary["true_positive"] = true_positive
        summary["false_positive"] = false_positive
        summary["false_negative"] = false_negative
        summary["true_negative"] = true_negative
        # missed_damaged_pct is the one that matters for a scrap line: the
        # share of genuinely damaged pieces the model let through.
        summary["missed_damaged_pct"] = _pct(false_negative, actual_defects)
        summary["false_alarm_pct"] = _pct(false_positive,
                                          total - actual_defects)
        return summary

    def _predictions_block(self):
        if self.y_true is None:
            return None
        class_names = self._resolve_class_names()
        rows = []
        for i in range(self.y_true.size):
            row = OrderedDict()
            if self.file_paths is not None and i < len(self.file_paths):
                row["file"] = self.file_paths[i]
            row["true"] = class_names[int(self.y_true[i])]
            row["predicted"] = class_names[int(self.y_pred[i])]
            row["correct"] = bool(self.y_true[i] == self.y_pred[i])
            if self.confidences is not None and i < self.confidences.size:
                row["confidence"] = _round(float(self.confidences[i]), 4)
            rows.append(row)
        return rows

    # -- output ----------------------------------------------------------

    def to_dict(self, include_predictions=True):
        finished_at = datetime.now(timezone.utc)
        document = OrderedDict()
        document["schema_version"] = self.SCHEMA_VERSION
        document["run"] = OrderedDict([
            ("model", self.model_name),
            ("started_at", self.started_at.isoformat()),
            ("finished_at", finished_at.isoformat()),
            ("duration_seconds",
             _round((finished_at - self.started_at).total_seconds(), 2)),
            ("python", sys.version.split()[0]),
            ("platform", platform.platform()),
            ("notes", self.notes),
        ])
        document["hyperparameters"] = self.hyperparameters
        document["epochs"] = self.epochs
        document["final_metrics"] = self.compute_metrics()
        if include_predictions:
            document["predictions"] = self._predictions_block()
        return document

    def save(self, output_dir="results", include_predictions=True):
        """Write results.json and summary.csv. Returns the paths written."""
        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(output_dir) / f"{self.model_name.lower()}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        document = self.to_dict(include_predictions=include_predictions)

        json_path = run_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(document, f, indent=2)

        csv_path = run_dir / "summary.csv"
        fieldnames = ["epoch", "train_loss", "train_accuracy_pct",
                      "val_loss", "val_accuracy_pct", "seconds",
                      "learning_rate"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.epochs:
                writer.writerow({k: entry.get(k) for k in fieldnames})

        paths = {"results_json": str(json_path), "summary_csv": str(csv_path)}

        if include_predictions and document.get("predictions"):
            predictions_path = run_dir / "predictions.csv"
            keys = list(document["predictions"][0].keys())
            with open(predictions_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(document["predictions"])
            paths["predictions_csv"] = str(predictions_path)

        print(f"Results written to {run_dir}")
        return paths

    def print_summary(self):
        """Human-readable version of the headline numbers."""
        metrics = self.compute_metrics()
        if metrics is None:
            print("No predictions recorded.")
            return

        print("=" * 45)
        print(f"Run summary: {self.model_name}")
        print("=" * 45)
        print(f"Pieces inspected : {metrics['total_samples']}")
        print(f"Accuracy         : {metrics['accuracy_pct']}%")
        print(f"Macro F1         : {metrics['macro_f1']}")

        inspection = metrics.get("inspection_summary")
        if inspection:
            print("-" * 45)
            print(f"Actually damaged : {inspection['actual_damaged_count']} "
                  f"({inspection['actual_damaged_pct']}%)")
            print(f"Flagged damaged  : "
                  f"{inspection['model_flagged_damaged_count']} "
                  f"({inspection['model_flagged_damaged_pct']}%)")
            print(f"Missed damaged   : {inspection['false_negative']} "
                  f"({inspection['missed_damaged_pct']}% of damaged)")
            print(f"False alarms     : {inspection['false_positive']} "
                  f"({inspection['false_alarm_pct']}% of good)")
        print("=" * 45)


def _round(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def _pct(part, whole):
    if not whole:
        return 0.0
    return round(100.0 * part / whole, 2)
