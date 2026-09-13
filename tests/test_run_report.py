import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ManufacturingNet.reporting import RunReport  # noqa: E402


class RunReportTests(unittest.TestCase):

    def _report(self):
        # 10 pieces. true:  6 ok, 4 defective
        #            pred:  1 defect missed, 1 good false-alarmed
        y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
        report = RunReport(model_name="TestNet",
                           class_names=["ok", "defective"])
        report.log_epoch(0, train_loss=0.5, train_accuracy=70.0,
                         val_loss=0.6, val_accuracy=65.0)
        report.set_predictions(y_true, y_pred)
        return report

    def test_headline_accuracy(self):
        metrics = self._report().compute_metrics()
        self.assertEqual(metrics["total_samples"], 10)
        self.assertEqual(metrics["correct"], 8)
        self.assertEqual(metrics["incorrect"], 2)
        self.assertEqual(metrics["accuracy_pct"], 80.0)

    def test_confusion_matrix_is_true_by_predicted(self):
        metrics = self._report().compute_metrics()
        # rows = true, cols = predicted
        # true ok (6):        5 predicted ok, 1 predicted defective
        # true defective (4): 1 predicted ok, 3 predicted defective
        self.assertEqual(metrics["confusion_matrix"], [[5, 1], [1, 3]])

    def test_inspection_summary_counts_and_percentages(self):
        summary = self._report().compute_metrics()["inspection_summary"]
        self.assertEqual(summary["defect_classes"], ["defective"])
        self.assertEqual(summary["pieces_inspected"], 10)
        self.assertEqual(summary["actual_damaged_count"], 4)
        self.assertEqual(summary["actual_damaged_pct"], 40.0)
        self.assertEqual(summary["model_flagged_damaged_count"], 4)
        self.assertEqual(summary["model_flagged_damaged_pct"], 40.0)
        self.assertEqual(summary["true_positive"], 3)
        self.assertEqual(summary["false_negative"], 1)
        self.assertEqual(summary["false_positive"], 1)
        self.assertEqual(summary["true_negative"], 5)
        # 1 of 4 genuine defects let through
        self.assertEqual(summary["missed_damaged_pct"], 25.0)
        # 1 of 6 good pieces wrongly rejected
        self.assertEqual(summary["false_alarm_pct"], 16.67)

    def test_defect_class_autodetection(self):
        report = RunReport(model_name="TestNet",
                           class_names=["good", "damaged_casting"])
        report.set_predictions([0, 1], [0, 1])
        summary = report.compute_metrics()["inspection_summary"]
        self.assertEqual(summary["defect_classes"], ["damaged_casting"])

    def test_no_defect_class_gives_no_inspection_summary(self):
        report = RunReport(model_name="TestNet",
                           class_names=["cat", "dog"])
        report.set_predictions([0, 1], [0, 1])
        self.assertIsNone(
            report.compute_metrics()["inspection_summary"])

    def test_mismatched_lengths_are_rejected(self):
        report = RunReport(model_name="TestNet", class_names=["a", "b"])
        with self.assertRaises(ValueError):
            report.set_predictions([0, 1, 0], [0, 1])

    def test_save_writes_readable_json_and_csv(self):
        report = self._report()
        with tempfile.TemporaryDirectory() as tmp:
            paths = report.save(tmp)
            document = json.loads(Path(paths["results_json"]).read_text())
            self.assertEqual(document["schema_version"], "1.0")
            self.assertEqual(document["run"]["model"], "TestNet")
            self.assertEqual(document["final_metrics"]["accuracy_pct"], 80.0)
            self.assertEqual(len(document["epochs"]), 1)

            csv_text = Path(paths["summary_csv"]).read_text()
            self.assertIn("epoch,train_loss", csv_text)
            self.assertEqual(len(csv_text.strip().split("\n")), 2)


if __name__ == "__main__":
    unittest.main()
