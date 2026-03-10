from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ttt.research.ruler_runner import compute_ruler_metrics_from_eval_csv


class RulerRunnerTest(unittest.TestCase):
    def test_compute_ruler_metrics_from_niah_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "eval_raw.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "record_type",
                        "status",
                        "niah_accuracy",
                        "accuracy",
                        "context_length",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_type": "niah_proxy",
                        "status": "ok",
                        "niah_accuracy": "0.40",
                        "accuracy": "",
                        "context_length": "8192",
                    }
                )
                writer.writerow(
                    {
                        "record_type": "niah_proxy",
                        "status": "ok",
                        "niah_accuracy": "",
                        "accuracy": "0.60",
                        "context_length": "8192",
                    }
                )
                writer.writerow(
                    {
                        "record_type": "niah_proxy",
                        "status": "ok",
                        "niah_accuracy": "0.50",
                        "accuracy": "",
                        "context_length": "32768",
                    }
                )
                # ignored: non-ok row
                writer.writerow(
                    {
                        "record_type": "niah_proxy",
                        "status": "failed",
                        "niah_accuracy": "1.00",
                        "accuracy": "",
                        "context_length": "8192",
                    }
                )
                # ignored: different record type
                writer.writerow(
                    {
                        "record_type": "decode_proxy",
                        "status": "ok",
                        "niah_accuracy": "1.00",
                        "accuracy": "",
                        "context_length": "8192",
                    }
                )

            metrics = compute_ruler_metrics_from_eval_csv(csv_path)
            self.assertAlmostEqual(metrics["ruler_accuracy_mean"], 0.5, places=6)
            self.assertAlmostEqual(metrics["ruler_by_length_8192"], 0.5, places=6)
            self.assertAlmostEqual(metrics["ruler_by_length_32768"], 0.5, places=6)

    def test_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "missing.csv"
            metrics = compute_ruler_metrics_from_eval_csv(csv_path)
            self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
