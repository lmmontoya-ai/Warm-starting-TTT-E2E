from __future__ import annotations

import unittest

from ttt.research.reporting import aggregate_stage_metrics, stage_delta_table


class ReportingTest(unittest.TestCase):
    def test_aggregate_and_delta(self) -> None:
        manifests = [
            {"stage_id": "S0", "status": "succeeded", "metrics": {"loss_mean": 1.0}},
            {"stage_id": "S0", "status": "succeeded", "metrics": {"loss_mean": 1.2}},
            {"stage_id": "S1", "status": "succeeded", "metrics": {"loss_mean": 1.5}},
            {"stage_id": "S1", "status": "succeeded", "metrics": {"loss_mean": 1.7}},
        ]
        summary = aggregate_stage_metrics(manifests, metric="loss_mean")
        self.assertIn("S0", summary)
        self.assertIn("S1", summary)
        row = stage_delta_table(
            paper_run_id="p1",
            metric="loss_mean",
            from_stage="S0",
            to_stage="S1",
            stage_summary=summary,
        )
        self.assertAlmostEqual(row.delta, 0.5)


if __name__ == "__main__":
    unittest.main()
