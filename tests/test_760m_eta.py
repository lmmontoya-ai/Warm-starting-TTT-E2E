from __future__ import annotations

import unittest

from ttt.research.eta_760m import estimate_stage_training_wall_seconds, step_wall_seconds


class Eta760MTest(unittest.TestCase):
    def test_step_wall_seconds_sums_components(self) -> None:
        row = {
            "data_wait_seconds": 0.2,
            "batch_sharding_seconds": 0.3,
            "train_step_seconds": 1.5,
        }
        self.assertAlmostEqual(step_wall_seconds(row), 2.0)

    def test_estimate_stage_training_wall_seconds_uses_post_warmup_median(self) -> None:
        rows = [
            {"step": 0, "data_wait_seconds": 0.5, "batch_sharding_seconds": 0.5, "train_step_seconds": 9.0},
            {"step": 1, "data_wait_seconds": 0.5, "batch_sharding_seconds": 0.5, "train_step_seconds": 7.0},
            {"step": 2, "data_wait_seconds": 0.1, "batch_sharding_seconds": 0.2, "train_step_seconds": 1.0},
            {"step": 3, "data_wait_seconds": 0.1, "batch_sharding_seconds": 0.2, "train_step_seconds": 1.2},
            {"step": 4, "data_wait_seconds": 0.1, "batch_sharding_seconds": 0.2, "train_step_seconds": 0.8},
        ]
        estimate = estimate_stage_training_wall_seconds(
            metrics_rows=rows,
            total_steps=10,
            warmup_steps=2,
        )
        self.assertEqual(estimate["warmup_steps"], 2.0)
        self.assertAlmostEqual(estimate["warmup_wall_seconds"], 18.0)
        self.assertAlmostEqual(estimate["steady_step_wall_seconds"], 1.3)
        self.assertAlmostEqual(estimate["estimated_training_wall_seconds"], 18.0 + 1.3 * 8.0)


if __name__ == "__main__":
    unittest.main()
