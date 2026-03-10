from __future__ import annotations

import unittest

from ttt.research.budget import build_budget_manifest, estimate_gpu_hours_from_wall, estimate_tokens
from ttt.research.types import BudgetSpec


class BudgetTest(unittest.TestCase):
    def test_estimates(self) -> None:
        self.assertEqual(estimate_tokens(seq_length=128, global_batch_size=4, total_steps=10), 5120)
        self.assertAlmostEqual(estimate_gpu_hours_from_wall(wall_seconds=7200, num_gpus=2), 4.0)

    def test_budget_manifest_caps(self) -> None:
        spec = BudgetSpec(budget_id="b", token_cap=100, gpu_hours_cap=1.0)
        manifest = build_budget_manifest(
            budget_spec=spec,
            tokens_planned=50,
            gpu_hours_planned=0.5,
            tokens_observed=120,
            gpu_hours_observed=1.5,
        )
        self.assertTrue(manifest["token_cap_exceeded"])
        self.assertTrue(manifest["gpu_hours_cap_exceeded"])


if __name__ == "__main__":
    unittest.main()
