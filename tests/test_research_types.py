from __future__ import annotations

import unittest

from ttt.research.types import BudgetSpec, CheckpointRef, DatasetRef, EvalSpec, StageSpec


class ResearchTypesTest(unittest.TestCase):
    def test_checkpoint_round_trip(self) -> None:
        ckpt = CheckpointRef(
            checkpoint_id="S0",
            exp_folder="paper",
            exp_name="run-a",
            step=42,
            checkpoint_path="/tmp/ckpt.json",
            payload_sha256="abc123",
        )
        payload = ckpt.to_dict()
        rebuilt = CheckpointRef.from_dict(payload)
        self.assertEqual(rebuilt, ckpt)

    def test_budget_from_dict(self) -> None:
        payload = {
            "budget_id": "b1",
            "pretrain_steps": 10,
            "adapt_steps": 5,
            "ext_steps": 2,
            "gpu_hours_cap": 4.5,
            "token_cap": 1000,
            "seed": 7,
        }
        budget = BudgetSpec.from_dict(payload)
        self.assertEqual(budget.budget_id, "b1")
        self.assertEqual(budget.pretrain_steps, 10)
        self.assertEqual(budget.gpu_hours_cap, 4.5)
        self.assertEqual(budget.seed, 7)

    def test_dataset_eval_stage_defaults(self) -> None:
        dataset = DatasetRef(dataset_id="dclm", path="/tmp/data", split="train")
        eval_spec = EvalSpec(eval_id="e1", contexts=[8192], datasets=["dclm"])
        stage = StageSpec(stage_id="S0", dataset_ids=["dclm"], experiment="x/y")

        self.assertEqual(dataset.dataset_id, "dclm")
        self.assertEqual(eval_spec.contexts, [8192])
        self.assertEqual(stage.stage_id, "S0")


if __name__ == "__main__":
    unittest.main()
