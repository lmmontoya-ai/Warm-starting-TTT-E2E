from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.types import BudgetSpec, EvalSpec, StageSpec


class OrchestratorIntegrationTest(unittest.TestCase):
    def test_run_stage_dry_run_writes_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp_dir = root / "experiments"
            ckpt_root = root / "checkpoints"
            profile_root = root / "profiles"
            dclm_root = root / "dclm"
            books_root = root / "books"
            for path in (exp_dir, ckpt_root, profile_root, dclm_root, books_root):
                path.mkdir(parents=True, exist_ok=True)

            stage = StageSpec(
                stage_id="S0",
                kind="pretrain",
                model_key="m1",
                path_group="scratch",
                experiment="760m/pretrain/pretrain-760m-fa",
                exp_name="pretrain-760m-fa",
                dataset_ids=["dclm_filter_8k"],
            )
            stage_map = {stage.stage_id: stage}
            opts = OrchestratorOptions(
                deploy="interactive",
                runtime_mode="simulate",
                exp_dir=exp_dir,
                checkpoint_root=ckpt_root,
                profile_root=profile_root,
                dclm_root=dclm_root,
                books_root=books_root,
                exp_folder="paper",
                wandb_entity="none",
                wandb_project="none",
                wandb_key="none",
                global_batch_size=2,
                seq_length=16,
                save_milestone_freq=1,
                dummy_dataset=True,
                dry_run=True,
                paper_run_id="paper1",
            )
            budget = BudgetSpec(budget_id="b1", pretrain_steps=2, adapt_steps=1, ext_steps=1)
            eval_spec = EvalSpec(eval_id="e1")
            repo_root = Path(__file__).resolve().parents[1]

            result = run_stage(
                stage=stage,
                stage_map=stage_map,
                opts=opts,
                budget=budget,
                eval_spec=eval_spec,
                repo_root=repo_root,
                run_id="run-s0",
            )

            run_dir = exp_dir / "paper1" / "S0" / "run-s0"
            self.assertEqual(result.status, "dry_run")
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "stage_manifest.json").exists())
            self.assertTrue((run_dir / "budget_manifest.json").exists())
            self.assertTrue((run_dir / "events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
