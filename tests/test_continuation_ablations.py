from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ttt.research.continuation_ablations import (
    CANONICAL_PAPER_RUN_ID,
    ISO_TOTAL_EXTRA_TOKENS,
    STEP_INTERVAL,
    cumulative_wall_by_checkpoint,
    extra_tokens_for_steps,
    iso_total_extra_steps,
    should_stop_for_plateau,
)


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContinuationAblationsTest(unittest.TestCase):
    def test_budget_math_is_exact(self) -> None:
        self.assertEqual(extra_tokens_for_steps(STEP_INTERVAL), 31_457_280)
        self.assertEqual(iso_total_extra_steps(), 960)
        self.assertEqual(extra_tokens_for_steps(iso_total_extra_steps()), ISO_TOTAL_EXTRA_TOKENS)

    def test_plateau_detection_requires_three_small_improvements(self) -> None:
        self.assertFalse(should_stop_for_plateau([3.90, 3.88, 3.85, 3.82]))
        self.assertTrue(should_stop_for_plateau([3.90, 3.889, 3.878, 3.867]))

    def test_cumulative_wall_by_checkpoint_tracks_train_and_save_time(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            metrics_path = Path(td) / "metrics.jsonl"
            metrics_path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 480, "data_wait_seconds": 0.1, "batch_sharding_seconds": 0.2, "train_step_seconds": 1.0}),
                        json.dumps({"event": "checkpoint_saved", "step": 480, "checkpoint_save_seconds": 0.3}),
                        json.dumps({"step": 600, "data_wait_seconds": 0.2, "batch_sharding_seconds": 0.1, "train_step_seconds": 1.1}),
                        json.dumps({"event": "checkpoint_saved", "step": 600, "checkpoint_save_seconds": 0.4}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cumulative = cumulative_wall_by_checkpoint(metrics_path)
            self.assertAlmostEqual(cumulative[480], 1.6)
            self.assertAlmostEqual(cumulative[600], 3.4)

    def test_iso_quality_summary_prefers_first_target_hit(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_script_module(
            "summarize_125m_continuation_ablations_61",
            repo_root / "scripts" / "61_summarize_125m_continuation_ablations.py",
        )
        payload = {
            "status": "succeeded",
            "source": {
                "stage_id": "S2_125M",
                "run_id": "ext-125m-e2e-32K-from-fa-bridge",
                "checkpoint_step": 479,
                "target_loss_ce_mean": 3.2729225158691406,
            },
            "terminal": {"reason": "target_reached"},
            "frontier_points": [
                {"checkpoint_step": 479, "extra_steps": 0, "loss_ce_mean": 3.917266845703125, "matched_target": False},
                {"checkpoint_step": 599, "extra_steps": 120, "extra_tokens": 31_457_280, "extra_gpu_hours": 0.5, "loss_ce_mean": 3.26, "matched_target": True, "stage_id": "S2_125M_ISOQ", "run_id": "s2-125m-isoq-to0600", "total_branch_tokens": 1.0, "total_branch_gpu_hours": 2.0},
                {"checkpoint_step": 719, "extra_steps": 240, "extra_tokens": 62_914_560, "extra_gpu_hours": 1.0, "loss_ce_mean": 3.20, "matched_target": True, "stage_id": "S2_125M_ISOQ", "run_id": "s2-125m-isoq-to0720", "total_branch_tokens": 2.0, "total_branch_gpu_hours": 3.0},
            ],
        }

        summary = module.summarize_iso_quality_payload(payload)
        self.assertTrue(summary["matched_target"])
        self.assertFalse(summary["did_not_reach_target"])
        self.assertEqual(summary["first_matching_checkpoint_step"], 599)
        self.assertEqual(summary["steps_to_target"], 120)
        self.assertEqual(summary["tokens_to_target"], 31_457_280)

    def test_iso_total_tokens_summary_compares_against_canonical_s2(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_script_module(
            "summarize_125m_continuation_ablations_61",
            repo_root / "scripts" / "61_summarize_125m_continuation_ablations.py",
        )
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td) / "experiments"
            canonical_root = exp_dir / CANONICAL_PAPER_RUN_ID
            manifests = [
                ("S0_PRETRAIN_FA_125M", "pretrain-125m-fa", 10.0, 1_000, 3.5),
                ("S2_ADAPT_125M", "adapt-125m-e2e-8K-from-fa", 2.0, 200, 4.3),
                ("S2_125M", "ext-125m-e2e-32K-from-fa-bridge", 1.0, 100, 3.917266845703125),
                ("S3_PRETRAIN_E2E_125M", "pretrain-125m-e2e", 12.0, 1_000, 3.48),
                ("S3_125M", "ext-125m-e2e-32K", 1.5, 100, 3.2729225158691406),
            ]
            for stage_id, run_id, gpu_hours, tokens_seen, loss_ce_mean in manifests:
                run_dir = canonical_root / stage_id / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_result.json").write_text(
                    json.dumps(
                        {
                            "stage_id": stage_id,
                            "run_id": run_id,
                            "status": "succeeded",
                            "gpu_hours": gpu_hours,
                            "tokens_seen": tokens_seen,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (run_dir / "eval_manifest.json").write_text(
                    json.dumps(
                        {
                            "stage_id": stage_id,
                            "run_id": run_id,
                            "status": "succeeded",
                            "metrics": {
                                "loss_ce_mean": loss_ce_mean,
                                "loss_mean": loss_ce_mean,
                                "tokens_per_second_mean": 1.0,
                                "eval_wall_seconds": 1.0,
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            payload = {
                "status": "succeeded",
                "source": {
                    "stage_id": "S3_125M",
                    "run_id": "ext-125m-e2e-32K",
                    "checkpoint_step": 479,
                    "extra_steps_budget": 960,
                    "extra_tokens_budget": 251_658_240,
                },
                "terminal": {"reason": "equal_total_tokens_endpoint"},
                "frontier_points": [
                    {
                        "mode": "iso_total_tokens",
                        "stage_id": "S3_125M",
                        "run_id": "ext-125m-e2e-32K",
                        "checkpoint_step": 479,
                        "extra_steps": 0,
                        "extra_tokens": 0,
                        "extra_gpu_hours": 0.0,
                        "total_branch_tokens": 1_100.0,
                        "total_branch_gpu_hours": 13.5,
                        "total_branch_marginal_gpu_hours": 1.5,
                        "loss_ce_mean": 3.2729225158691406,
                    },
                    {
                        "mode": "iso_total_tokens",
                        "stage_id": "S3_125M_ISOTOK",
                        "run_id": "s3-125m-isotok-to1440",
                        "checkpoint_step": 1439,
                        "extra_steps": 960,
                        "extra_tokens": 251_658_240,
                        "extra_gpu_hours": 2.25,
                        "total_branch_tokens": 1_100.0 + 251_658_240,
                        "total_branch_gpu_hours": 15.75,
                        "total_branch_marginal_gpu_hours": 3.75,
                        "loss_ce_mean": 3.10,
                    },
                ],
            }

            summary = module.summarize_iso_total_tokens_payload(payload, exp_dir=exp_dir)
            self.assertTrue(summary["has_final_endpoint"])
            self.assertAlmostEqual(summary["canonical_s2_loss_ce_mean"], 3.917266845703125)
            self.assertLess(summary["quality_delta_vs_canonical_s2"], 0.0)
            self.assertIn("Scratch TTT-E2E still wins", summary["interpretation"])


if __name__ == "__main__":
    unittest.main()
