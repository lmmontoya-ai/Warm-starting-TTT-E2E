"""scripts/80_analyze_e1_bridge_effect.py on synthetic eval summaries."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_e1_80", REPO_ROOT / "scripts" / "80_analyze_e1_bridge_effect.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load()


def _row(stage: str, run_id: str, loss: float, status: str = "succeeded") -> dict:
    return {"paper_run_id": "p", "stage_id": stage, "run_id": run_id, "status": status, "loss_mean": loss}


def _summary(rows: list[dict]) -> dict:
    return {"schema_version": "1.0", "paper_run_id": "revision_v3_test", "eval_id": "jax_books32k_eval64", "rows": rows}


S2 = "ext-125m-e2e-32K-from-fa-bridge"
S2M = "ext-125m-e2e-32K-from-fa-direct"


class E1AnalysisTest(unittest.TestCase):
    def test_lr_sweep_is_resolved_by_variant_and_shared_s2_minus(self) -> None:
        rows = [_row("S0_125M", "ext-125m-fa-32K-seed001", 3.0)]
        for seed in (1, 2, 3):
            rows += [
                _row("S2_125M", f"{S2}-lr1e-3-seed{seed:03d}", 3.20 + 0.01 * seed),
                _row("S2_125M", f"{S2}-lr3e-3-seed{seed:03d}", 3.40),
                _row("S2_MINUS_125M", f"{S2M}-seed{seed:03d}", 3.60 + 0.01 * seed),
            ]
        with self.assertRaisesRegex(ValueError, "Pass --variant"):
            analyze.paired_losses(_summary(rows))
        seeds, s2, s2m = analyze.paired_losses(_summary(rows), variant="lr1e-3")
        self.assertEqual(seeds, [1, 2, 3])
        self.assertEqual(s2, [3.21, 3.22, 3.23])
        self.assertEqual(s2m, [3.61, 3.62, 3.63])

    def test_cli_writes_decision_and_tables(self) -> None:
        rows = []
        for seed, (s2, s2m) in enumerate([(3.20, 3.50), (3.25, 3.52), (3.22, 3.55)], start=1):
            rows += [_row("S2_125M", f"{S2}-seed{seed:03d}", s2), _row("S2_MINUS_125M", f"{S2M}-seed{seed:03d}", s2m)]
        with tempfile.TemporaryDirectory() as td:
            summary_path = Path(td) / "books32k.json"
            summary_path.write_text(json.dumps(_summary(rows)), encoding="utf-8")
            run_summary = Path(td) / "ladder.json"
            run_summary.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"stage_id": "S2_125M", "status": "succeeded", "gpu_hours": 1.5},
                            {"stage_id": "S2_125M", "status": "skipped_complete_checkpoint"},
                            {"stage_id": "S2_MINUS_125M", "status": "resumed_succeeded", "gpu_hours": 0.5},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(td) / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                code = analyze.main(
                    [
                        "--eval-summary", str(summary_path),
                        "--run-summary", str(run_summary),
                        "--out-dir", str(out_dir),
                        "--bootstrap-resamples", "500",
                    ]
                )
            self.assertEqual(code, 0)
            analysis = json.loads((out_dir / "e1_bridge_effect_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["paper_run_id"], "revision_v3_test")
            self.assertEqual(analysis["seeds"], [1, 2, 3])
            self.assertAlmostEqual(analysis["delta_mean"], (0.30 + 0.27 + 0.33) / 3)
            self.assertEqual(analysis["decision"], "helps")
            self.assertAlmostEqual(analysis["training_gpu_hours"]["total"], 2.0)
            self.assertTrue((out_dir / "e1_bridge_effect_paired_losses.csv").is_file())
            self.assertIn("Decision: `helps`", (out_dir / "e1_bridge_effect_analysis.md").read_text(encoding="utf-8"))

    def test_too_few_or_unpaired_seeds_fail(self) -> None:
        rows = [
            _row("S2_125M", f"{S2}-seed001", 3.2),
            _row("S2_MINUS_125M", f"{S2M}-seed001", 3.5),
            _row("S2_125M", f"{S2}-seed002", 3.2),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "at least 3 paired seeds"):
                analyze.paired_losses(_summary(rows))
        with self.assertRaisesRegex(ValueError, "Missing paired losses"):
            analyze.paired_losses(_summary(rows), seeds=[1, 2], min_seeds=1)

    def test_failed_eval_rows_are_refused(self) -> None:
        rows = [
            _row("S2_125M", f"{S2}-seed001", 3.2, status="failed"),
            _row("S2_MINUS_125M", f"{S2M}-seed001", 3.5),
        ]
        with self.assertRaisesRegex(ValueError, "failed eval row"):
            analyze.paired_losses(_summary(rows), min_seeds=1)


if __name__ == "__main__":
    unittest.main()
