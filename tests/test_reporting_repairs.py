from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ttt.jax_runtime.loop import M, _aggregate_metric_history
from ttt.research.orchestrator import observed_tokens_from_runtime_artifacts


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReportingRepairsTest(unittest.TestCase):
    def test_aggregate_metric_history_uses_float32_mean(self) -> None:
        first = np.asarray(jnp.array([3.46875], dtype=jnp.bfloat16))
        second = np.asarray(jnp.array([3.484375], dtype=jnp.bfloat16))

        aggregated = _aggregate_metric_history(
            [
                {M.loss: first, M.token_nll_loss: first},
                {M.loss: second, M.token_nll_loss: second},
            ]
        )

        loss_curve = aggregated[M.loss]
        self.assertEqual(loss_curve.dtype, np.float32)
        self.assertAlmostEqual(float(loss_curve[0]), 3.4765625)

        bf16_bucket = np.asarray(jnp.asarray(loss_curve, dtype=jnp.bfloat16), dtype=np.float32)
        self.assertNotAlmostEqual(float(bf16_bucket[0]), float(loss_curve[0]))

    def test_eval_row_uses_eval_loss_ce(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = _load_script_module(
            "eval_matrix_jax_34",
            repo_root / "scripts" / "34_eval_matrix_jax.py",
        )

        row = script._build_eval_row(
            summary={
                "eval_loss": 1.25,
                "eval_loss_ce": 2.5,
                "tokens_per_second": 100.0,
                "elapsed_seconds": 1.0,
                "eval_batches": 8,
                "eval_tokens": 1024,
                "step": 17,
            },
            run_ref={"stage_id": "S3", "run_id": "run", "exp_name": "run"},
            dataset="books3",
            context=32768,
            paper_run_id="protocol_r_125m_main_v1",
            nll_path=Path("/tmp/per_position_nll.npy"),
        )

        self.assertEqual(row["loss"], 1.25)
        self.assertEqual(row["loss_ce"], 2.5)

    def test_observed_tokens_prefers_run_finished_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metrics_path = root / "metrics.jsonl"
            events_path = root / "events.jsonl"

            metrics_path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 0, "tokens_seen": 128}),
                        json.dumps({"step": 1, "tokens_seen": 256}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            events_path.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "run_finished", "tokens_seen": 384}),
                        json.dumps({"event": "run_finished", "status": "succeeded"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            observed = observed_tokens_from_runtime_artifacts(
                metrics_path=metrics_path,
                events_path=events_path,
                fallback=0,
            )

            self.assertEqual(observed, 384)

    def test_backfill_repairs_successful_run_metadata(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = _load_script_module(
            "backfill_run_metadata_59",
            repo_root / "scripts" / "59_backfill_run_metadata.py",
        )

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "experiments" / "protocol_r_125m_main_v1" / "S3_125M" / "ext-125m-e2e-32K"
            run_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "run_manifest.json").write_text(
                json.dumps({"stage_id": "S3_125M", "run_id": "ext-125m-e2e-32K"}, indent=2) + "\n",
                encoding="utf-8",
            )
            (run_dir / "run_result.json").write_text(
                json.dumps(
                    {
                        "stage_id": "S3_125M",
                        "run_id": "ext-125m-e2e-32K",
                        "status": "succeeded",
                        "run_dir": "/remote/run",
                        "metrics_path": "/remote/run/metrics.jsonl",
                        "events_path": "/remote/run/events.jsonl",
                        "tokens_seen": 0,
                        "wall_seconds": 0.0,
                        "gpu_hours": 0.0,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "budget_manifest.json").write_text(
                json.dumps(
                    {
                        "tokens_observed": 0,
                        "usage": {
                            "tokens_observed": 0,
                            "gpu_hours_observed": 0.0,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"step": 0, "tokens_seen": 512}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "run_started", "device_info": {"device_count": 8}}),
                        json.dumps({"event": "run_finished", "tokens_seen": 1024, "elapsed_seconds": 12.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            row = script.repair_run_dir(run_dir=run_dir, dry_run=False)
            self.assertEqual(row["status"], "updated")

            run_result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
            budget_manifest = json.loads((run_dir / "budget_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(run_result["tokens_seen"], 1024)
            self.assertEqual(run_result["stage_id"], "S3_125M")
            self.assertEqual(run_result["run_id"], "ext-125m-e2e-32K")
            self.assertEqual(budget_manifest["tokens_observed"], 1024)
            self.assertEqual(budget_manifest["usage"]["tokens_observed"], 1024)
            self.assertAlmostEqual(run_result["wall_seconds"], 12.0)
            self.assertAlmostEqual(run_result["gpu_hours"], 12.0 * 8 / 3600.0)
            self.assertAlmostEqual(
                budget_manifest["usage"]["gpu_hours_observed"],
                12.0 * 8 / 3600.0,
            )
            self.assertEqual(run_result["run_dir"], str(run_dir.resolve()))
            self.assertEqual(run_result["metrics_path"], str((run_dir / "metrics.jsonl").resolve()))
            self.assertEqual(run_result["events_path"], str((run_dir / "events.jsonl").resolve()))


if __name__ == "__main__":
    unittest.main()
