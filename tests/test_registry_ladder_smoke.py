"""End-to-end smoke test of the registry ladder on toy data.

Runs the same scripts as the README quick start (04 then 23) in a temporary
directory with tiny shapes. The token_stats subset covers a scratch pretrain,
the bridge path (adapt -> extension), the direct S2-minus extension and the
S2-minus load_part=all continuation, whose registry override names a
seed-suffixed parent run. The simulate subset checks that the simulator
writes the checkpoints a child stage needs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_DATA_SCRIPT = REPO_ROOT / "scripts" / "04_make_token_data.py"
LADDER_SCRIPT = REPO_ROOT / "scripts" / "23_warmstart_registry.py"
REPO_OUTPUT_DIRS = ("experiments", "checkpoints", "outputs", "reports")

TOKEN_STATS_STAGES = (
    "S0_PRETRAIN_FA_125M",
    "S2_ADAPT_125M",
    "S2_125M",
    "S2_MINUS_125M",
    "S2_MINUS_CONT_125M",
)
SIMULATE_STAGES = ("S0_PRETRAIN_FA_125M", "S0_125M")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]]
    )
    env.pop("WANDB_API_KEY", None)
    return env


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=_env(),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RegistryLadderSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._repo_dirs_before = {name: (REPO_ROOT / name).exists() for name in REPO_OUTPUT_DIRS}
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "tokens"
        gen = _run(
            [str(TOKEN_DATA_SCRIPT), "--out", str(self.data), "--train-tokens", "4000", "--val-tokens", "1000"],
            cwd=self.root,
        )
        self.assertEqual(gen.returncode, 0, gen.stdout + gen.stderr)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        repo_dirs_after = {name: (REPO_ROOT / name).exists() for name in REPO_OUTPUT_DIRS}
        self.assertEqual(self._repo_dirs_before, repo_dirs_after, "smoke run wrote into the repo tree")

    def _ladder(self, *, runtime_mode: str, name: str, stages: tuple[str, ...]) -> dict:
        # Relative exp folder and run id, no escape-hatch flags (fingerprints
        # come from the toy generator). cwd is the temp dir, so relative
        # defaults could not land in the repo either.
        result = _run(
            [
                str(LADDER_SCRIPT),
                "--paper-run-id",
                name,
                "--exp-folder",
                name,
                "--exp-dir",
                str(self.root / "experiments"),
                "--checkpoint-root",
                str(self.root / "checkpoints"),
                "--dclm-root",
                str(self.data),
                "--books-root",
                str(self.data),
                "--runtime-mode",
                runtime_mode,
                "--stage-ids",
                ",".join(stages),
                "--seq-length",
                "64",
                "--global-batch-size",
                "2",
                "--pretrain-steps",
                "4",
                "--adapt-steps",
                "2",
                "--ext-steps",
                "2",
                "--wandb-project",
                "none",
            ],
            cwd=self.root,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])
        summary = json.loads((self.root / "experiments" / name / "registry_run_summary.json").read_text())
        statuses = {row["stage_id"]: row["status"] for row in summary["rows"]}
        self.assertEqual(statuses, {stage: "succeeded" for stage in stages})
        return summary

    def _run_dir(self, name: str, stage_id: str, run_id: str) -> Path:
        return self.root / "experiments" / name / stage_id / run_id

    def test_token_stats_ladder_restores_declared_parents(self) -> None:
        name = "smoke_token_stats"
        self._ladder(runtime_mode="token_stats", name=name, stages=TOKEN_STATS_STAGES)

        ckpt_dir = self.root / "checkpoints" / name
        parent_run = "ext-125m-e2e-32K-from-fa-direct"
        parent_latest = json.loads((ckpt_dir / parent_run / "latest.json").read_text())
        parent_step = int(parent_latest["step"])

        cont_run = "ext-125m-e2e-32K-from-fa-direct-cont1440-seed001"
        cont_dir = self._run_dir(name, "S2_MINUS_CONT_125M", cont_run)
        metrics = _read_jsonl(cont_dir / "metrics.jsonl")
        # load_part=all continues the parent's step counter for ext_steps more steps.
        self.assertEqual([row["step"] for row in metrics], [parent_step + 1, parent_step + 2])
        restore = metrics[0]["restore"]
        self.assertEqual(restore["resume_exp_name"], parent_run)
        self.assertEqual(restore["restore_step"], parent_step)

        # The lineage manifest names the checkpoint that was actually restored.
        manifest = json.loads((cont_dir / "checkpoint_manifest.json").read_text())
        (parent_ref,) = manifest["parent_checkpoints"]
        self.assertEqual(parent_ref["checkpoint_id"], "S2_MINUS_125M")
        self.assertEqual(parent_ref["exp_name"], parent_run)
        self.assertEqual(Path(parent_ref["checkpoint_path"]).name, parent_latest["path"])

        bridge_metrics = _read_jsonl(
            self._run_dir(name, "S2_125M", "ext-125m-e2e-32K-from-fa-bridge") / "metrics.jsonl"
        )
        self.assertEqual(bridge_metrics[0]["restore"]["resume_exp_name"], "adapt-125m-e2e-8K-from-fa")

        # Hydra's per-run output lives in the run directory.
        self.assertTrue((cont_dir / "hydra" / ".hydra" / "config.yaml").exists())

        # The orchestrator's lineage record survives the runtime start-up and
        # carries the dataset identity from the toy generator's fingerprints.
        run_manifest = json.loads((cont_dir / "run_manifest.json").read_text())
        self.assertEqual(run_manifest["checkpoint_parents"][0]["exp_name"], parent_run)
        dataset_ids = {ref["dataset_id"] for ref in run_manifest["datasets"]}
        self.assertEqual(dataset_ids, {"toy_uniform_tokens"})
        self.assertEqual(run_manifest["runtime"]["runtime_mode"], "token_stats")

    def test_simulate_ladder_writes_parent_checkpoints(self) -> None:
        name = "smoke_simulate"
        self._ladder(runtime_mode="simulate", name=name, stages=SIMULATE_STAGES)
        ext_metrics = _read_jsonl(self._run_dir(name, "S0_125M", "ext-125m-fa-32K") / "metrics.jsonl")
        self.assertEqual(ext_metrics[0]["runtime_mode"], "simulate")
        self.assertEqual(ext_metrics[0]["restore"]["resume_exp_name"], "pretrain-125m-fa")
        self.assertEqual(ext_metrics[0]["restore"]["restore_status"], "ok")


if __name__ == "__main__":
    unittest.main()
