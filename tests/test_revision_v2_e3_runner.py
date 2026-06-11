from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_revision_v2_e3_runner_dry_run_emits_valid_plans() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        exp_dir = tmp / "experiments"
        checkpoint_root = tmp / "checkpoints"
        summary = tmp / "dry_run_summary.json"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "82_run_revision_v2_e3_frontier.py"),
            "--dry-run",
            "--dclm-root",
            str(tmp / "dclm"),
            "--books-root",
            str(tmp / "books"),
            "--exp-dir",
            str(exp_dir),
            "--checkpoint-root",
            str(checkpoint_root),
            "--bridge-accum-steps",
            "32",
            "--summary-out",
            str(summary),
        ]
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

        payload = json.loads(summary.read_text(encoding="utf-8"))
        assert payload["e3_paper_run_id"] == "revision_v2_e3_frontier_v1"
        assert payload["cont_paper_run_id"] == "revision_v2_s2minus_cont_v1"
        assert payload["deploy"] == "revision_v2_prime_h100_2x"
        assert payload["bridge_accum_steps"] == 32
        assert payload["ext_accum_steps"] == 0
        assert payload["cont_accum_steps"] == 0
        assert all(check["status"] == "PASS" for check in payload["validity_checks"])

        rows = {(row["paper_run_id"], row["stage_id"]): row for row in payload["rows"]}
        assert rows[("revision_v2_e3_frontier_v1", "S2_BRIDGE_40PCT_ADAPT_125M")]["target_total_steps"] == 1920
        assert rows[("revision_v2_e3_frontier_v1", "S2_BRIDGE_40PCT_125M")]["target_total_steps"] == 480
        assert rows[("revision_v2_e3_frontier_v1", "S2_BRIDGE_20PCT_ADAPT_125M")]["target_total_steps"] == 960
        assert rows[("revision_v2_e3_frontier_v1", "S2_BRIDGE_5PCT_ADAPT_125M")]["target_total_steps"] == 240
        assert rows[("revision_v2_s2minus_cont_v1", "S2_MINUS_CONT_125M")]["target_total_steps"] == 1920

        adapt_command = (
            exp_dir
            / "revision_v2_e3_frontier_v1"
            / "S2_BRIDGE_5PCT_ADAPT_125M"
            / "adapt-125m-e2e-8K-from-fa-bridge5pct-seed001"
            / "command.sh"
        ).read_text(encoding="utf-8")
        assert "+deploy=revision_v2_prime_h100_2x" in adapt_command
        assert "training.seq_length=8192" in adapt_command
        assert "training.global_batch_size=64" in adapt_command
        assert "training.accum_steps=32" in adapt_command

        ext_command = (
            exp_dir
            / "revision_v2_e3_frontier_v1"
            / "S2_BRIDGE_5PCT_125M"
            / "ext-125m-e2e-32K-from-fa-bridge5pct-seed001"
            / "command.sh"
        ).read_text(encoding="utf-8")
        assert "training.accum_steps=32" not in ext_command

        cont_command = (
            exp_dir
            / "revision_v2_s2minus_cont_v1"
            / "S2_MINUS_CONT_125M"
            / "ext-125m-e2e-32K-from-fa-direct-cont1440-seed001"
            / "command.sh"
        ).read_text(encoding="utf-8")
        assert "training.total_steps=1920" in cont_command
        assert "training.load_part=all" in cont_command
        assert "training.paper_run_id=revision_v2_s2minus_cont_v1" in cont_command
        assert "training.accum_steps=32" not in cont_command
