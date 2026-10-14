#!/usr/bin/env python3
"""Dry-run or execute the first-class Tier-1 inference-stage registry.

Every selected stage passes a fail-closed preflight first. A stage whose
condition is `target: pending` in configs/research/tier1_eval.yaml fails it,
so the full plan cannot run until a dated Tier-1 addendum names revision-v3
checkpoints. Restore stages copy checkpoints with scripts/46_restore_stage.py
from --restore-source-dir (a local stage export) or --restore-repo-id (a
Hugging Face repo you control).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ttt.research.tier1 import (
    condition_spec,
    is_pending,
    load_tier1_config,
    pending_message,
    sha256_file,
    sha256_payload,
    write_json,
)
from ttt.research.tier1_registry import (
    Tier1Stage,
    dependency_closure,
    load_tier1_registry,
    registry_plan_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/research/tier1_registry.yaml")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument("--stage-ids", default="")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--books-root", type=Path, default=None)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/tier1"))
    parser.add_argument(
        "--report-root", type=Path, default=Path("./reports/revision_v2/tier1")
    )
    restore_source = parser.add_mutually_exclusive_group()
    restore_source.add_argument(
        "--restore-source-dir",
        type=Path,
        default=None,
        help="Local stage-export directory for restore stages (46_restore_stage.py --source-dir).",
    )
    restore_source.add_argument(
        "--restore-repo-id",
        default=None,
        help="Hugging Face repo holding stage exports for restore stages (46_restore_stage.py --repo-id).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Passed to 92_make_tier1_reports.py (default there: <report>/aggregate/figures).",
    )
    parser.add_argument(
        "--started-at", type=dt.date.fromisoformat, default=dt.date.today()
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--preflight-out",
        type=Path,
        default=Path(
            "./reports/revision_v2/tier1/preflight/tier1_registry_preflight.json"
        ),
    )
    return parser.parse_args(argv)


def _script(name: str) -> str:
    return str(REPO_ROOT / "scripts" / name)


def _command(
    stage: Tier1Stage, args: argparse.Namespace, config: dict[str, Any]
) -> list[str]:
    python = sys.executable
    artifact_root = args.artifact_root.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()
    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    common_config = ["--config", str(config_path)]
    action = stage.action
    if action == "prereg_gate":
        return []
    if action == "prepare_t1a":
        return [
            python,
            _script("85_prepare_t1a_zero_shot.py"),
            *common_config,
            "--out",
            str(artifact_root / "t1a_manifest.json"),
        ]
    if action == "prepare_t1b":
        if args.books_root is None:
            raise ValueError("T1b stages require --books-root")
        return [
            python,
            _script("87_run_t1b_fast_weight_probe.py"),
            "prepare-manifest",
            *common_config,
            "--books-root",
            str(args.books_root.expanduser().resolve()),
            "--out",
            str(artifact_root / "t1b_manifest.json"),
        ]
    if action in {"prepare_t1c_smoke", "prepare_t1c"}:
        out_dir = artifact_root / (
            "t1c_smoke" if action == "prepare_t1c_smoke" else "t1c_ruler"
        )
        command = [
            python,
            _script("88_prepare_t1c_ruler.py"),
            *common_config,
            "--out-dir",
            str(out_dir),
        ]
        if action == "prepare_t1c_smoke":
            command.extend(["--examples-per-task", "2"])
        return command
    if action == "validate_t1c":
        return [
            python,
            _script("89_validate_t1c_ruler_harness.py"),
            *common_config,
            "--smoke-manifest",
            str(artifact_root / "t1c_smoke" / "manifest.json"),
            "--started-at",
            args.started_at.isoformat(),
            "--out",
            str(artifact_root / "t1c_harness_validation.json"),
        ]
    if action == "restore":
        spec = condition_spec(config, stage.condition)
        if args.restore_source_dir is not None:
            source = ["--source-dir", str(args.restore_source_dir.expanduser().resolve())]
        elif args.restore_repo_id:
            source = ["--repo-id", str(args.restore_repo_id)]
        else:
            raise ValueError(
                "Restore stages need --restore-source-dir (local stage export) or "
                "--restore-repo-id (your Hugging Face repo)"
            )
        command = [
            python,
            _script("46_restore_stage.py"),
            *source,
            "--source-paper-run-id",
            spec.source_paper_run_id,
            "--source-stage-id",
            spec.source_stage_id,
            "--source-run-id",
            spec.source_run_id,
            "--target-paper-run-id",
            str(config["paper_run_id"]),
            "--target-stage-id",
            stage.condition,
            "--target-run-id",
            spec.source_run_id,
            "--exp-dir",
            str(exp_dir),
            "--checkpoint-root",
            str(checkpoint_root),
            "--params-only",
        ]
        if stage.all_model_checkpoints:
            command.append("--all-model-checkpoints")
        return command
    if action == "eval_t1a":
        return [
            python,
            _script("86_run_t1a_zero_shot_jax.py"),
            *common_config,
            "--condition",
            stage.condition,
            "--manifest",
            str(artifact_root / "t1a_manifest.json"),
            "--exp-dir",
            str(exp_dir),
            "--checkpoint-root",
            str(checkpoint_root),
            "--batch-size",
            str(max(8, args.batch_size)),
            "--out-root",
            str(report_root),
        ]
    if action == "eval_t1b":
        if args.books_root is None:
            raise ValueError("T1b stages require --books-root")
        return [
            python,
            _script("87_run_t1b_fast_weight_probe.py"),
            "run",
            *common_config,
            "--condition",
            stage.condition,
            "--manifest",
            str(artifact_root / "t1b_manifest.json"),
            "--books-root",
            str(args.books_root.expanduser().resolve()),
            "--exp-dir",
            str(exp_dir),
            "--checkpoint-root",
            str(checkpoint_root),
            "--batch-size",
            str(args.batch_size),
            "--checkpoint-steps",
            "all",
            "--out-root",
            str(report_root),
        ]
    if action in {"eval_t1c", "eval_t1c_760_smoke"}:
        is_smoke = action == "eval_t1c_760_smoke"
        command = [
            python,
            _script("90_run_t1c_ruler_jax.py"),
            *common_config,
            "--condition",
            stage.condition,
            "--manifest",
            str(
                artifact_root
                / ("t1c_smoke" if is_smoke else "t1c_ruler")
                / "manifest.json"
            ),
            "--validation-report",
            str(artifact_root / "t1c_harness_validation.json"),
            "--exp-dir",
            str(exp_dir),
            "--checkpoint-root",
            str(checkpoint_root),
            "--batch-size",
            str(args.batch_size),
            "--out-root",
            str(artifact_root if is_smoke else report_root),
        ]
        if is_smoke:
            command.append("--smoke")
        return command
    if action == "report":
        tier1_root = report_root / str(config["paper_run_id"])
        command = [
            python,
            _script("92_make_tier1_reports.py"),
            *common_config,
            "--tier1-root",
            str(tier1_root),
        ]
        if args.figures_dir is not None:
            command.extend(["--figures-dir", str(args.figures_dir.expanduser().resolve())])
        if args.include_optional:
            command.extend(["--include-optional-t1b", "--include-760m-t1c"])
        return command
    raise ValueError(f"Unsupported Tier-1 action {action}")


def _git_clean_prereg(config: dict[str, Any]) -> tuple[bool, str]:
    paths = [
        Path(config["preregistration"]),
        Path(config["machine_plan"]),
        Path(config["_config_path"]).relative_to(REPO_ROOT),
        Path("configs/research/tier1_registry.yaml"),
    ]
    for path in paths:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            return False, f"preregistration input is not committed/tracked: {path}"
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *map(str, paths)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (not dirty, dirty or "all preregistration inputs are committed and clean")


def _git_worktree_state() -> tuple[bool, str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    detail = f"head={head}" if not dirty else dirty
    return not dirty, detail, head


def _preflight(
    stages: list[Tier1Stage], args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(stage_id: str, check: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "stage_id": stage_id,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    clean, detail = _git_clean_prereg(config)
    record("T1_PREREG_GATE", "preregistration_committed_before_results", clean, detail)
    worktree_clean, worktree_detail, git_head = _git_worktree_state()
    record(
        "T1_PREREG_GATE",
        "production_git_worktree_clean",
        worktree_clean,
        worktree_detail,
    )
    ids_seen: set[str] = set()
    for stage in stages:
        record(
            stage.stage_id,
            "dependency_order",
            set(stage.depends_on) <= ids_seen,
            str(stage.depends_on),
        )
        ids_seen.add(stage.stage_id)
        if stage.condition and stage.condition in config["conditions"]:
            pending = is_pending(config, stage.condition)
            record(
                stage.stage_id,
                "condition_target_available",
                not pending,
                pending_message(config, [stage.condition]) if pending else stage.condition,
            )
        try:
            command = _command(stage, args, config)
            record(
                stage.stage_id,
                "command_materialized",
                True,
                " ".join(command) or "internal gate",
            )
        except Exception as exc:
            record(
                stage.stage_id,
                "command_materialized",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        if stage.condition:
            try:
                condition_spec(config, stage.condition)
                record(stage.stage_id, "condition_registered", True, stage.condition)
            except Exception as exc:
                record(stage.stage_id, "condition_registered", False, str(exc))
        if stage.action in {"prepare_t1b", "eval_t1b"}:
            root_ok = (
                args.books_root is not None
                and args.books_root.expanduser().resolve().is_dir()
            )
            record(
                stage.stage_id, "books_root_available", root_ok, str(args.books_root)
            )
    status = (
        "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL"
    )
    return {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "status": status,
        "checks_executed_count": len(checks),
        "checks_skipped_count": 0,
        "tier1_config_sha256": config["_config_sha256"],
        "registry_sha256": sha256_file(args.registry.expanduser().resolve()),
        "git_head": git_head,
        "plan": registry_plan_payload(stages),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_tier1_config(args.config)
    all_stages = load_tier1_registry(args.registry)
    requested = {part.strip() for part in args.stage_ids.split(",") if part.strip()}
    if not requested:
        requested = {
            stage.stage_id
            for stage in all_stages
            if stage.required or args.include_optional
        }
    selected = dependency_closure(all_stages, requested)
    preflight = _preflight(selected, args, config)
    preflight["preflight_sha256"] = sha256_payload(preflight)
    write_json(args.preflight_out, preflight)
    print(
        f"Tier-1 preflight {preflight['status']}: {preflight['checks_executed_count']} checks; "
        f"report={args.preflight_out.resolve()}"
    )
    if preflight["status"] != "PASS":
        pending = sorted(
            {
                stage.condition
                for stage in selected
                if stage.condition
                and stage.condition in config["conditions"]
                and is_pending(config, stage.condition)
            }
        )
        if pending:
            print(f"Refusing to run: {pending_message(config, pending)}")
        for row in preflight["checks"]:
            if row["status"] == "FAIL" and row["check"] != "condition_target_available":
                print(f"FAIL {row['stage_id']} {row['check']}: {row['detail']}")
        return 1
    if not args.execute:
        return 0

    marker_root = args.artifact_root.expanduser().resolve() / "stage_markers"
    marker_root.mkdir(parents=True, exist_ok=True)
    for stage in selected:
        marker = marker_root / f"{stage.stage_id}.json"
        if args.resume and marker.exists():
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("status") != "succeeded":
                raise RuntimeError(f"Cannot resume from failed marker: {marker}")
            continue
        for dependency in stage.depends_on:
            dependency_marker = marker_root / f"{dependency}.json"
            if not dependency_marker.exists():
                raise RuntimeError(
                    f"Stage {stage.stage_id} missing dependency marker {dependency}"
                )
            dependency_payload = json.loads(
                dependency_marker.read_text(encoding="utf-8")
            )
            if dependency_payload.get("status") != "succeeded":
                raise RuntimeError(
                    f"Stage {stage.stage_id} dependency failed: {dependency}"
                )
        started = time.time()
        command = _command(stage, args, config)
        if stage.action == "prereg_gate":
            return_code = 0
        else:
            return_code = subprocess.run(command, cwd=REPO_ROOT).returncode
        payload = {
            "schema_version": "1.0",
            "paper_run_id": config["paper_run_id"],
            "stage_id": stage.stage_id,
            "action": stage.action,
            "condition": stage.condition,
            "git_head": preflight["git_head"],
            "dependencies": list(stage.depends_on),
            "command": command,
            "elapsed_seconds": float(time.time() - started),
            "return_code": return_code,
            "status": "succeeded" if return_code == 0 else "failed",
        }
        payload["marker_sha256"] = sha256_payload(payload)
        write_json(marker, payload)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
