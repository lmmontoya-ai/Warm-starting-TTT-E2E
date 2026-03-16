#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ttt.research.author_checkpoints import load_env_file
from ttt.research.continuation_ablations import (
    ABLATION_PAPER_RUN_ID,
    CANONICAL_PAPER_RUN_ID,
    ISO_QUALITY_HARD_CAP_EXTRA_STEPS,
    ISO_QUALITY_PLATEAU_INTERVALS,
    ISO_QUALITY_PLATEAU_MIN_IMPROVEMENT,
    ISO_QUALITY_TARGET_LOSS,
    ISO_TOTAL_EXTRA_TOKENS,
    S2_ISOQ_SOURCE,
    S2_ISOQ_STAGE_ID,
    S3_ISOTOK_SOURCE,
    S3_ISOTOK_STAGE_ID,
    BOOKS_SEQ_LENGTH,
    PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE,
    STEP_INTERVAL,
    canonical_branch_costs,
    canonical_eval_loss,
    cumulative_wall_by_checkpoint,
    device_count_from_events,
    expected_canonical_checkpoint_step,
    extra_steps_from_checkpoint_step,
    extra_tokens_for_steps,
    final_step_to_total_steps,
    iso_total_extra_steps,
    latest_checkpoint_step,
    should_stop_for_plateau,
    total_steps_to_final_step,
)
from ttt.research.lineage import resolve_checkpoint_ref
from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec, CheckpointRef, EvalSpec, StageSpec, utc_now_iso


DEFAULT_REGISTRY = Path("./configs/research/warmstart_registry.yaml")


def _resolve_uv_executable() -> str:
    candidates = [shutil.which("uv")]
    home = Path.home()
    candidates.extend(
        [
            str(home / ".local" / "bin" / "uv"),
            "/root/.local/bin/uv",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return str(path)
    raise FileNotFoundError("Could not locate `uv`.")


UV_EXECUTABLE = _resolve_uv_executable()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _redact_cmd(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    for part in cmd:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        redacted.append(part)
        if part == "--token":
            skip_next = True
    return redacted


def _run(cmd: list[str], *, cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> int:
    print("$ " + shlex.join(_redact_cmd(cmd)), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False, cwd=cwd, env=env).returncode


def _parse_mode(raw: str) -> list[str]:
    if raw == "all":
        return ["iso_quality", "iso_total_tokens"]
    return [raw]


def _canonical_stage_refs() -> list[tuple[str, str]]:
    return [
        ("S0_PRETRAIN_FA_125M", "pretrain-125m-fa"),
        ("S2_ADAPT_125M", "adapt-125m-e2e-8K-from-fa"),
        S2_ISOQ_SOURCE,
        ("S3_PRETRAIN_E2E_125M", "pretrain-125m-e2e"),
        S3_ISOTOK_SOURCE,
    ]


def _restore_stage_if_missing(*, repo_root: Path, args: argparse.Namespace, stage_id: str, run_id: str) -> None:
    exp_target = args.exp_dir / CANONICAL_PAPER_RUN_ID / stage_id / run_id
    checkpoint_target = args.checkpoint_root / CANONICAL_PAPER_RUN_ID / run_id / "latest.json"
    if exp_target.exists() and checkpoint_target.exists():
        return
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/46_restore_stage_from_hf.py",
        "--repo-id",
        args.repo_id,
        "--token",
        args.token,
        "--source-paper-run-id",
        CANONICAL_PAPER_RUN_ID,
        "--source-stage-id",
        stage_id,
        "--source-run-id",
        run_id,
        "--target-paper-run-id",
        CANONICAL_PAPER_RUN_ID,
        "--target-stage-id",
        stage_id,
        "--target-run-id",
        run_id,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
    ]
    if args.dry_run:
        _run(cmd, cwd=repo_root, dry_run=True)
        return
    rc = _run(cmd, cwd=repo_root, dry_run=False)
    if rc != 0:
        raise RuntimeError(f"Failed to restore canonical stage {stage_id}/{run_id} from HF (rc={rc})")


def _checkpoint_root_for(paper_run_id: str, checkpoint_root: Path, run_id: str) -> Path:
    return checkpoint_root / paper_run_id / run_id


def _latest_checkpoint_step_if_present(checkpoint_dir: Path, *, dry_run: bool) -> int | None:
    if dry_run or not checkpoint_dir.exists():
        return None
    latest_json = checkpoint_dir / "latest.json"
    if not latest_json.exists():
        return None
    return latest_checkpoint_step(checkpoint_dir)


def _run_dir_for(exp_dir: Path, paper_run_id: str, stage_id: str, run_id: str) -> Path:
    return exp_dir / paper_run_id / stage_id / run_id


def _snapshot_eval_outputs(*, run_dir: Path, snapshot_dir: Path, checkpoint_step: int) -> dict[str, Any]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_src = run_dir / "eval_manifest.json"
    raw_json_src = run_dir / "eval_parity_raw.json"
    raw_csv_src = run_dir / "eval_parity_raw.csv"
    eval_snap = snapshot_dir / "eval_manifest_snapshot.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, eval_snap)
    if raw_json_src.exists():
        shutil.copy2(raw_json_src, snapshot_dir / "eval_parity_raw.json")
    if raw_csv_src.exists():
        shutil.copy2(raw_csv_src, snapshot_dir / "eval_parity_raw.csv")

    jax_eval_root = run_dir / "jax_eval" / "books3" / "ctx_32768"
    if (jax_eval_root / "per_position_nll.npy").exists():
        shutil.copy2(jax_eval_root / "per_position_nll.npy", snapshot_dir / "per_position_nll.npy")
    if (jax_eval_root / "loss_curve.npy").exists():
        shutil.copy2(jax_eval_root / "loss_curve.npy", snapshot_dir / "loss_curve.npy")
    if (jax_eval_root / "token_nll_curve.npy").exists():
        shutil.copy2(jax_eval_root / "token_nll_curve.npy", snapshot_dir / "token_nll_curve.npy")

    summary = _load_json(snapshot_dir / "summary.json")
    rows = summary.get("rows", [])
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"Expected exactly one eval row in {snapshot_dir / 'summary.json'}")
    row = rows[0]
    if not isinstance(row, dict):
        raise ValueError(f"Invalid eval row in {snapshot_dir / 'summary.json'}")
    row = dict(row)
    row["checkpoint_step"] = int(row.get("checkpoint_step", checkpoint_step))
    return row


def _eval_checkpoint_step(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    stage_id: str,
    run_id: str,
    checkpoint_step: int,
    mode: str,
) -> dict[str, Any]:
    run_dir = _run_dir_for(args.exp_dir, args.paper_run_id, stage_id, run_id)
    snapshot_dir = (
        args.reports_root
        / "eval_snapshots"
        / mode
        / run_id
        / f"step_{checkpoint_step}"
    )
    summary_json = snapshot_dir / "summary.json"
    summary_csv = snapshot_dir / "summary.csv"
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/34_eval_matrix_jax.py",
        "--paper-run-id",
        args.paper_run_id,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--exp-folder",
        args.paper_run_id,
        "--stages",
        stage_id,
        "--runs",
        run_id,
        "--datasets",
        "books3",
        "--contexts",
        str(BOOKS_SEQ_LENGTH),
        "--dclm-root",
        str(args.dclm_root),
        "--books-root",
        str(args.books_root),
        "--eval-split",
        "val",
        "--eval-batches",
        str(args.eval_batches),
        "--checkpoint-step",
        str(checkpoint_step),
        "--summary-json",
        str(summary_json),
        "--summary-csv",
        str(summary_csv),
        "--strict",
    ]
    env = os.environ.copy()
    env["TTT_ATTENTION_IMPLEMENTATION"] = "xla"
    rc = _run(cmd, cwd=repo_root, dry_run=args.dry_run, env=env)
    if rc != 0:
        raise RuntimeError(f"Eval failed for {stage_id}/{run_id} step={checkpoint_step} (rc={rc})")
    if args.dry_run:
        return {
            "stage_id": stage_id,
            "run_id": run_id,
            "checkpoint_step": checkpoint_step,
            "loss_ce": None,
            "loss": None,
            "tokens_per_second": None,
            "eval_wall_seconds": None,
            "snapshot_dir": str(snapshot_dir),
            "status": "dry_run",
        }
    row = _snapshot_eval_outputs(run_dir=run_dir, snapshot_dir=snapshot_dir, checkpoint_step=checkpoint_step)
    row["snapshot_dir"] = str(snapshot_dir)
    return row


def _ablation_stage(template: StageSpec, *, stage_id: str, run_id: str, notes_suffix: str) -> StageSpec:
    payload = template.to_dict()
    payload["stage_id"] = stage_id
    payload["name"] = f"{template.name} continuation ablation"
    payload["exp_name"] = run_id
    payload["path_group"] = "ablation"
    payload["required_parent_checkpoint_ids"] = []
    payload["notes"] = f"{template.notes} {notes_suffix}".strip()
    return StageSpec.from_dict(payload)


def _base_opts(args: argparse.Namespace) -> OrchestratorOptions:
    return OrchestratorOptions(
        deploy=args.deploy,
        runtime_mode=args.runtime_mode,
        exp_dir=args.exp_dir,
        checkpoint_root=args.checkpoint_root,
        profile_root=args.profile_root,
        dclm_root=args.dclm_root,
        books_root=args.books_root,
        exp_folder=args.paper_run_id,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_key=args.wandb_key,
        ext_global_batch_size=PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE,
        seq_length=BOOKS_SEQ_LENGTH,
        save_milestone_freq=args.step_interval,
        dummy_dataset=args.dummy_dataset,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.allow_missing_fingerprints),
    )


def _budget_for_total_steps(total_steps: int) -> BudgetSpec:
    return BudgetSpec(
        budget_id="125m_continuation_ablation",
        pretrain_steps=0,
        adapt_steps=0,
        ext_steps=int(total_steps),
        seed=0,
    )


def _canonical_baseline_point(*, exp_dir: Path, stage_id: str, run_id: str, source_checkpoint_step: int, branch_costs: dict[str, float], branch_label: str) -> dict[str, Any]:
    try:
        eval_loss = canonical_eval_loss(exp_dir, stage_id=stage_id, run_id=run_id)
        run_result = _load_json(exp_dir / CANONICAL_PAPER_RUN_ID / stage_id / run_id / "run_result.json")
    except FileNotFoundError:
        eval_loss = None
        run_result = {"status": "dry_run"}
    return {
        "mode": branch_label,
        "stage_id": stage_id,
        "run_id": run_id,
        "checkpoint_step": int(source_checkpoint_step),
        "extra_steps": 0,
        "extra_tokens": 0,
        "extra_gpu_hours": 0.0,
        "total_branch_tokens": branch_costs["base_total_tokens"],
        "total_branch_gpu_hours": branch_costs["base_total_gpu_hours"],
        "total_branch_marginal_gpu_hours": branch_costs["base_marginal_gpu_hours"],
        "loss_ce_mean": eval_loss,
        "loss_mean": eval_loss,
        "tokens_per_second_mean": None,
        "eval_wall_seconds": None,
        "matched_target": eval_loss is not None and eval_loss <= ISO_QUALITY_TARGET_LOSS,
        "status": str(run_result.get("status", "")),
        "snapshot_dir": "",
    }


def _branch_costs_or_placeholder(exp_dir: Path, branch_key: str, *, dry_run: bool) -> dict[str, float]:
    try:
        return canonical_branch_costs(exp_dir)[branch_key]
    except FileNotFoundError:
        if not dry_run:
            raise
        return {
            "base_total_tokens": 0.0,
            "base_total_gpu_hours": 0.0,
            "base_marginal_gpu_hours": 0.0,
        }


def _frontier_point_from_eval(
    *,
    mode: str,
    run_dir: Path,
    row: dict[str, Any],
    source_checkpoint_step: int,
    base_costs: dict[str, float],
) -> dict[str, Any]:
    checkpoint_step = int(row["checkpoint_step"])
    extra_steps = extra_steps_from_checkpoint_step(
        source_checkpoint_step=source_checkpoint_step,
        checkpoint_step=checkpoint_step,
    )
    extra_tokens = extra_tokens_for_steps(extra_steps)
    metrics_path = run_dir / "metrics.jsonl"
    events_path = run_dir / "events.jsonl"
    cumulative = cumulative_wall_by_checkpoint(metrics_path)
    extra_wall_seconds = float(cumulative.get(checkpoint_step, 0.0))
    device_count = device_count_from_events(events_path)
    extra_gpu_hours = extra_wall_seconds * float(device_count) / 3600.0
    raw_loss_ce = row.get("loss_ce")
    raw_loss = row.get("loss")
    raw_tps = row.get("tokens_per_second")
    raw_eval_wall = row.get("eval_wall_seconds")
    loss_ce = float(raw_loss_ce) if raw_loss_ce is not None else None
    return {
        "mode": mode,
        "stage_id": str(row["stage_id"]),
        "run_id": str(row["run_id"]),
        "checkpoint_step": checkpoint_step,
        "extra_steps": extra_steps,
        "extra_tokens": extra_tokens,
        "extra_gpu_hours": extra_gpu_hours,
        "total_branch_tokens": float(base_costs["base_total_tokens"]) + float(extra_tokens),
        "total_branch_gpu_hours": float(base_costs["base_total_gpu_hours"]) + float(extra_gpu_hours),
        "total_branch_marginal_gpu_hours": float(base_costs["base_marginal_gpu_hours"]) + float(extra_gpu_hours),
        "loss_ce_mean": loss_ce,
        "loss_mean": float(raw_loss) if raw_loss is not None else None,
        "tokens_per_second_mean": float(raw_tps) if raw_tps is not None else None,
        "eval_wall_seconds": float(raw_eval_wall) if raw_eval_wall is not None else None,
        "matched_target": loss_ce is not None and loss_ce <= ISO_QUALITY_TARGET_LOSS,
        "status": str(row["status"]),
        "snapshot_dir": str(row["snapshot_dir"]),
    }


def _target_run_id(prefix: str, total_steps: int) -> str:
    return f"{prefix}-to{int(total_steps):04d}"


def _copytree_overwrite(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _truncated_metrics_rows(rows: list[dict[str, Any]], target_step: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        step = row.get("step")
        if step is None:
            continue
        try:
            parsed_step = int(step)
        except (TypeError, ValueError):
            continue
        if parsed_step <= target_step:
            kept.append(row)
    return kept


def _materialize_checkpoint_step_as_run(
    *,
    source_stage_id: str,
    source_run_id: str,
    source_run_dir: Path,
    source_checkpoint_dir: Path,
    target_stage_id: str,
    target_run_id: str,
    target_paper_run_id: str,
    exp_dir: Path,
    checkpoint_root: Path,
    target_step: int,
) -> tuple[Path, Path]:
    target_run_dir = _run_dir_for(exp_dir, target_paper_run_id, target_stage_id, target_run_id)
    target_checkpoint_dir = _checkpoint_root_for(target_paper_run_id, checkpoint_root, target_run_id)
    _copytree_overwrite(source_run_dir, target_run_dir)
    if target_checkpoint_dir.exists():
        shutil.rmtree(target_checkpoint_dir)
    target_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_step_dir = source_checkpoint_dir / str(target_step)
    if not source_step_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint step directory: {source_step_dir}")
    shutil.copytree(source_step_dir, target_checkpoint_dir / str(target_step))
    step_metadata_name = f"step_metadata_{target_step:08d}.json"
    source_step_metadata = source_checkpoint_dir / step_metadata_name
    if source_step_metadata.exists():
        shutil.copy2(source_step_metadata, target_checkpoint_dir / step_metadata_name)
    _write_json(target_checkpoint_dir / "latest.json", {"step": int(target_step), "path": str(target_step)})

    run_manifest_path = target_run_dir / "run_manifest.json"
    run_result_path = target_run_dir / "run_result.json"
    checkpoint_manifest_path = target_run_dir / "checkpoint_manifest.json"
    budget_manifest_path = target_run_dir / "budget_manifest.json"
    metrics_path = target_run_dir / "metrics.jsonl"
    events_path = target_run_dir / "events.jsonl"

    run_manifest = _load_json(run_manifest_path)
    run_result = _load_json(run_result_path)
    checkpoint_manifest = _load_json(checkpoint_manifest_path)
    budget_manifest = _load_json(budget_manifest_path) if budget_manifest_path.exists() else {}

    metrics_rows = _load_jsonl(source_run_dir / "metrics.jsonl")
    events_rows = _load_jsonl(source_run_dir / "events.jsonl")
    kept_metrics = _truncated_metrics_rows(metrics_rows, target_step)
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept_metrics),
        encoding="utf-8",
    )

    checkpoint_wall = cumulative_wall_by_checkpoint(source_run_dir / "metrics.jsonl")
    wall_seconds = float(checkpoint_wall.get(target_step, 0.0))
    device_count = device_count_from_events(source_run_dir / "events.jsonl")
    gpu_hours = wall_seconds * float(device_count) / 3600.0

    last_train_row = None
    for row in reversed(kept_metrics):
        if "train_step_seconds" in row:
            last_train_row = row
            break
    tokens_seen = int((last_train_row or {}).get("tokens_seen", 0) or 0)

    retained_events: list[dict[str, Any]] = []
    for row in events_rows:
        event = str(row.get("event", "")).strip()
        if event == "run_finished":
            continue
        retained_events.append(row)
    retained_events.append(
        {
            "event": "run_finished",
            "runtime_mode": "jax_train",
            "elapsed_seconds": wall_seconds,
            "total_steps": final_step_to_total_steps(target_step),
            "tokens_seen": tokens_seen,
        }
    )
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained_events),
        encoding="utf-8",
    )

    run_manifest["run_id"] = target_run_id
    run_manifest["exp_name"] = target_run_id
    run_manifest["paper_run_id"] = target_paper_run_id
    run_manifest["stage_id"] = target_stage_id
    tags = run_manifest.get("tags")
    if not isinstance(tags, dict):
        tags = {}
    tags.update(
        {
            "materialized_checkpoint_step": str(target_step),
            "materialized_from_stage_id": source_stage_id,
            "materialized_from_run_id": source_run_id,
        }
    )
    run_manifest["tags"] = tags
    _write_json(run_manifest_path, run_manifest)

    run_result["run_id"] = target_run_id
    run_result["stage_id"] = target_stage_id
    run_result["run_dir"] = str(target_run_dir.resolve())
    run_result["metrics_path"] = str(metrics_path.resolve())
    run_result["events_path"] = str(events_path.resolve())
    run_result["step"] = int(target_step)
    run_result["tokens_seen"] = tokens_seen
    run_result["wall_seconds"] = wall_seconds
    run_result["gpu_hours"] = gpu_hours
    checkpoint = run_result.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    checkpoint.update(
        {
            "checkpoint_id": target_stage_id,
            "exp_folder": target_paper_run_id,
            "exp_name": target_run_id,
            "step": int(target_step),
            "checkpoint_path": str((target_checkpoint_dir / str(target_step)).resolve()),
        }
    )
    run_result["checkpoint"] = checkpoint
    _write_json(run_result_path, run_result)

    run_checkpoint = checkpoint_manifest.get("run_checkpoint", {})
    if not isinstance(run_checkpoint, dict):
        run_checkpoint = {}
    run_checkpoint.update(
        {
            "checkpoint_id": target_stage_id,
            "exp_folder": target_paper_run_id,
            "exp_name": target_run_id,
            "step": int(target_step),
            "checkpoint_path": str((target_checkpoint_dir / str(target_step)).resolve()),
        }
    )
    checkpoint_manifest["run_checkpoint"] = run_checkpoint
    _write_json(checkpoint_manifest_path, checkpoint_manifest)

    usage = budget_manifest.get("usage")
    if not isinstance(usage, dict):
        usage = {}
        budget_manifest["usage"] = usage
    usage["tokens_observed"] = tokens_seen
    usage["gpu_hours_observed"] = gpu_hours
    budget_manifest["tokens_observed"] = tokens_seen
    _write_json(budget_manifest_path, budget_manifest)

    return target_run_dir, target_checkpoint_dir


def _export_terminal_run(*, repo_root: Path, args: argparse.Namespace, stage_id: str, run_id: str) -> None:
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/40_export_stage_to_hf.py",
        "--paper-run-id",
        args.paper_run_id,
        "--stage-id",
        stage_id,
        "--run-id",
        run_id,
        "--repo-id",
        args.repo_id,
        "--token",
        args.token,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--require-eval-success",
    ]
    rc = _run(cmd, cwd=repo_root, dry_run=args.dry_run)
    if rc != 0:
        raise RuntimeError(f"Failed to export terminal ablation stage {stage_id}/{run_id} (rc={rc})")


def _stage_template(registry_path: Path, stage_id: str) -> StageSpec:
    registry = load_registry(registry_path)
    return registry.stage(stage_id)


def _baseline_source_ref(*, checkpoint_root: Path, stage_id: str, run_id: str) -> CheckpointRef:
    ref = resolve_checkpoint_ref(
        checkpoint_root=checkpoint_root,
        exp_folder=CANONICAL_PAPER_RUN_ID,
        checkpoint_id=stage_id,
        exp_name=run_id,
        allow_missing=True,
    )
    if ref.step is not None:
        return ref
    expected_step = expected_canonical_checkpoint_step(stage_id, run_id)
    return CheckpointRef(
        checkpoint_id=stage_id,
        exp_folder=CANONICAL_PAPER_RUN_ID,
        exp_name=run_id,
        step=expected_step if expected_step > 0 else None,
        checkpoint_path=str((_checkpoint_root_for(CANONICAL_PAPER_RUN_ID, checkpoint_root, run_id)).resolve()),
    )


def _ablation_source_ref(*, checkpoint_root: Path, paper_run_id: str, stage_id: str, run_id: str) -> CheckpointRef:
    ref = resolve_checkpoint_ref(
        checkpoint_root=checkpoint_root,
        exp_folder=paper_run_id,
        checkpoint_id=stage_id,
        exp_name=run_id,
        allow_missing=True,
    )
    if ref.step is not None:
        return ref
    return CheckpointRef(
        checkpoint_id=stage_id,
        exp_folder=paper_run_id,
        exp_name=run_id,
        checkpoint_path=str((_checkpoint_root_for(paper_run_id, checkpoint_root, run_id)).resolve()),
    )


def _iso_quality(args: argparse.Namespace, repo_root: Path, ledger: dict[str, Any]) -> None:
    template = _stage_template(args.registry, S2_ISOQ_SOURCE[0])
    stage_id = S2_ISOQ_STAGE_ID
    source_stage_id, source_run_id = S2_ISOQ_SOURCE
    source_ref = _baseline_source_ref(
        checkpoint_root=args.checkpoint_root,
        stage_id=source_stage_id,
        run_id=source_run_id,
    )
    source_checkpoint_step = int(source_ref.step or 0)
    source_checkpoint_dir = _checkpoint_root_for(CANONICAL_PAPER_RUN_ID, args.checkpoint_root, source_run_id)
    branch_costs = _branch_costs_or_placeholder(args.exp_dir, "s2_branch", dry_run=args.dry_run)

    mode_payload: dict[str, Any] = {
        "mode": "iso_quality",
        "source": {
            "stage_id": source_stage_id,
            "run_id": source_run_id,
            "checkpoint_step": source_checkpoint_step,
            "checkpoint_path": source_ref.checkpoint_path,
            "target_loss_ce_mean": ISO_QUALITY_TARGET_LOSS,
        },
        "runs": [],
        "frontier_points": [
            _canonical_baseline_point(
                exp_dir=args.exp_dir,
                stage_id=source_stage_id,
                run_id=source_run_id,
                source_checkpoint_step=source_checkpoint_step,
                branch_costs=branch_costs,
                branch_label="iso_quality",
            )
        ],
        "terminal": {},
        "status": "running",
    }
    current_ref = source_ref
    current_checkpoint_dir = source_checkpoint_dir
    current_source_step = source_checkpoint_step

    max_total_steps = final_step_to_total_steps(source_checkpoint_step) + ISO_QUALITY_HARD_CAP_EXTRA_STEPS

    while final_step_to_total_steps(current_source_step) < max_total_steps:
        remaining = max_total_steps - final_step_to_total_steps(current_source_step)
        block_extra_steps = min(360, remaining)
        target_total_steps = final_step_to_total_steps(current_source_step) + block_extra_steps
        run_id = _target_run_id("s2-125m-isoq", target_total_steps)
        stage = _ablation_stage(
            template,
            stage_id=stage_id,
            run_id=run_id,
            notes_suffix="Continuation ablation from canonical S2_125M with load_part=all.",
        )
        opts = _base_opts(args)
        budget = _budget_for_total_steps(target_total_steps)
        eval_spec = EvalSpec(eval_id="ablation_books32k", contexts=[BOOKS_SEQ_LENGTH], datasets=["books3"], eval_split="val", eval_batches=args.eval_batches)
        result = run_stage(
            stage=stage,
            stage_map={},
            opts=opts,
            budget=budget,
            eval_spec=eval_spec,
            repo_root=repo_root,
            run_id=run_id,
            explicit_resume_checkpoint_path=current_checkpoint_dir,
            explicit_resume_checkpoint_format="orbax",
            parent_refs_override=[current_ref],
            extra_overrides=["training.load_part=all"],
            extra_tags={
                "ablation_mode": "iso_quality",
                "continuation_source_stage_id": current_ref.checkpoint_id,
                "continuation_source_run_id": current_ref.exp_name,
                "continuation_source_step": str(current_source_step),
            },
        )
        run_dir = _run_dir_for(args.exp_dir, args.paper_run_id, stage_id, run_id)
        checkpoint_dir = _checkpoint_root_for(args.paper_run_id, args.checkpoint_root, run_id)
        latest_step = _latest_checkpoint_step_if_present(checkpoint_dir, dry_run=args.dry_run)
        if latest_step is None and args.dry_run:
            latest_step = total_steps_to_final_step(target_total_steps)
        mode_payload["runs"].append(
            {
                "stage_id": stage_id,
                "run_id": run_id,
                "status": result.status,
                "source_checkpoint_step": current_source_step,
                "target_total_steps": target_total_steps,
                "latest_checkpoint_step": latest_step,
                "run_dir": str(run_dir),
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
        if result.status not in {"succeeded", "dry_run"}:
            mode_payload["status"] = "failed"
            mode_payload["terminal"] = {
                "reason": "run_failed",
                "run_id": run_id,
                "status": result.status,
                "latest_checkpoint_step": latest_step,
            }
            ledger["modes"]["iso_quality"] = mode_payload
            return

        block_terminal_point: dict[str, Any] | None = None
        for eval_step in [current_source_step + args.step_interval, current_source_step + 2 * args.step_interval, current_source_step + 3 * args.step_interval]:
            if eval_step > latest_step:
                continue
            row = _eval_checkpoint_step(
                repo_root=repo_root,
                args=args,
                stage_id=stage_id,
                run_id=run_id,
                checkpoint_step=eval_step,
                mode="iso_quality",
            )
            point = _frontier_point_from_eval(
                mode="iso_quality",
                run_dir=run_dir,
                row=row,
                source_checkpoint_step=source_checkpoint_step,
                base_costs=branch_costs,
            )
            mode_payload["frontier_points"].append(point)
            losses = [float(item["loss_ce_mean"]) for item in mode_payload["frontier_points"] if item.get("loss_ce_mean") is not None]
            if bool(point["matched_target"]):
                block_terminal_point = point
                mode_payload["terminal"] = {
                    "reason": "target_reached",
                    "stage_id": stage_id,
                    "run_id": run_id,
                    "checkpoint_step": int(point["checkpoint_step"]),
                    "extra_steps": int(point["extra_steps"]),
                    "extra_tokens": int(point["extra_tokens"]),
                }
                break
            if should_stop_for_plateau(
                losses,
                min_improvement=ISO_QUALITY_PLATEAU_MIN_IMPROVEMENT,
                intervals=ISO_QUALITY_PLATEAU_INTERVALS,
            ):
                block_terminal_point = point
                mode_payload["terminal"] = {
                    "reason": "plateau",
                    "stage_id": stage_id,
                    "run_id": run_id,
                    "checkpoint_step": int(point["checkpoint_step"]),
                    "extra_steps": int(point["extra_steps"]),
                    "extra_tokens": int(point["extra_tokens"]),
                }
                break
        if block_terminal_point is not None:
            terminal_step = int(block_terminal_point["checkpoint_step"])
            if terminal_step != latest_step and not args.dry_run:
                terminal_total_steps = final_step_to_total_steps(terminal_step)
                terminal_run_id = _target_run_id("s2-125m-isoq-terminal", terminal_total_steps)
                _materialize_checkpoint_step_as_run(
                    source_stage_id=stage_id,
                    source_run_id=run_id,
                    source_run_dir=run_dir,
                    source_checkpoint_dir=checkpoint_dir,
                    target_stage_id=stage_id,
                    target_run_id=terminal_run_id,
                    target_paper_run_id=args.paper_run_id,
                    exp_dir=args.exp_dir,
                    checkpoint_root=args.checkpoint_root,
                    target_step=terminal_step,
                )
                mode_payload["terminal"]["materialized_run_id"] = terminal_run_id
            mode_payload["status"] = "dry_run" if args.dry_run else "succeeded"
            ledger["modes"]["iso_quality"] = mode_payload
            return

        current_ref = _ablation_source_ref(
            checkpoint_root=args.checkpoint_root,
            paper_run_id=args.paper_run_id,
            stage_id=stage_id,
            run_id=run_id,
        )
        current_checkpoint_dir = checkpoint_dir
        current_source_step = latest_step

    last_point = mode_payload["frontier_points"][-1]
    mode_payload["terminal"] = {
        "reason": "hard_cap",
        "stage_id": stage_id,
        "run_id": str(last_point["run_id"]),
        "checkpoint_step": int(last_point["checkpoint_step"]),
        "extra_steps": int(last_point["extra_steps"]),
        "extra_tokens": int(last_point["extra_tokens"]),
        "did_not_reach_target": True,
    }
    mode_payload["status"] = "dry_run" if args.dry_run else "succeeded"
    ledger["modes"]["iso_quality"] = mode_payload


def _iso_total_tokens_mode(args: argparse.Namespace, repo_root: Path, ledger: dict[str, Any]) -> None:
    template = _stage_template(args.registry, S3_ISOTOK_SOURCE[0])
    stage_id = S3_ISOTOK_STAGE_ID
    source_stage_id, source_run_id = S3_ISOTOK_SOURCE
    source_ref = _baseline_source_ref(
        checkpoint_root=args.checkpoint_root,
        stage_id=source_stage_id,
        run_id=source_run_id,
    )
    source_checkpoint_step = int(source_ref.step or 0)
    source_checkpoint_dir = _checkpoint_root_for(CANONICAL_PAPER_RUN_ID, args.checkpoint_root, source_run_id)
    branch_costs = _branch_costs_or_placeholder(args.exp_dir, "s3_branch", dry_run=args.dry_run)
    extra_steps = iso_total_extra_steps()
    target_total_steps = final_step_to_total_steps(source_checkpoint_step) + extra_steps
    run_id = _target_run_id("s3-125m-isotok", target_total_steps)
    stage = _ablation_stage(
        template,
        stage_id=stage_id,
        run_id=run_id,
        notes_suffix="Iso-total-tokens continuation from canonical S3_125M with load_part=all.",
    )
    opts = _base_opts(args)
    budget = _budget_for_total_steps(target_total_steps)
    eval_spec = EvalSpec(eval_id="ablation_books32k", contexts=[BOOKS_SEQ_LENGTH], datasets=["books3"], eval_split="val", eval_batches=args.eval_batches)
    result = run_stage(
        stage=stage,
        stage_map={},
        opts=opts,
        budget=budget,
        eval_spec=eval_spec,
        repo_root=repo_root,
        run_id=run_id,
        explicit_resume_checkpoint_path=source_checkpoint_dir,
        explicit_resume_checkpoint_format="orbax",
        parent_refs_override=[source_ref],
        extra_overrides=["training.load_part=all"],
        extra_tags={
            "ablation_mode": "iso_total_tokens",
            "continuation_source_stage_id": source_ref.checkpoint_id,
            "continuation_source_run_id": source_ref.exp_name,
            "continuation_source_step": str(source_checkpoint_step),
        },
    )
    run_dir = _run_dir_for(args.exp_dir, args.paper_run_id, stage_id, run_id)
    checkpoint_dir = _checkpoint_root_for(args.paper_run_id, args.checkpoint_root, run_id)
    latest_step = _latest_checkpoint_step_if_present(checkpoint_dir, dry_run=args.dry_run)
    if latest_step is None and args.dry_run:
        latest_step = total_steps_to_final_step(target_total_steps)
    mode_payload: dict[str, Any] = {
        "mode": "iso_total_tokens",
        "source": {
            "stage_id": source_stage_id,
            "run_id": source_run_id,
            "checkpoint_step": source_checkpoint_step,
            "checkpoint_path": source_ref.checkpoint_path,
            "extra_tokens_budget": ISO_TOTAL_EXTRA_TOKENS,
            "extra_steps_budget": extra_steps,
        },
        "runs": [
            {
                "stage_id": stage_id,
                "run_id": run_id,
                "status": result.status,
                "source_checkpoint_step": source_checkpoint_step,
                "target_total_steps": target_total_steps,
                "latest_checkpoint_step": latest_step,
                "run_dir": str(run_dir),
                "checkpoint_dir": str(checkpoint_dir),
            }
        ],
        "frontier_points": [
            _canonical_baseline_point(
                exp_dir=args.exp_dir,
                stage_id=source_stage_id,
                run_id=source_run_id,
                source_checkpoint_step=source_checkpoint_step,
                branch_costs=branch_costs,
                branch_label="iso_total_tokens",
            )
        ],
        "terminal": {},
        "status": "running",
    }
    if result.status not in {"succeeded", "dry_run"}:
        mode_payload["status"] = "failed"
        mode_payload["terminal"] = {
            "reason": "run_failed",
            "run_id": run_id,
            "status": result.status,
            "latest_checkpoint_step": latest_step,
        }
        ledger["modes"]["iso_total_tokens"] = mode_payload
        return

    for offset in range(args.step_interval, extra_steps + 1, args.step_interval):
        checkpoint_step = source_checkpoint_step + offset
        row = _eval_checkpoint_step(
            repo_root=repo_root,
            args=args,
            stage_id=stage_id,
            run_id=run_id,
            checkpoint_step=checkpoint_step,
            mode="iso_total_tokens",
        )
        point = _frontier_point_from_eval(
            mode="iso_total_tokens",
            run_dir=run_dir,
            row=row,
            source_checkpoint_step=source_checkpoint_step,
            base_costs=branch_costs,
        )
        mode_payload["frontier_points"].append(point)

    final_point = mode_payload["frontier_points"][-1]
    mode_payload["terminal"] = {
        "reason": "equal_total_tokens_endpoint",
        "stage_id": stage_id,
        "run_id": run_id,
        "checkpoint_step": int(final_point["checkpoint_step"]),
        "extra_steps": int(final_point["extra_steps"]),
        "extra_tokens": int(final_point["extra_tokens"]),
    }
    mode_payload["status"] = "dry_run" if args.dry_run else "succeeded"
    ledger["modes"]["iso_total_tokens"] = mode_payload


def _export_terminals(repo_root: Path, args: argparse.Namespace, ledger: dict[str, Any]) -> None:
    for mode_name, payload in ledger.get("modes", {}).items():
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status", "")) != "succeeded":
            continue
        terminal = payload.get("terminal", {})
        if not isinstance(terminal, dict):
            continue
        stage_id = str(terminal.get("stage_id", "")).strip()
        run_id = str(terminal.get("materialized_run_id") or terminal.get("run_id") or "").strip()
        if not stage_id or not run_id:
            continue
        _export_terminal_run(repo_root=repo_root, args=args, stage_id=stage_id, run_id=run_id)


def _write_ledger(args: argparse.Namespace, ledger: dict[str, Any]) -> Path:
    ledger_path = args.reports_root / "runner" / "ablation_ledger.json"
    _write_json(ledger_path, ledger)
    return ledger_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 125M continuation ablations outside the canonical ladder.")
    parser.add_argument("--paper-run-id", default=ABLATION_PAPER_RUN_ID)
    parser.add_argument("--canonical-paper-run-id", default=CANONICAL_PAPER_RUN_ID)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--mode", choices=["iso_quality", "iso_total_tokens", "all"], default="all")
    parser.add_argument("--deploy", default="interactive")
    parser.add_argument("--runtime-mode", default="jax_train", choices=["simulate", "token_stats", "jax_train"])
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--wandb-entity", default="phase1")
    parser.add_argument("--wandb-project", default="phase1")
    parser.add_argument("--wandb-key", default="none")
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--step-interval", type=int, default=STEP_INTERVAL)
    parser.add_argument("--export-final-to-hf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()
    args.registry = args.registry.expanduser().resolve()
    args.exp_dir = args.exp_dir.expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.profile_root = args.profile_root.expanduser().resolve()
    args.dclm_root = args.dclm_root.expanduser().resolve()
    args.books_root = args.books_root.expanduser().resolve()
    if args.reports_root is None:
        args.reports_root = (Path("./reports/paper") / args.paper_run_id).expanduser().resolve()
    else:
        args.reports_root = args.reports_root.expanduser().resolve()

    if args.step_interval != STEP_INTERVAL:
        raise ValueError(f"This runner currently requires --step-interval={STEP_INTERVAL}")

    if args.canonical_paper_run_id != CANONICAL_PAPER_RUN_ID:
        raise ValueError(f"This runner expects canonical-paper-run-id={CANONICAL_PAPER_RUN_ID}")

    if not args.dry_run:
        for stage_id, run_id in _canonical_stage_refs():
            _restore_stage_if_missing(repo_root=repo_root, args=args, stage_id=stage_id, run_id=run_id)

    ledger: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "paper_run_id": args.paper_run_id,
        "canonical_paper_run_id": args.canonical_paper_run_id,
        "mode": args.mode,
        "step_interval": args.step_interval,
        "eval_batches": args.eval_batches,
        "iso_quality_target_loss_ce_mean": ISO_QUALITY_TARGET_LOSS,
        "iso_quality_plateau_min_improvement": ISO_QUALITY_PLATEAU_MIN_IMPROVEMENT,
        "iso_quality_plateau_intervals": ISO_QUALITY_PLATEAU_INTERVALS,
        "iso_quality_hard_cap_extra_steps": ISO_QUALITY_HARD_CAP_EXTRA_STEPS,
        "iso_total_extra_tokens": ISO_TOTAL_EXTRA_TOKENS,
        "iso_total_extra_steps": iso_total_extra_steps(),
        "modes": {},
    }

    for mode_name in _parse_mode(args.mode):
        if mode_name == "iso_quality":
            _iso_quality(args, repo_root, ledger)
        elif mode_name == "iso_total_tokens":
            _iso_total_tokens_mode(args, repo_root, ledger)
        else:
            raise ValueError(f"Unsupported mode: {mode_name}")
        _write_ledger(args, ledger)

    if args.export_final_to_hf:
        _export_terminals(repo_root, args, ledger)
        _write_ledger(args, ledger)

    summarize_cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/61_summarize_125m_continuation_ablations.py",
        "--paper-run-id",
        args.paper_run_id,
        "--canonical-paper-run-id",
        args.canonical_paper_run_id,
        "--exp-dir",
        str(args.exp_dir),
        "--reports-root",
        str(args.reports_root),
    ]
    rc = _run(summarize_cmd, cwd=repo_root, dry_run=args.dry_run)
    if rc != 0:
        raise RuntimeError(f"Failed to summarize continuation ablations (rc={rc})")
    print(f"Wrote ablation ledger: {_write_ledger(args, ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
