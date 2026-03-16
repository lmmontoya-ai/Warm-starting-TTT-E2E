"""Helpers for 125M continuation ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CANONICAL_PAPER_RUN_ID = "protocol_r_125m_main_v1"
ABLATION_PAPER_RUN_ID = "protocol_r_125m_ablations_v1"

BOOKS_SEQ_LENGTH = 32768
PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE = 8
STEP_INTERVAL = 120
ISO_QUALITY_TARGET_LOSS = 3.2729225158691406
ISO_QUALITY_PLATEAU_MIN_IMPROVEMENT = 0.02
ISO_QUALITY_PLATEAU_INTERVALS = 3
ISO_QUALITY_HARD_CAP_EXTRA_STEPS = 1440
ISO_TOTAL_EXTRA_TOKENS = 251_658_240

S2_ISOQ_STAGE_ID = "S2_125M_ISOQ"
S3_ISOTOK_STAGE_ID = "S3_125M_ISOTOK"
S2_ISOQ_SOURCE = ("S2_125M", "ext-125m-e2e-32K-from-fa-bridge")
S3_ISOTOK_SOURCE = ("S3_125M", "ext-125m-e2e-32K")
CANONICAL_FINAL_CHECKPOINT_STEPS: dict[tuple[str, str], int] = {
    S2_ISOQ_SOURCE: 479,
    S3_ISOTOK_SOURCE: 479,
}


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


def extra_tokens_for_steps(steps: int) -> int:
    return max(0, int(steps)) * PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE * BOOKS_SEQ_LENGTH


def iso_total_extra_steps() -> int:
    step_tokens = extra_tokens_for_steps(1)
    if ISO_TOTAL_EXTRA_TOKENS % step_tokens != 0:
        raise ValueError(
            f"Expected ISO_TOTAL_EXTRA_TOKENS={ISO_TOTAL_EXTRA_TOKENS} to be divisible by step_tokens={step_tokens}"
        )
    return ISO_TOTAL_EXTRA_TOKENS // step_tokens


def final_step_to_total_steps(final_checkpoint_step: int) -> int:
    return max(0, int(final_checkpoint_step) + 1)


def total_steps_to_final_step(total_steps: int) -> int:
    return max(0, int(total_steps) - 1)


def extra_steps_from_checkpoint_step(*, source_checkpoint_step: int, checkpoint_step: int) -> int:
    return max(0, int(checkpoint_step) - int(source_checkpoint_step))


def should_stop_for_plateau(
    losses: list[float],
    *,
    min_improvement: float = ISO_QUALITY_PLATEAU_MIN_IMPROVEMENT,
    intervals: int = ISO_QUALITY_PLATEAU_INTERVALS,
) -> bool:
    if len(losses) < intervals + 1:
        return False
    recent = losses[-(intervals + 1) :]
    improvements = [float(prev) - float(curr) for prev, curr in zip(recent, recent[1:])]
    return all(improvement < float(min_improvement) for improvement in improvements)


def device_count_from_events(events_path: Path) -> int:
    for row in _load_jsonl(events_path):
        if str(row.get("event", "")).strip() != "run_started":
            continue
        device_info = row.get("device_info", {})
        if isinstance(device_info, dict):
            value = device_info.get("device_count")
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0:
                return parsed
    return 8


def cumulative_wall_by_checkpoint(metrics_path: Path) -> dict[int, float]:
    cumulative = 0.0
    checkpoint_wall: dict[int, float] = {}
    for row in _load_jsonl(metrics_path):
        if "train_step_seconds" in row:
            cumulative += float(row.get("data_wait_seconds", 0.0) or 0.0)
            cumulative += float(row.get("batch_sharding_seconds", 0.0) or 0.0)
            cumulative += float(row.get("train_step_seconds", 0.0) or 0.0)
        if str(row.get("event", "")).strip() == "checkpoint_saved":
            step = int(row.get("step", -1))
            cumulative += float(row.get("checkpoint_save_seconds", 0.0) or 0.0)
            if step >= 0:
                checkpoint_wall[step] = cumulative
    return checkpoint_wall


def latest_checkpoint_step(checkpoint_dir: Path) -> int:
    latest_json = checkpoint_dir / "latest.json"
    latest = _load_json(latest_json)
    return int(latest["step"])


def checkpoint_steps_from_root(checkpoint_dir: Path) -> list[int]:
    steps: list[int] = []
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    return sorted(steps)


def canonical_eval_loss(exp_dir: Path, *, stage_id: str, run_id: str, paper_run_id: str = CANONICAL_PAPER_RUN_ID) -> float:
    path = exp_dir / paper_run_id / stage_id / run_id / "eval_manifest.json"
    payload = _load_json(path)
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics payload in {path}")
    return float(metrics["loss_ce_mean"])


def canonical_run_result(exp_dir: Path, *, stage_id: str, run_id: str, paper_run_id: str = CANONICAL_PAPER_RUN_ID) -> dict[str, Any]:
    path = exp_dir / paper_run_id / stage_id / run_id / "run_result.json"
    return _load_json(path)


def canonical_branch_costs(exp_dir: Path) -> dict[str, dict[str, float]]:
    s0_pre = canonical_run_result(exp_dir, stage_id="S0_PRETRAIN_FA_125M", run_id="pretrain-125m-fa")
    s2_adapt = canonical_run_result(exp_dir, stage_id="S2_ADAPT_125M", run_id="adapt-125m-e2e-8K-from-fa")
    s2 = canonical_run_result(exp_dir, stage_id=S2_ISOQ_SOURCE[0], run_id=S2_ISOQ_SOURCE[1])
    s3_pre = canonical_run_result(exp_dir, stage_id="S3_PRETRAIN_E2E_125M", run_id="pretrain-125m-e2e")
    s3 = canonical_run_result(exp_dir, stage_id=S3_ISOTOK_SOURCE[0], run_id=S3_ISOTOK_SOURCE[1])

    return {
        "s2_branch": {
            "base_total_tokens": float(
                int(s0_pre.get("tokens_seen", 0) or 0)
                + int(s2_adapt.get("tokens_seen", 0) or 0)
                + int(s2.get("tokens_seen", 0) or 0)
            ),
            "base_total_gpu_hours": float(
                float(s0_pre.get("gpu_hours", 0.0) or 0.0)
                + float(s2_adapt.get("gpu_hours", 0.0) or 0.0)
                + float(s2.get("gpu_hours", 0.0) or 0.0)
            ),
            "base_marginal_gpu_hours": float(
                float(s2_adapt.get("gpu_hours", 0.0) or 0.0)
                + float(s2.get("gpu_hours", 0.0) or 0.0)
            ),
        },
        "s3_branch": {
            "base_total_tokens": float(
                int(s3_pre.get("tokens_seen", 0) or 0)
                + int(s3.get("tokens_seen", 0) or 0)
            ),
            "base_total_gpu_hours": float(
                float(s3_pre.get("gpu_hours", 0.0) or 0.0)
                + float(s3.get("gpu_hours", 0.0) or 0.0)
            ),
            "base_marginal_gpu_hours": float(s3.get("gpu_hours", 0.0) or 0.0),
        },
    }


def load_snapshot_eval(snapshot_dir: Path) -> dict[str, Any]:
    payload = _load_json(snapshot_dir / "eval_manifest_snapshot.json")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics payload in snapshot: {snapshot_dir}")
    return {
        "status": str(payload.get("status", "")),
        "loss_ce_mean": float(metrics["loss_ce_mean"]),
        "loss_mean": float(metrics.get("loss_mean", metrics["loss_ce_mean"])),
        "tokens_per_second_mean": float(metrics["tokens_per_second_mean"]),
        "eval_wall_seconds": float(metrics["eval_wall_seconds"]),
    }


def expected_canonical_checkpoint_step(stage_id: str, run_id: str) -> int:
    return int(CANONICAL_FINAL_CHECKPOINT_STEPS.get((stage_id, run_id), 0))
