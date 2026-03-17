from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any


def read_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and "step" in payload:
            rows.append(payload)
    return rows


def step_wall_seconds(row: dict[str, Any]) -> float:
    total = 0.0
    for key in ("data_wait_seconds", "batch_sharding_seconds", "train_step_seconds"):
        try:
            total += float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return max(total, 0.0)


def estimate_stage_training_wall_seconds(
    *,
    metrics_rows: list[dict[str, Any]],
    total_steps: int,
    warmup_steps: int = 2,
) -> dict[str, float]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if not metrics_rows:
        raise ValueError("metrics_rows must be non-empty.")

    observed = len(metrics_rows)
    warmup_count = min(warmup_steps, observed)
    warmup_rows = metrics_rows[:warmup_count]
    steady_rows = metrics_rows[warmup_count:] or metrics_rows[-1:]

    warmup_wall = sum(step_wall_seconds(row) for row in warmup_rows)
    steady_step_wall = median(step_wall_seconds(row) for row in steady_rows)
    remaining_steps = max(total_steps - warmup_count, 0)
    estimated_wall = warmup_wall + steady_step_wall * remaining_steps

    return {
        "observed_steps": float(observed),
        "warmup_steps": float(warmup_count),
        "warmup_wall_seconds": float(warmup_wall),
        "steady_step_wall_seconds": float(steady_step_wall),
        "estimated_training_wall_seconds": float(estimated_wall),
    }
