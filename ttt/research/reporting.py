"""Reporting helpers for stage aggregation and paper tables."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .types import PaperTableRow



def _safe_float(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None



def _ci95(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(float(n))



def collect_eval_manifests(exp_dir: Path, paper_run_id: str) -> list[dict[str, Any]]:
    root = exp_dir / paper_run_id
    if not root.exists():
        return []

    out: list[dict[str, Any]] = []
    for manifest in root.rglob("eval_manifest.json"):
        payload = json.loads(manifest.read_text())
        if isinstance(payload, dict):
            out.append(payload)
    return out



def aggregate_stage_metrics(
    manifests: list[dict[str, Any]],
    metric: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in manifests:
        status = str(item.get("status", ""))
        if status not in {"succeeded", "dry_run"}:
            continue
        stage_id = str(item.get("stage_id", ""))
        value = _safe_float(item.get("metrics", {}).get(metric))
        if stage_id and value is not None:
            grouped[stage_id].append(value)

    summary: dict[str, dict[str, float]] = {}
    for stage_id, values in grouped.items():
        n = len(values)
        m = mean(values)
        std = pstdev(values) if n > 1 else 0.0
        ci = _ci95(std, n)
        summary[stage_id] = {
            "n": float(n),
            "mean": m,
            "std": std,
            "ci95_low": m - ci,
            "ci95_high": m + ci,
        }
    return summary



def stage_delta_table(
    *,
    paper_run_id: str,
    metric: str,
    from_stage: str,
    to_stage: str,
    stage_summary: dict[str, dict[str, float]],
    unit: str = "",
    notes: str = "",
) -> PaperTableRow:
    from_stats = stage_summary.get(from_stage)
    to_stats = stage_summary.get(to_stage)
    if from_stats is None or to_stats is None:
        raise KeyError(
            f"Missing stage summary for delta table: from={from_stage} to={to_stage}"
        )

    n = int(min(from_stats["n"], to_stats["n"]))
    delta = to_stats["mean"] - from_stats["mean"]

    # Conservative approximation since pairwise seed alignment may be absent.
    std_delta = math.sqrt(from_stats["std"] ** 2 + to_stats["std"] ** 2)
    ci = _ci95(std_delta, max(n, 1))

    return PaperTableRow(
        paper_run_id=paper_run_id,
        metric=metric,
        stage_from=from_stage,
        stage_to=to_stage,
        n=n,
        mean_from=from_stats["mean"],
        mean_to=to_stats["mean"],
        delta=delta,
        std_delta=std_delta,
        ci95_low=delta - ci,
        ci95_high=delta + ci,
        unit=unit,
        notes=notes,
    )



def write_stage_summary_csv(path: Path, stage_summary: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage_id", "n", "mean", "std", "ci95_low", "ci95_high"],
        )
        writer.writeheader()
        for stage_id in sorted(stage_summary.keys()):
            row = {"stage_id": stage_id, **stage_summary[stage_id]}
            writer.writerow(row)



def write_paper_rows_csv(path: Path, rows: list[PaperTableRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "paper_run_id",
                    "metric",
                    "stage_from",
                    "stage_to",
                    "n",
                    "mean_from",
                    "mean_to",
                    "delta",
                    "std_delta",
                    "ci95_low",
                    "ci95_high",
                    "unit",
                    "notes",
                ]
            )
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].to_dict().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())



def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
