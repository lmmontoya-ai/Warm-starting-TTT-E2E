#!/usr/bin/env python3
"""Analyze revision-v2 E1 paired bridge-effect results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from ttt.research.preregistration import bridge_effect_from_losses


SEED_RE = re.compile(r"seed(\d{3})$")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _parse_losses(eval_summary: dict[str, Any]) -> tuple[list[int], list[float], list[float]]:
    losses: dict[str, dict[int, float]] = {"S2_125M": {}, "S2_MINUS_125M": {}}
    for row in eval_summary.get("rows", []):
        if not isinstance(row, dict):
            continue
        stage_id = str(row.get("stage_id", ""))
        if stage_id not in losses:
            continue
        run_id = str(row.get("run_id", ""))
        match = SEED_RE.search(run_id)
        if match is None:
            continue
        if str(row.get("status", "")) != "succeeded":
            raise ValueError(f"Cannot analyze failed eval row: {row}")
        losses[stage_id][int(match.group(1))] = float(row["loss_mean"])

    seeds = [1, 2, 3, 4, 5]
    missing = {stage: sorted(set(seeds) - set(vals)) for stage, vals in losses.items()}
    if any(missing.values()):
        raise ValueError(f"Missing paired losses: {missing}")
    s2_losses = [losses["S2_125M"][seed] for seed in seeds]
    s2_minus_losses = [losses["S2_MINUS_125M"][seed] for seed in seeds]
    return seeds, s2_losses, s2_minus_losses


def _baseline_losses(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) == "succeeded" and row.get("loss_mean") is not None:
            out[str(row["stage_id"])] = float(row["loss_mean"])
    return out


def _gpu_hours(run_summary: dict[str, Any]) -> tuple[float, dict[str, float]]:
    total = 0.0
    by_stage: dict[str, float] = {}
    for row in run_summary.get("rows", []):
        if not isinstance(row, dict) or str(row.get("status", "")) != "succeeded":
            continue
        gpu_hours = float(row["gpu_hours"])
        stage_id = str(row["stage_id"])
        total += gpu_hours
        by_stage[stage_id] = by_stage.get(stage_id, 0.0) + gpu_hours
    return total, by_stage


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "s2_loss", "s2_minus_loss", "delta_s2_minus_minus_s2"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    decision: str,
    paired_rows: list[dict[str, Any]],
    s2_mean: float,
    s2_minus_mean: float,
    delta_mean: float,
    ci_low: float,
    ci_high: float,
    baseline_losses: dict[str, float],
    gpu_total: float,
    gpu_by_stage: dict[str, float],
) -> None:
    lines = [
        "# Revision V2 E1 Bridge-Effect Analysis",
        "",
        f"**Decision:** `{decision}`",
        "",
        "Preregistered statistic: `loss(S2_MINUS_s) - loss(S2_s)` on paired seeds `1..5`.",
        "Positive values mean the bridge improves Books32K validation loss.",
        "",
        f"- Mean S2 loss: `{s2_mean:.6f}`",
        f"- Mean S2-minus loss: `{s2_minus_mean:.6f}`",
        f"- Mean delta: `{delta_mean:.6f}`",
        f"- 95% paired seed-level bootstrap CI: `[{ci_low:.6f}, {ci_high:.6f}]`",
        "- Margin: `0.10`",
        "",
        "## Paired Losses",
        "",
        "| seed | S2 | S2-minus | delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            "| {seed} | {s2_loss:.6f} | {s2_minus_loss:.6f} | "
            "{delta_s2_minus_minus_s2:.6f} |".format(**row)
        )
    lines.extend(["", "## Current-Pipeline Baseline Re-Eval", ""])
    for stage_id in ("S1_125M", "S2_125M", "S3_125M"):
        value = baseline_losses.get(stage_id)
        if value is None:
            lines.append(f"- {stage_id}: missing")
        else:
            lines.append(f"- {stage_id}: `{value:.6f}`")
    lines.extend(
        [
            "",
            "## Training Cost Observed",
            "",
            f"- Total paired training GPU-hours: `{gpu_total:.3f}`",
            f"- S2_125M seed arms: `{gpu_by_stage.get('S2_125M', 0.0):.3f}`",
            f"- S2_MINUS_125M seed arms: `{gpu_by_stage.get('S2_MINUS_125M', 0.0):.3f}`",
            "",
            "This analysis does not use retrieval results and does not alter the preregistered E1 loss decision.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=Path("reports/revision_v2/e1_pairs/e1_pairs_books32k_jax_eval_summary.json"),
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=Path("reports/revision_v2/e1_pairs/e1_pairs_run_summary.json"),
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("reports/revision_v2/current_eval/s1_s2_s3_books32k_jax_eval_summary.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/revision_v2/e1_pairs"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_summary = _load_json(args.eval_summary)
    run_summary = _load_json(args.run_summary)
    baseline_summary = _load_json(args.baseline_summary)

    seeds, s2_losses, s2_minus_losses = _parse_losses(eval_summary)
    result = bridge_effect_from_losses(
        s2_minus_losses=s2_minus_losses,
        s2_losses=s2_losses,
        margin=args.margin,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    paired_rows = [
        {
            "seed": seed,
            "s2_loss": s2_loss,
            "s2_minus_loss": s2_minus_loss,
            "delta_s2_minus_minus_s2": delta,
        }
        for seed, s2_loss, s2_minus_loss, delta in zip(
            seeds,
            s2_losses,
            s2_minus_losses,
            result.deltas,
            strict=True,
        )
    ]
    baseline_losses = _baseline_losses(baseline_summary)
    gpu_total, gpu_by_stage = _gpu_hours(run_summary)

    analysis = {
        "schema_version": "1.0",
        "paper_run_id": "revision_v2_e1_paired_v1",
        "eval_surface": {
            "dataset": "books3",
            "context_length": 32768,
            "eval_batches": 8,
            "eval_pipeline": (
                "scripts/34_eval_matrix_jax.py current-pipeline JAX eval; "
                "float32 aggregation in ttt/jax_runtime/loop.py"
            ),
            "summary_json": str(args.eval_summary),
        },
        "preregistered_statistic": "delta_bridge_s = loss(S2_MINUS_s) - loss(S2_s)",
        "margin": args.margin,
        "seeds": seeds,
        "paired_rows": paired_rows,
        "s2_loss_mean": mean(s2_losses),
        "s2_minus_loss_mean": mean(s2_minus_losses),
        "delta_mean": result.mean_delta,
        "delta_ci95": {"low": result.ci95.low, "high": result.ci95.high},
        "decision": result.decision,
        "bootstrap": {
            "n_resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "level": 0.95,
            "unit": "paired training seed delta",
        },
        "baseline_current_pipeline_losses": baseline_losses,
        "training_gpu_hours": {"total": gpu_total, "by_stage": gpu_by_stage},
        "training_run_summary": str(args.run_summary),
        "notes": [
            "Decision is based only on paired Books32K validation loss, not retrieval or training loss.",
            "Positive delta means the bridge helps because S2-minus loss is higher than S2 loss.",
            "All ten paired training runs and HF exports succeeded before evaluation.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "e1_bridge_effect_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.out_dir / "e1_bridge_effect_paired_losses.csv", paired_rows)
    _write_markdown(
        args.out_dir / "e1_bridge_effect_analysis.md",
        decision=result.decision,
        paired_rows=paired_rows,
        s2_mean=mean(s2_losses),
        s2_minus_mean=mean(s2_minus_losses),
        delta_mean=result.mean_delta,
        ci_low=result.ci95.low,
        ci_high=result.ci95.high,
        baseline_losses=baseline_losses,
        gpu_total=gpu_total,
        gpu_by_stage=gpu_by_stage,
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
