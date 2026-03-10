#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ttt.research.reporting import (
    aggregate_stage_metrics,
    collect_eval_manifests,
    stage_delta_table,
    write_paper_rows_csv,
    write_stage_summary_csv,
)


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _collect_run_results(exp_dir: Path, paper_run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    root = exp_dir / paper_run_id
    if not root.exists():
        return out

    for path in root.rglob("run_result.json"):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            continue
        stage_id = str(payload.get("stage_id", "")).strip()
        run_id = str(payload.get("run_id", "")).strip()
        if not stage_id or not run_id:
            continue
        out[(stage_id, run_id)] = payload
    return out


def _write_frontier_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fields = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate eval manifests into paper-facing stage deltas and "
            "quality-vs-compute frontier tables."
        )
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument(
        "--metrics",
        default="loss_mean,niah_accuracy_mean,ruler_accuracy_mean,tokens_per_second_mean",
        help="Comma-separated eval metrics to aggregate.",
    )

    parser.add_argument("--s0-stage", default="S0")
    parser.add_argument("--s1-stage", default="S1")
    parser.add_argument("--s2-stage", default="S2")
    parser.add_argument("--s3-stage", default="S3")

    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()

    if args.tables_dir is None:
        tables_dir = (Path("./reports/paper") / args.paper_run_id / "tables").resolve()
    else:
        tables_dir = args.tables_dir.expanduser().resolve()

    if args.summary_json is None:
        summary_json = (Path("./reports/paper") / args.paper_run_id / "eval" / "aggregate_summary.json").resolve()
    else:
        summary_json = args.summary_json.expanduser().resolve()

    metrics = _parse_csv(args.metrics)
    if not metrics:
        raise ValueError("No metrics provided")

    manifests = collect_eval_manifests(exp_dir=exp_dir, paper_run_id=args.paper_run_id)
    if not manifests:
        raise FileNotFoundError(
            f"No eval manifests found under {exp_dir / args.paper_run_id}. "
            "Run scripts/18_eval_matrix.py first."
        )

    degradation_rows = []
    recovery_rows = []
    warmstart_tax_rows = []
    stage_summary_paths: list[str] = []

    for metric in metrics:
        summary = aggregate_stage_metrics(manifests=manifests, metric=metric)
        summary_path = tables_dir / f"stage_summary_{metric}.csv"
        write_stage_summary_csv(summary_path, summary)
        stage_summary_paths.append(str(summary_path))

        if args.s0_stage in summary and args.s1_stage in summary:
            degradation_rows.append(
                stage_delta_table(
                    paper_run_id=args.paper_run_id,
                    metric=metric,
                    from_stage=args.s0_stage,
                    to_stage=args.s1_stage,
                    stage_summary=summary,
                    notes="S1-S0 degradation: stage_to - stage_from",
                )
            )

        if args.s1_stage in summary and args.s2_stage in summary:
            recovery_rows.append(
                stage_delta_table(
                    paper_run_id=args.paper_run_id,
                    metric=metric,
                    from_stage=args.s1_stage,
                    to_stage=args.s2_stage,
                    stage_summary=summary,
                    notes="S2-S1 recovery: stage_to - stage_from",
                )
            )

        if args.s2_stage in summary and args.s3_stage in summary:
            warmstart_tax_rows.append(
                stage_delta_table(
                    paper_run_id=args.paper_run_id,
                    metric=metric,
                    from_stage=args.s3_stage,
                    to_stage=args.s2_stage,
                    stage_summary=summary,
                    notes="S2-S3 warm-start tax: stage_to - stage_from",
                )
            )

    degradation_path = tables_dir / "s1_s0_degradation.csv"
    recovery_path = tables_dir / "s2_s1_recovery.csv"
    tax_path = tables_dir / "s2_s3_warmstart_tax.csv"
    write_paper_rows_csv(degradation_path, degradation_rows)
    write_paper_rows_csv(recovery_path, recovery_rows)
    write_paper_rows_csv(tax_path, warmstart_tax_rows)

    run_results = _collect_run_results(exp_dir=exp_dir, paper_run_id=args.paper_run_id)
    frontier_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        stage_id = str(manifest.get("stage_id", ""))
        run_id = str(manifest.get("run_id", ""))
        status = str(manifest.get("status", ""))
        if status not in {"succeeded", "dry_run"}:
            continue
        if not stage_id or not run_id:
            continue
        run_result = run_results.get((stage_id, run_id), {})
        metrics_map = manifest.get("metrics", {})
        if not isinstance(metrics_map, dict):
            metrics_map = {}
        frontier_rows.append(
            {
                "paper_run_id": args.paper_run_id,
                "stage_id": stage_id,
                "run_id": run_id,
                "status": status,
                "gpu_hours": run_result.get("gpu_hours"),
                "wall_seconds": run_result.get("wall_seconds"),
                "tokens_seen": run_result.get("tokens_seen"),
                "loss_mean": metrics_map.get("loss_mean"),
                "niah_accuracy_mean": metrics_map.get("niah_accuracy_mean"),
                "tokens_per_second_mean": metrics_map.get("tokens_per_second_mean"),
                "eval_wall_seconds": metrics_map.get("eval_wall_seconds"),
            }
        )

    frontier_path = tables_dir / "quality_vs_gpu_hour_frontier.csv"
    _write_frontier_csv(frontier_path, frontier_rows)

    summary = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "metrics": metrics,
        "n_eval_manifests": len(manifests),
        "n_frontier_rows": len(frontier_rows),
        "stage_summary_paths": stage_summary_paths,
        "degradation_table": str(degradation_path),
        "recovery_table": str(recovery_path),
        "warmstart_tax_table": str(tax_path),
        "frontier_table": str(frontier_path),
    }
    _write_json(summary_json, summary)

    print(f"Wrote aggregate summary: {summary_json}")
    print(f"Wrote table: {degradation_path}")
    print(f"Wrote table: {recovery_path}")
    print(f"Wrote table: {tax_path}")
    print(f"Wrote table: {frontier_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
