#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_aggregate_if_needed(args: argparse.Namespace, tables_dir: Path) -> int:
    if args.skip_aggregate:
        return 0
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/19_eval_aggregate.py",
        "--paper-run-id",
        args.paper_run_id,
        "--exp-dir",
        str(args.exp_dir),
        "--tables-dir",
        str(tables_dir),
        "--s0-stage",
        args.s0_stage,
        "--s1-stage",
        args.s1_stage,
        "--s2-stage",
        args.s2_stage,
        "--s3-stage",
        args.s3_stage,
    ]
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _build_core_delta_table(tables_dir: Path) -> list[dict[str, Any]]:
    deg = _read_csv(tables_dir / "s1_s0_degradation.csv")
    rec = _read_csv(tables_dir / "s2_s1_recovery.csv")
    tax = _read_csv(tables_dir / "s2_s3_warmstart_tax.csv")

    by_metric: dict[str, dict[str, Any]] = {}

    def ingest(rows: list[dict[str, str]], prefix: str) -> None:
        for row in rows:
            metric = row.get("metric", "")
            if not metric:
                continue
            slot = by_metric.setdefault("metric:" + metric, {"metric": metric})
            slot[f"{prefix}_delta"] = row.get("delta", "")
            slot[f"{prefix}_ci95_low"] = row.get("ci95_low", "")
            slot[f"{prefix}_ci95_high"] = row.get("ci95_high", "")
            slot[f"{prefix}_n"] = row.get("n", "")

    ingest(deg, "s1_s0")
    ingest(rec, "s2_s1")
    ingest(tax, "s2_s3")

    rows = list(by_metric.values())
    rows.sort(key=lambda x: str(x.get("metric", "")))
    return rows


def _collect_run_inventory(exp_dir: Path, paper_run_id: str) -> list[dict[str, Any]]:
    root = exp_dir / paper_run_id
    if not root.exists():
        return []

    out: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        run_manifest = json.loads(manifest_path.read_text())
        run_result_path = run_dir / "run_result.json"
        eval_manifest_path = run_dir / "eval_manifest.json"

        run_result = json.loads(run_result_path.read_text()) if run_result_path.exists() else {}
        eval_manifest = json.loads(eval_manifest_path.read_text()) if eval_manifest_path.exists() else {}
        eval_metrics = eval_manifest.get("metrics", {}) if isinstance(eval_manifest, dict) else {}
        if not isinstance(eval_metrics, dict):
            eval_metrics = {}

        out.append(
            {
                "stage_id": run_manifest.get("stage_id", ""),
                "run_id": run_manifest.get("run_id", ""),
                "exp_name": run_manifest.get("exp_name", ""),
                "model_key": run_manifest.get("model_key", ""),
                "path_group": run_manifest.get("path_group", ""),
                "train_kind": (run_manifest.get("tags", {}) or {}).get("kind", ""),
                "train_mode": (run_manifest.get("tags", {}) or {}).get("train_mode", ""),
                "status": run_result.get("status", ""),
                "tokens_seen": run_result.get("tokens_seen", ""),
                "gpu_hours": run_result.get("gpu_hours", ""),
                "wall_seconds": run_result.get("wall_seconds", ""),
                "loss_mean": eval_metrics.get("loss_mean", ""),
                "niah_accuracy_mean": eval_metrics.get("niah_accuracy_mean", ""),
                "tokens_per_second_mean": eval_metrics.get("tokens_per_second_mean", ""),
            }
        )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic paper tables from manifests and aggregate outputs."
        )
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--s0-stage", default="S0")
    parser.add_argument("--s1-stage", default="S1")
    parser.add_argument("--s2-stage", default="S2")
    parser.add_argument("--s3-stage", default="S3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()

    if args.tables_dir is None:
        tables_dir = (Path("./reports/paper") / args.paper_run_id / "tables").resolve()
    else:
        tables_dir = args.tables_dir.expanduser().resolve()

    if args.summary_json is None:
        summary_json = (
            Path("./reports/paper") / args.paper_run_id / "tables" / "paper_tables_summary.json"
        ).resolve()
    else:
        summary_json = args.summary_json.expanduser().resolve()

    rc = _run_aggregate_if_needed(args=args, tables_dir=tables_dir)
    if rc != 0:
        return rc

    core_rows = _build_core_delta_table(tables_dir=tables_dir)
    core_path = tables_dir / "warmstart_core_deltas.csv"
    _write_csv(
        core_path,
        core_rows,
        fields=[
            "metric",
            "s1_s0_delta",
            "s1_s0_ci95_low",
            "s1_s0_ci95_high",
            "s1_s0_n",
            "s2_s1_delta",
            "s2_s1_ci95_low",
            "s2_s1_ci95_high",
            "s2_s1_n",
            "s2_s3_delta",
            "s2_s3_ci95_low",
            "s2_s3_ci95_high",
            "s2_s3_n",
        ],
    )

    inventory_rows = _collect_run_inventory(exp_dir=exp_dir, paper_run_id=args.paper_run_id)
    inventory_path = tables_dir / "run_inventory.csv"
    _write_csv(
        inventory_path,
        inventory_rows,
        fields=[
            "stage_id",
            "run_id",
            "exp_name",
            "model_key",
            "path_group",
            "train_kind",
            "train_mode",
            "status",
            "tokens_seen",
            "gpu_hours",
            "wall_seconds",
            "loss_mean",
            "niah_accuracy_mean",
            "tokens_per_second_mean",
        ],
    )

    summary = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "tables_dir": str(tables_dir),
        "core_deltas_table": str(core_path),
        "run_inventory_table": str(inventory_path),
        "n_core_rows": len(core_rows),
        "n_inventory_rows": len(inventory_rows),
    }
    _write_json(summary_json, summary)

    print(f"Wrote table: {core_path}")
    print(f"Wrote table: {inventory_path}")
    print(f"Wrote summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
