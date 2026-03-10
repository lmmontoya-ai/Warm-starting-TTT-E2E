#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ttt.research.ruler_runner import compute_ruler_metrics_from_eval_csv, merge_metrics


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            "Attach RULER-style retrieval metrics to eval manifests from existing "
            "eval_raw.csv files."
        )
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_root = args.exp_dir.expanduser().resolve() / args.paper_run_id
    if not exp_root.exists():
        raise FileNotFoundError(f"Missing paper run root: {exp_root}")

    rows: list[dict[str, Any]] = []
    failed = 0

    for run_dir in sorted(p for p in exp_root.rglob("eval_raw.csv") if p.is_file()):
        run_root = run_dir.parent
        eval_manifest_path = run_root / "eval_manifest.json"
        if not eval_manifest_path.exists():
            rows.append(
                {
                    "run_dir": str(run_root),
                    "status": "missing_eval_manifest",
                }
            )
            failed += 1
            if args.strict:
                break
            continue

        try:
            eval_manifest = json.loads(eval_manifest_path.read_text())
            if not isinstance(eval_manifest, dict):
                raise ValueError("eval manifest must be dict")

            ruler_metrics = compute_ruler_metrics_from_eval_csv(run_dir)
            merged = merge_metrics(eval_manifest.get("metrics", {}), ruler_metrics)
            eval_manifest["metrics"] = merged
            _write_json(eval_manifest_path, eval_manifest)

            row = {
                "run_dir": str(run_root),
                "status": "ok",
                "ruler_accuracy_mean": merged.get("ruler_accuracy_mean"),
            }
            for key, value in merged.items():
                if str(key).startswith("ruler_by_length_"):
                    row[key] = value
            rows.append(row)

        except Exception as exc:
            rows.append({"run_dir": str(run_root), "status": "failed", "error": str(exc)})
            failed += 1
            if args.strict:
                break

    if args.summary_json is None:
        summary_json = (
            Path("./reports/paper") / args.paper_run_id / "eval" / "ruler_summary.json"
        ).resolve()
    else:
        summary_json = args.summary_json.expanduser().resolve()

    if args.summary_csv is None:
        summary_csv = (
            Path("./reports/paper") / args.paper_run_id / "eval" / "ruler_summary.csv"
        ).resolve()
    else:
        summary_csv = args.summary_csv.expanduser().resolve()

    payload = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "n_runs": len(rows),
        "n_failed": failed,
        "rows": rows,
    }
    _write_json(summary_json, payload)
    _write_csv(summary_csv, rows)

    print(f"Wrote RULER summary JSON: {summary_json}")
    print(f"Wrote RULER summary CSV:  {summary_csv}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
