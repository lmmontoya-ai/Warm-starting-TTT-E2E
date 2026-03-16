#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ttt.research.budget import estimate_gpu_hours_from_wall
from ttt.research.orchestrator import _read_jsonl_rows
from ttt.research.orchestrator import observed_tokens_from_runtime_artifacts
from ttt.research.types import utc_now_iso


def _parse_csv(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _nonnegative_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0:
        return None
    return parsed


def _observed_runtime_from_events(
    *,
    events_path: Path,
    fallback_wall_seconds: float | None,
    fallback_num_gpus: int | None,
) -> tuple[float | None, int | None]:
    wall_seconds = fallback_wall_seconds
    num_gpus = fallback_num_gpus

    for row in _read_jsonl_rows(events_path):
        event = str(row.get("event", "")).strip()
        if event == "run_started":
            device_info = row.get("device_info", {})
            if isinstance(device_info, dict):
                parsed_gpus = _positive_int(device_info.get("device_count"))
                if parsed_gpus is not None:
                    num_gpus = parsed_gpus
        elif event == "run_finished":
            parsed_wall = _nonnegative_float(row.get("wall_seconds"))
            if parsed_wall is None:
                parsed_wall = _nonnegative_float(row.get("elapsed_seconds"))
            if parsed_wall is not None:
                wall_seconds = parsed_wall

    return wall_seconds, num_gpus


def _discover_run_dirs(
    *,
    exp_dir: Path,
    paper_run_id: str,
    stage_filter: set[str],
    run_filter: set[str],
) -> list[Path]:
    root = exp_dir / paper_run_id
    if not root.exists():
        raise FileNotFoundError(f"Missing paper run root: {root}")

    run_dirs: list[Path] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        run_manifest = _load_json(manifest_path)
        stage_id = str(run_manifest.get("stage_id", "")).strip() or run_dir.parent.name
        run_id = str(run_manifest.get("run_id", "")).strip() or run_dir.name
        exp_name = str(run_manifest.get("exp_name", "")).strip() or run_id
        if stage_filter and stage_id not in stage_filter:
            continue
        if run_filter and run_id not in run_filter and exp_name not in run_filter:
            continue
        run_dirs.append(run_dir)
    return run_dirs


def repair_run_dir(*, run_dir: Path, dry_run: bool) -> dict[str, Any]:
    run_manifest_path = run_dir / "run_manifest.json"
    run_result_path = run_dir / "run_result.json"
    budget_manifest_path = run_dir / "budget_manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    events_path = run_dir / "events.jsonl"

    run_manifest = _load_json(run_manifest_path)
    run_result = _load_json(run_result_path)
    budget_manifest = _load_json(budget_manifest_path) if budget_manifest_path.exists() else {}

    stage_id = str(run_manifest.get("stage_id", "")).strip() or run_dir.parent.name
    run_id = str(run_manifest.get("run_id", "")).strip() or run_dir.name
    status = str(run_result.get("status", "")).strip()

    if status != "succeeded":
        return {
            "stage_id": stage_id,
            "run_id": run_id,
            "status": "skipped_non_success",
        }

    previous_tokens = int(run_result.get("tokens_seen", 0) or 0)
    observed_tokens = observed_tokens_from_runtime_artifacts(
        metrics_path=metrics_path,
        events_path=events_path,
        fallback=previous_tokens,
    )
    previous_budget_tokens = int(budget_manifest.get("tokens_observed", 0) or 0)
    usage = budget_manifest.get("usage")
    if not isinstance(usage, dict):
        usage = {}
        budget_manifest["usage"] = usage
    previous_usage_tokens = int(usage.get("tokens_observed", 0) or 0)
    previous_wall_seconds = _nonnegative_float(run_result.get("wall_seconds"))
    previous_gpu_hours = _nonnegative_float(run_result.get("gpu_hours"))
    previous_usage_gpu_hours = _nonnegative_float(usage.get("gpu_hours_observed"))
    observed_wall_seconds, observed_num_gpus = _observed_runtime_from_events(
        events_path=events_path,
        fallback_wall_seconds=previous_wall_seconds,
        fallback_num_gpus=None,
    )
    observed_gpu_hours = previous_gpu_hours
    if observed_wall_seconds is not None:
        observed_gpu_hours = estimate_gpu_hours_from_wall(
            wall_seconds=observed_wall_seconds,
            num_gpus=observed_num_gpus,
        )
    local_run_dir = str(run_dir.resolve())
    local_metrics_path = str(metrics_path.resolve())
    local_events_path = str(events_path.resolve())
    changed = (
        observed_tokens != previous_tokens
        or previous_budget_tokens != observed_tokens
        or previous_usage_tokens != observed_tokens
        or str(run_result.get("stage_id", "")) != stage_id
        or str(run_result.get("run_id", "")) != run_id
        or str(run_result.get("run_dir", "")) != local_run_dir
        or str(run_result.get("metrics_path", "")) != local_metrics_path
        or str(run_result.get("events_path", "")) != local_events_path
        or observed_wall_seconds != previous_wall_seconds
        or observed_gpu_hours != previous_gpu_hours
        or observed_gpu_hours != previous_usage_gpu_hours
    )

    if not dry_run:
        run_result["stage_id"] = stage_id
        run_result["run_id"] = run_id
        run_result["tokens_seen"] = observed_tokens
        run_result["run_dir"] = local_run_dir
        run_result["metrics_path"] = local_metrics_path
        run_result["events_path"] = local_events_path
        if observed_wall_seconds is not None:
            run_result["wall_seconds"] = observed_wall_seconds
        if observed_gpu_hours is not None:
            run_result["gpu_hours"] = observed_gpu_hours
        _write_json(run_result_path, run_result)

        if budget_manifest:
            budget_manifest["tokens_observed"] = observed_tokens
            usage["tokens_observed"] = observed_tokens
            if observed_gpu_hours is not None:
                usage["gpu_hours_observed"] = observed_gpu_hours
            _write_json(budget_manifest_path, budget_manifest)

    return {
        "stage_id": stage_id,
        "run_id": run_id,
        "status": "updated" if changed else "unchanged",
        "tokens_seen_before": previous_tokens,
        "tokens_seen_after": observed_tokens,
        "budget_tokens_observed_before": previous_budget_tokens,
        "budget_tokens_observed_after": observed_tokens,
        "usage_tokens_observed_before": previous_usage_tokens,
        "usage_tokens_observed_after": observed_tokens,
        "wall_seconds_before": previous_wall_seconds,
        "wall_seconds_after": observed_wall_seconds,
        "gpu_hours_before": previous_gpu_hours,
        "gpu_hours_after": observed_gpu_hours,
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill run_result.json and budget_manifest.json for successful canonical "
            "runs using observed runtime tokens from metrics/events."
        )
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--stages", default="")
    parser.add_argument("--runs", default="")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    stage_filter = _parse_csv(args.stages)
    run_filter = _parse_csv(args.runs)

    if args.summary_json is None:
        summary_json = (
            Path("./reports/paper")
            / args.paper_run_id
            / "metadata"
            / "run_metadata_backfill_summary.json"
        ).resolve()
    else:
        summary_json = args.summary_json.expanduser().resolve()

    run_dirs = _discover_run_dirs(
        exp_dir=exp_dir,
        paper_run_id=args.paper_run_id,
        stage_filter=stage_filter,
        run_filter=run_filter,
    )

    rows = [repair_run_dir(run_dir=run_dir, dry_run=args.dry_run) for run_dir in run_dirs]
    summary = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "paper_run_id": args.paper_run_id,
        "exp_dir": str(exp_dir),
        "dry_run": args.dry_run,
        "n_runs": len(rows),
        "n_updated": sum(1 for row in rows if row.get("status") == "updated"),
        "n_unchanged": sum(1 for row in rows if row.get("status") == "unchanged"),
        "n_skipped_non_success": sum(1 for row in rows if row.get("status") == "skipped_non_success"),
        "rows": rows,
    }
    _write_json(summary_json, summary)

    print(f"Wrote summary: {summary_json}")
    for row in rows:
        stage_id = row.get("stage_id", "")
        run_id = row.get("run_id", "")
        status = row.get("status", "")
        before = row.get("tokens_seen_before", "")
        after = row.get("tokens_seen_after", "")
        print(f"{stage_id}/{run_id}: {status} tokens_seen {before} -> {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
