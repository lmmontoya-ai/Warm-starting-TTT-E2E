"""Evaluation runner helpers for warm-start research experiments."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

from .tracking import ensure_run_dir, write_eval_manifest
from .types import EvalResult, utc_now_iso



def _safe_float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None



def _summarize_eval_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}

    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return {}

    loss_vals: list[float] = []
    tps_vals: list[float] = []
    niah_vals: list[float] = []

    for row in rows:
        loss = _safe_float(row.get("loss"))
        if loss is not None:
            loss_vals.append(loss)
        tps = _safe_float(row.get("tokens_per_second"))
        if tps is not None:
            tps_vals.append(tps)
        niah = _safe_float(row.get("niah_accuracy"))
        if niah is not None:
            niah_vals.append(niah)

    metrics: dict[str, float] = {}
    if loss_vals:
        metrics["loss_mean"] = sum(loss_vals) / len(loss_vals)
    if tps_vals:
        metrics["tokens_per_second_mean"] = sum(tps_vals) / len(tps_vals)
    if niah_vals:
        metrics["niah_accuracy_mean"] = sum(niah_vals) / len(niah_vals)
    return metrics



def run_eval_command(
    *,
    eval_id: str,
    paper_run_id: str,
    stage_id: str,
    run_id: str,
    exp_dir: Path,
    command: list[str],
    raw_json_path: Path,
    raw_csv_path: Path,
    repo_root: Path,
    dry_run: bool = False,
) -> EvalResult:
    run_dir = ensure_run_dir(
        exp_dir=exp_dir,
        paper_run_id=paper_run_id,
        stage_id=stage_id,
        run_id=run_id,
    )
    started = time.perf_counter()
    created = utc_now_iso()

    if dry_run:
        result = EvalResult(
            run_id=run_id,
            stage_id=stage_id,
            eval_id=eval_id,
            status="dry_run",
            created_at_utc=created,
            finished_at_utc=created,
            eval_manifest_path=str(run_dir / "eval_manifest.json"),
            raw_json_path=str(raw_json_path),
            raw_csv_path=str(raw_csv_path),
            metrics={},
        )
        write_eval_manifest(run_dir / "eval_manifest.json", result, repo_root=repo_root)
        return result

    completed = subprocess.run(command, check=False)
    finished = utc_now_iso()

    if completed.returncode != 0:
        result = EvalResult(
            run_id=run_id,
            stage_id=stage_id,
            eval_id=eval_id,
            status="failed",
            created_at_utc=created,
            finished_at_utc=finished,
            eval_manifest_path=str(run_dir / "eval_manifest.json"),
            raw_json_path=str(raw_json_path),
            raw_csv_path=str(raw_csv_path),
            metrics={},
            error_message=f"command exited with rc={completed.returncode}",
        )
        write_eval_manifest(run_dir / "eval_manifest.json", result, repo_root=repo_root)
        return result

    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = _summarize_eval_csv(raw_csv_path)
    metrics["eval_wall_seconds"] = elapsed

    result = EvalResult(
        run_id=run_id,
        stage_id=stage_id,
        eval_id=eval_id,
        status="succeeded",
        created_at_utc=created,
        finished_at_utc=finished,
        eval_manifest_path=str(run_dir / "eval_manifest.json"),
        raw_json_path=str(raw_json_path),
        raw_csv_path=str(raw_csv_path),
        metrics=metrics,
    )
    write_eval_manifest(run_dir / "eval_manifest.json", result, repo_root=repo_root)
    return result



def load_eval_result(path: Path) -> EvalResult:
    payload = json.loads(path.read_text())
    return EvalResult(
        schema_version=str(payload.get("schema_version", "1.0")),
        run_id=str(payload.get("run_id", "")),
        stage_id=str(payload.get("stage_id", "")),
        eval_id=str(payload.get("eval_id", "")),
        status=str(payload.get("status", "unknown")),
        created_at_utc=str(payload.get("created_at_utc", "")),
        finished_at_utc=str(payload.get("finished_at_utc", "")),
        eval_manifest_path=str(payload.get("eval_manifest_path", "")),
        raw_json_path=str(payload.get("raw_json_path", "")),
        raw_csv_path=str(payload.get("raw_csv_path", "")),
        metrics={str(k): float(v) for k, v in payload.get("metrics", {}).items()},
        error_message=str(payload.get("error_message", "")),
    )
