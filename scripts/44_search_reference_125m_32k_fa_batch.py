#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BATCH_SIZES = [32, 24, 16, 8]
OOM_MARKERS = [
    "RESOURCE_EXHAUSTED",
    "Out of memory",
    "RESOURCE EXHAUSTED",
    "Can't reduce memory use below",
]
INVALID_SHARDING_MARKERS = [
    "should be divisible by 8",
    "should be divisible by",
]


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_batch_sizes(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("No batch sizes provided.")
    for value in values:
        if value <= 0:
            raise ValueError(f"Invalid batch size: {value}")
    return values


def _filter_batch_sizes(values: list[int], *, data_parallel_width: int) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []
    for value in values:
        if value % data_parallel_width == 0:
            valid.append(value)
        else:
            invalid.append(value)
    return valid, invalid


def _read_latest_metrics(experiment_dir: Path) -> dict:
    metrics_path = experiment_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def _load_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _classify_run(rc: int, log_text: str, metrics: dict, steps: int) -> tuple[str, bool]:
    oom = any(marker in log_text for marker in OOM_MARKERS)
    invalid_sharding = any(marker in log_text for marker in INVALID_SHARDING_MARKERS)
    latest_step = int(metrics.get("step", -1) or -1)
    loss_ce = metrics.get("loss_ce")
    loss_ok = isinstance(loss_ce, (int, float)) and math.isfinite(float(loss_ce)) and float(loss_ce) > 0.0
    completed_steps = latest_step >= max(0, steps - 1)
    passed = (rc == 0) and (not oom) and completed_steps and loss_ok
    if passed:
        return "passed", True
    if invalid_sharding:
        return "invalid_sharding", False
    if oom:
        return "oom", False
    if rc != 0:
        return "failed_rc", False
    if not completed_steps:
        return "short_run", False
    return "invalid_metrics", False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reference-first batch search for the 125M 32K FA extension. Runs the "
            "read-only reference smoke in descending batch order and selects the highest "
            "passing batch size under Protocol R."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--resume-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, default=4560)
    parser.add_argument("--batch-sizes", default="32,24,16,8")
    parser.add_argument("--data-parallel-width", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--exp-folder", default="reference_protocol_r_search")
    parser.add_argument("--exp-name-prefix", default="ext-125m-fa-32K-ref")
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/protocol_r"))
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--wandb-key", default=os.environ.get("WANDB_API_KEY", ""))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "none"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "none"))
    parser.add_argument("--continue-after-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    artifact_dir = (
        args.artifact_dir.expanduser().resolve()
        if args.artifact_dir is not None
        else (args.artifact_root.expanduser().resolve() / f"reference_batch_search_{_utc_slug()}")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    requested_batch_sizes = _parse_batch_sizes(args.batch_sizes)
    batch_sizes, skipped_batch_sizes = _filter_batch_sizes(
        requested_batch_sizes,
        data_parallel_width=args.data_parallel_width,
    )
    if not batch_sizes:
        raise SystemExit("No valid batch sizes remain after data-parallel divisibility filtering.")

    rows: list[dict[str, object]] = []
    selected_batch_size: int | None = None

    for batch_size in skipped_batch_sizes:
        row = {
            "batch_size": batch_size,
            "returncode": None,
            "status": "skipped_invalid_sharding",
            "passed": False,
            "exp_name": f"{args.exp_name_prefix}-b{batch_size}",
            "experiment_dir": "",
            "log_path": "",
            "latest_metrics": {},
        }
        rows.append(row)

    for batch_size in batch_sizes:
        exp_name = f"{args.exp_name_prefix}-b{batch_size}"
        log_path = artifact_dir / f"b{batch_size}" / "reference_125m_32k_fa_smoke.log"
        cmd = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/41_run_reference_125m_32k_fa_smoke.py",
            "--repo-root",
            str(repo_root),
            "--books-root",
            str(args.books_root.expanduser().resolve()),
            "--checkpoint-root",
            str(args.checkpoint_root.expanduser().resolve()),
            "--resume-checkpoint-dir",
            str(args.resume_checkpoint_dir.expanduser().resolve()),
            "--resume-step",
            str(args.resume_step),
            "--steps",
            str(args.steps),
            "--save-milestone-freq",
            str(args.save_milestone_freq),
            "--python-version",
            args.python_version,
            "--global-batch-size",
            str(batch_size),
            "--exp-dir",
            str(args.exp_dir.expanduser().resolve()),
            "--exp-folder",
            args.exp_folder,
            "--exp-name",
            exp_name,
            "--log-path",
            str(log_path),
            "--wandb-key",
            args.wandb_key,
            "--wandb-entity",
            args.wandb_entity,
            "--wandb-project",
            args.wandb_project,
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        redacted_cmd = []
        skip_value = False
        for index, part in enumerate(cmd):
            if skip_value:
                redacted_cmd.append("<redacted>")
                skip_value = False
                continue
            if part == "--wandb-key":
                redacted_cmd.append(part)
                skip_value = True
                continue
            redacted_cmd.append(part)

        print("$ " + shlex.join(redacted_cmd), flush=True)
        rc = 0 if args.dry_run else subprocess.run(cmd, check=False, cwd=repo_root).returncode

        experiment_dir = args.exp_dir.expanduser().resolve() / args.exp_folder / exp_name
        metrics = _read_latest_metrics(experiment_dir)
        log_text = _load_log_text(log_path)
        status, passed = _classify_run(rc, log_text, metrics, steps=args.steps)
        row = {
            "batch_size": batch_size,
            "returncode": rc,
            "status": status,
            "passed": passed,
            "exp_name": exp_name,
            "experiment_dir": str(experiment_dir),
            "log_path": str(log_path),
            "latest_metrics": metrics,
        }
        rows.append(row)
        _write_json(artifact_dir / f"b{batch_size}" / "run_summary.json", row)

        if passed and selected_batch_size is None:
            selected_batch_size = batch_size
            if not args.continue_after_pass:
                break

    payload = {
        "schema_version": "1.0",
        "search_order": requested_batch_sizes,
        "executed_batch_sizes": batch_sizes,
        "skipped_batch_sizes": skipped_batch_sizes,
        "data_parallel_width": args.data_parallel_width,
        "selected_batch_size": selected_batch_size,
        "rows": rows,
    }
    _write_json(artifact_dir / "reference_protocol_r_batch_search.json", payload)

    report_lines = [
        "# Reference Protocol R Batch Search",
        "",
        f"- Selected batch size: `{selected_batch_size}`" if selected_batch_size is not None else "- Selected batch size: none",
        "",
        "| Batch | Status | RC | loss_ce | step |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row.get("latest_metrics", {}) or {}
        report_lines.append(
            f"| {row['batch_size']} | {row['status']} | {row['returncode']} | "
            f"{metrics.get('loss_ce', '—')} | {metrics.get('step', '—')} |"
        )
    (artifact_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote reference batch search summary: {artifact_dir / 'reference_protocol_r_batch_search.json'}")
    if selected_batch_size is None and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
