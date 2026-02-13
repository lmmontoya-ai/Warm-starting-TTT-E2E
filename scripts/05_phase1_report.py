#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class RunSummary:
    exp_folder: str
    exp_name: str
    run_dir: str
    runtime_mode: str
    train_mode: str
    seq_length: int
    global_batch_size: int
    steps_logged: int
    first_step: int | None
    last_step: int | None
    tokens_seen: int
    avg_loss: float | None
    final_loss: float | None
    checkpoint_dir: str
    checkpoint_step: int | None
    checkpoint_path: str
    elapsed_seconds: float | None
    tokens_per_second: float | None
    load_part: str
    resume_exp_name: str
    restore_status: str
    init_source: str
    external_model_id: str
    adapter_recipe: str


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _collect_run_dirs(exp_dir: Path, filters: set[str]) -> list[Path]:
    run_dirs = []
    for cfg_path in exp_dir.rglob("phase1_resolved_config.yaml"):
        run_dir = cfg_path.parent
        if filters and run_dir.parent.name not in filters:
            continue
        run_dirs.append(run_dir)
    return sorted(set(run_dirs), key=lambda p: str(p))


def _summarize_metrics(
    records: list[dict[str, Any]], seq_length: int, global_batch_size: int
) -> tuple[int, int | None, int | None, int, float | None, float | None, str]:
    if not records:
        return 0, None, None, 0, None, None, "not_attempted"

    default_tokens_per_step = max(seq_length, 0) * max(global_batch_size, 0)

    steps_logged = len(records)
    first_step = _safe_int(records[0].get("step"), default=0)
    last_step = _safe_int(records[-1].get("step"), default=0)

    tokens_seen = 0
    losses: list[float] = []
    restore_status = "unknown"

    first_restore = records[0].get("restore")
    if isinstance(first_restore, dict):
        raw_status = first_restore.get("restore_status")
        if isinstance(raw_status, str):
            restore_status = raw_status

    for record in records:
        tokens = record.get("tokens_in_batch")
        if tokens is None:
            tokens_seen += default_tokens_per_step
        else:
            tokens_seen += max(_safe_int(tokens), 0)

        loss = _safe_float(record.get("loss"))
        if loss is not None:
            losses.append(loss)

    avg_loss = sum(losses) / len(losses) if losses else None
    final_loss = losses[-1] if losses else None

    return (
        steps_logged,
        first_step,
        last_step,
        tokens_seen,
        avg_loss,
        final_loss,
        restore_status,
    )


def _read_checkpoint(checkpoint_dir: Path) -> tuple[int | None, str, float | None]:
    latest = _read_json(checkpoint_dir / "latest.json")
    if latest is None:
        return None, "", None

    step = latest.get("step")
    ckpt_step = _safe_int(step, default=-1)
    if ckpt_step < 0:
        return None, "", None

    rel_path = latest.get("path")
    if not isinstance(rel_path, str):
        rel_path = f"phase1_ckpt_step_{ckpt_step:08d}.json"

    ckpt_path = checkpoint_dir / rel_path
    ckpt_payload = _read_json(ckpt_path)

    elapsed_seconds = None
    if ckpt_payload is not None:
        elapsed_seconds = _safe_float(ckpt_payload.get("elapsed_seconds"))

    return ckpt_step, str(ckpt_path), elapsed_seconds


def _load_run_summary(run_dir: Path) -> RunSummary | None:
    cfg_path = run_dir / "phase1_resolved_config.yaml"
    if not cfg_path.exists():
        return None

    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(cfg, dict):
        return None

    training = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    checkpoint = cfg.get("checkpoint", {}) if isinstance(cfg.get("checkpoint"), dict) else {}

    exp_folder = str(training.get("exp_folder", run_dir.parent.name))
    exp_name = str(training.get("exp_name", run_dir.name))
    runtime_mode = str(training.get("runtime_mode", "unknown"))
    train_mode = str(training.get("train_mode", "unknown"))
    seq_length = _safe_int(training.get("seq_length"), default=0)
    global_batch_size = _safe_int(training.get("global_batch_size"), default=0)
    load_part = str(training.get("load_part", "none"))
    resume_exp_name = str(training.get("resume_exp_name", ""))
    init_source = str(training.get("init_source", "scratch"))
    external_model_id = str(training.get("external_model_id", ""))
    adapter_recipe = str(training.get("adapter_recipe", "none"))

    metrics_records = _read_jsonl(run_dir / "phase1_metrics.jsonl")
    (
        steps_logged,
        first_step,
        last_step,
        tokens_seen,
        avg_loss,
        final_loss,
        restore_status,
    ) = _summarize_metrics(metrics_records, seq_length, global_batch_size)

    if load_part == "none" and restore_status == "unknown":
        restore_status = "not_attempted"

    checkpoint_dir_raw = checkpoint.get("checkpoint_dir", "")
    checkpoint_dir = Path(str(checkpoint_dir_raw)).expanduser().resolve() if checkpoint_dir_raw else Path("")
    ckpt_step = None
    ckpt_path = ""
    elapsed_seconds = None

    if checkpoint_dir_raw:
        ckpt_step, ckpt_path, elapsed_seconds = _read_checkpoint(checkpoint_dir)

    tokens_per_second = None
    if elapsed_seconds is not None and elapsed_seconds > 0:
        tokens_per_second = tokens_seen / elapsed_seconds

    return RunSummary(
        exp_folder=exp_folder,
        exp_name=exp_name,
        run_dir=str(run_dir.resolve()),
        runtime_mode=runtime_mode,
        train_mode=train_mode,
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        steps_logged=steps_logged,
        first_step=first_step,
        last_step=last_step,
        tokens_seen=tokens_seen,
        avg_loss=avg_loss,
        final_loss=final_loss,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir_raw else "",
        checkpoint_step=ckpt_step,
        checkpoint_path=ckpt_path,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_per_second,
        load_part=load_part,
        resume_exp_name=resume_exp_name,
        restore_status=restore_status,
        init_source=init_source,
        external_model_id=external_model_id,
        adapter_recipe=adapter_recipe,
    )


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _render_table(summaries: list[RunSummary]) -> str:
    columns = [
        ("exp_folder", "Folder"),
        ("exp_name", "Experiment"),
        ("runtime_mode", "Runtime"),
        ("train_mode", "Train"),
        ("seq_length", "Seq"),
        ("global_batch_size", "GBS"),
        ("steps_logged", "Steps"),
        ("last_step", "Last"),
        ("tokens_seen", "Tokens"),
        ("final_loss", "FinalLoss"),
        ("elapsed_seconds", "WallSec"),
        ("tokens_per_second", "Tok/Sec"),
        ("checkpoint_step", "Ckpt"),
        ("load_part", "Load"),
        ("resume_exp_name", "ResumeFrom"),
        ("restore_status", "Restore"),
        ("init_source", "Init"),
        ("external_model_id", "ExternalModel"),
        ("adapter_recipe", "Adapter"),
    ]

    rows: list[list[str]] = []
    for summary in summaries:
        row: list[str] = []
        for key, _ in columns:
            value = getattr(summary, key)
            if key in {"seq_length", "global_batch_size", "steps_logged", "last_step", "tokens_seen", "checkpoint_step"}:
                row.append(_fmt_int(value if isinstance(value, int) else None))
            elif key in {"final_loss"}:
                row.append(_fmt_float(value if isinstance(value, float) else None, digits=5))
            elif key in {"elapsed_seconds"}:
                row.append(_fmt_float(value if isinstance(value, float) else None, digits=2))
            elif key in {"tokens_per_second"}:
                row.append(_fmt_float(value if isinstance(value, float) else None, digits=1))
            else:
                row.append(str(value) if value else "-")
        rows.append(row)

    headers = [label for _, label in columns]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(widths)))

    body_lines = [
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    ]

    return "\n".join([header_line, sep_line, *body_lines])


def _summaries_to_json(summaries: list[RunSummary]) -> str:
    return json.dumps([asdict(s) for s in summaries], indent=2, sort_keys=True)


def _write_csv(path: Path, summaries: list[RunSummary]) -> None:
    if not summaries:
        path.write_text("")
        return

    rows = [asdict(s) for s in summaries]
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize phase-1 experiment runs with tokens, wall-clock, "
            "checkpoint lineage, and restore metadata."
        )
    )
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument(
        "--exp-folder",
        action="append",
        default=[],
        help="Filter to one or more experiment folders (repeatable).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="CSV output path when --format=csv (default: ./phase1_report.csv).",
    )
    args = parser.parse_args()

    exp_dir = args.exp_dir.expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    filters = set(args.exp_folder)
    run_dirs = _collect_run_dirs(exp_dir, filters)

    summaries: list[RunSummary] = []
    for run_dir in run_dirs:
        summary = _load_run_summary(run_dir)
        if summary is not None:
            summaries.append(summary)

    summaries.sort(key=lambda s: (s.exp_folder, s.exp_name))

    if args.format == "json":
        print(_summaries_to_json(summaries))
        return 0

    if args.format == "csv":
        csv_path = args.csv_out.expanduser().resolve() if args.csv_out else Path("./phase1_report.csv").resolve()
        _write_csv(csv_path, summaries)
        print(f"Wrote {len(summaries)} rows to {csv_path}")
        return 0

    if not summaries:
        print(f"No phase-1 runs found in {exp_dir}")
        return 0

    print(_render_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
