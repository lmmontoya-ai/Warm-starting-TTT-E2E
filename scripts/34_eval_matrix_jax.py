#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from omegaconf import OmegaConf

from ttt.config import Config, register_configs
from ttt.jax_runtime.eval import run as jax_eval_run
from ttt.research.tracking import ensure_run_dir, write_eval_manifest
from ttt.research.types import EvalResult, utc_now_iso
from ttt.runtime import RunArtifacts


register_configs()


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part) for part in _parse_csv(raw)]


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _discover_runs(exp_dir: Path, paper_run_id: str, stage_filter: set[str], run_filter: set[str]) -> list[dict[str, Any]]:
    root = exp_dir / paper_run_id
    if not root.exists():
        raise FileNotFoundError(f"Missing paper run root: {root}")

    refs: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        run_manifest = _load_json(manifest_path)
        stage_id = str(run_manifest.get("stage_id", "")).strip()
        run_id = str(run_manifest.get("run_id", run_dir.name)).strip() or run_dir.name
        exp_name = str(run_manifest.get("exp_name", run_id)).strip() or run_id
        if stage_filter and stage_id not in stage_filter:
            continue
        if run_filter and run_id not in run_filter and exp_name not in run_filter:
            continue
        refs.append(
            {
                "stage_id": stage_id,
                "run_id": run_id,
                "exp_name": exp_name,
                "run_dir": run_dir,
            }
        )
    return refs


def _dataset_root(dataset_key: str, args: argparse.Namespace) -> Path:
    key = dataset_key.strip().lower()
    if key in {"books3", "books"}:
        return args.books_root.expanduser().resolve()
    if key in {"dclm_filter_8k", "dclm", "dclm8k"}:
        return args.dclm_root.expanduser().resolve()
    raise ValueError(f"Unsupported parity-eval dataset: {dataset_key}")


def _load_cfg(path: Path) -> Config:
    raw_cfg = OmegaConf.load(path)
    merged = OmegaConf.merge(OmegaConf.structured(Config), raw_cfg)
    cfg_obj = OmegaConf.to_object(merged)
    if not isinstance(cfg_obj, Config):
        raise TypeError(f"Expected Config object from {path}")
    return cfg_obj


def _artifacts(eval_dir: Path, cfg: Config) -> RunArtifacts:
    eval_dir.mkdir(parents=True, exist_ok=True)
    resolved = eval_dir / "resolved_config.yaml"
    unresolved = eval_dir / "unresolved_config.yaml"
    resolved.write_text(OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=True))
    unresolved.write_text(OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=False))
    return RunArtifacts(
        run_dir=eval_dir,
        resolved_config_path=resolved,
        unresolved_config_path=unresolved,
        metrics_path=eval_dir / "metrics.jsonl",
        events_path=eval_dir / "events.jsonl",
        run_manifest_path=eval_dir / "run_manifest.json",
        environment_manifest_path=eval_dir / "environment_manifest.json",
    )


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    last = rows[-1]
    if not isinstance(last, dict):
        raise ValueError(f"Expected dict row in {path}")
    return last


def _run_one_eval(*, checkpoint_root: Path, run_ref: dict[str, Any], dataset: str, context: int, args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_cfg(run_ref["run_dir"] / "resolved_config.yaml")
    cfg.training.runtime_mode = cfg.training.RuntimeMode.jax_eval
    cfg.training.load_part = cfg.training.LoadPart.params
    cfg.training.resume_checkpoint_format = "orbax"
    cfg.training.resume_checkpoint_path = str((checkpoint_root / args.exp_folder / run_ref["exp_name"]).resolve())
    cfg.training.eval_split = args.eval_split
    cfg.training.jax_eval_batches = int(args.eval_batches)
    cfg.training.eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size > 0 else int(cfg.training.eval_batch_size or cfg.training.global_batch_size)
    cfg.training.seq_length = int(context)
    cfg.training.dataset_path = str(_dataset_root(dataset, args))
    cfg.training.dataset_name = dataset
    cfg.training.run_id = str(run_ref["run_id"])
    cfg.training.stage_id = str(run_ref["stage_id"])
    cfg.training.paper_run_id = str(args.paper_run_id)
    cfg.model.seq_len = int(context)

    eval_dir = run_ref["run_dir"] / "jax_eval" / dataset / f"ctx_{context}"
    artifacts = _artifacts(eval_dir, cfg)

    import logging

    logger = logging.getLogger(f"jax_eval:{run_ref['stage_id']}:{run_ref['run_id']}:{dataset}:{context}")
    logger.setLevel(logging.INFO)
    jax_eval_run(cfg=cfg, artifacts=artifacts, logger=logger)

    summary = _read_last_jsonl(artifacts.metrics_path)
    nll_path = eval_dir / "per_position_nll.npy"
    return {
        "paper_run_id": args.paper_run_id,
        "stage_id": run_ref["stage_id"],
        "run_id": run_ref["run_id"],
        "exp_name": run_ref["exp_name"],
        "dataset": dataset,
        "context": int(context),
        "loss": summary.get("eval_loss"),
        "loss_ce": summary.get("eval_loss"),
        "tokens_per_second": summary.get("tokens_per_second"),
        "eval_wall_seconds": summary.get("elapsed_seconds"),
        "eval_batches": summary.get("eval_batches"),
        "eval_tokens": summary.get("eval_tokens"),
        "checkpoint_step": summary.get("step"),
        "per_position_nll_path": str(nll_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parity jax_eval over all ladder runs under experiments/<paper_run_id>/<stage>/<run>."
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", required=True)
    parser.add_argument("--stages", default="")
    parser.add_argument("--runs", default="")
    parser.add_argument("--eval-id", default="jax_parity")
    parser.add_argument("--contexts", default="8192,32768")
    parser.add_argument("--datasets", default="books3")
    parser.add_argument("--dclm-root", type=Path, default=Path("/tmp/phase1_token_data_dclm"))
    parser.add_argument("--books-root", type=Path, default=Path("/tmp/phase1_token_data_books"))
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    contexts = _parse_int_csv(args.contexts)
    datasets = _parse_csv(args.datasets)
    stage_filter = set(_parse_csv(args.stages))
    run_filter = set(_parse_csv(args.runs))

    run_refs = _discover_runs(exp_dir, args.paper_run_id, stage_filter, run_filter)
    if not run_refs:
        raise FileNotFoundError("No runs found for parity eval.")

    repo_root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    failed = 0

    for run_ref in run_refs:
        run_root = ensure_run_dir(
            exp_dir=exp_dir,
            paper_run_id=args.paper_run_id,
            stage_id=run_ref["stage_id"],
            run_id=run_ref["run_id"],
        )
        raw_json = run_root / "eval_parity_raw.json"
        raw_csv = run_root / "eval_parity_raw.csv"
        created = utc_now_iso()

        if args.dry_run:
            result = EvalResult(
                run_id=run_ref["run_id"],
                stage_id=run_ref["stage_id"],
                eval_id=args.eval_id,
                status="dry_run",
                created_at_utc=created,
                finished_at_utc=created,
                eval_manifest_path=str(run_root / "eval_manifest.json"),
                raw_json_path=str(raw_json),
                raw_csv_path=str(raw_csv),
                metrics={},
                artifacts={},
            )
            write_eval_manifest(run_root / "eval_manifest.json", result, repo_root=repo_root)
            continue

        try:
            per_run_rows: list[dict[str, Any]] = []
            for dataset in datasets:
                for context in contexts:
                    per_run_rows.append(
                        _run_one_eval(
                            checkpoint_root=checkpoint_root,
                            run_ref=run_ref,
                            dataset=dataset,
                            context=context,
                            args=args,
                        )
                    )

            _write_json(raw_json, {"rows": per_run_rows})
            _write_csv(raw_csv, per_run_rows)

            metrics = {
                "loss_mean": mean(float(row["loss"]) for row in per_run_rows if row.get("loss") is not None),
                "loss_ce_mean": mean(float(row["loss_ce"]) for row in per_run_rows if row.get("loss_ce") is not None),
                "tokens_per_second_mean": mean(float(row["tokens_per_second"]) for row in per_run_rows if row.get("tokens_per_second") is not None),
                "eval_wall_seconds": sum(float(row["eval_wall_seconds"]) for row in per_run_rows if row.get("eval_wall_seconds") is not None),
            }
            longest = max(per_run_rows, key=lambda row: (int(row["context"]), str(row["dataset"])))
            result = EvalResult(
                run_id=run_ref["run_id"],
                stage_id=run_ref["stage_id"],
                eval_id=args.eval_id,
                status="succeeded",
                created_at_utc=created,
                finished_at_utc=utc_now_iso(),
                eval_manifest_path=str(run_root / "eval_manifest.json"),
                raw_json_path=str(raw_json),
                raw_csv_path=str(raw_csv),
                metrics=metrics,
                artifacts={
                    "per_position_nll_path": str(longest["per_position_nll_path"]),
                    "raw_eval_csv": str(raw_csv),
                    "raw_eval_json": str(raw_json),
                },
            )
            write_eval_manifest(run_root / "eval_manifest.json", result, repo_root=repo_root)
            rows.append(
                {
                    "paper_run_id": args.paper_run_id,
                    "stage_id": run_ref["stage_id"],
                    "run_id": run_ref["run_id"],
                    "status": result.status,
                    **metrics,
                    "raw_csv_path": str(raw_csv),
                    "raw_json_path": str(raw_json),
                    "per_position_nll_path": str(longest["per_position_nll_path"]),
                }
            )
        except Exception as exc:
            failed += 1
            result = EvalResult(
                run_id=run_ref["run_id"],
                stage_id=run_ref["stage_id"],
                eval_id=args.eval_id,
                status="failed",
                created_at_utc=created,
                finished_at_utc=utc_now_iso(),
                eval_manifest_path=str(run_root / "eval_manifest.json"),
                raw_json_path=str(raw_json),
                raw_csv_path=str(raw_csv),
                metrics={},
                artifacts={},
                error_message=str(exc),
            )
            write_eval_manifest(run_root / "eval_manifest.json", result, repo_root=repo_root)
            rows.append(
                {
                    "paper_run_id": args.paper_run_id,
                    "stage_id": run_ref["stage_id"],
                    "run_id": run_ref["run_id"],
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            if args.strict:
                break

    summary_json = args.summary_json.expanduser().resolve() if args.summary_json else (Path("./reports/paper") / args.paper_run_id / "eval" / "eval_parity_summary.json").resolve()
    summary_csv = args.summary_csv.expanduser().resolve() if args.summary_csv else (Path("./reports/paper") / args.paper_run_id / "eval" / "eval_parity_summary.csv").resolve()
    _write_json(
        summary_json,
        {
            "schema_version": "1.0",
            "paper_run_id": args.paper_run_id,
            "eval_id": args.eval_id,
            "contexts": contexts,
            "datasets": datasets,
            "n_runs": len(rows),
            "n_failed": failed,
            "rows": rows,
        },
    )
    _write_csv(summary_csv, rows)
    print(f"Wrote parity eval summary JSON: {summary_json}")
    print(f"Wrote parity eval summary CSV:  {summary_csv}")
    return 1 if failed and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
