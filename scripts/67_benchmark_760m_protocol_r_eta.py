#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ttt.research.author_checkpoints import author_seed_checkpoint_ref, load_env_file
from ttt.research.eta_760m import estimate_stage_training_wall_seconds, read_metrics_rows
from ttt.research.lineage import resolve_checkpoint_ref
from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.protocol_760m import (
    ALL_760M_STAGE_IDS,
    AUTHOR_SEED_SOURCES_760M,
    PAPER_STAGE_LABELS_760M,
    build_protocol_r_760m_manifest,
    build_protocol_r_760m_stage_map,
)
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec, CheckpointRef, utc_now_iso


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics.jsonl"


def _run_id(stage_exp_name: str, steps: int) -> str:
    return f"{stage_exp_name}-eta-bench{steps}"


def _read_last_loss(metrics_rows: list[dict[str, Any]]) -> float | None:
    if not metrics_rows:
        return None
    value = metrics_rows[-1].get("loss_ce")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run short real 760M revised-protocol benchmarks and estimate stage ETAs "
            "from post-warmup step times."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/author_checkpoints"))
    parser.add_argument("--paper-run-id", default="protocol_r_760m_eta_bench_v1")
    parser.add_argument("--exp-folder", default="protocol_r_760m_eta_bench_v1")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--revised-global-batch-size", type=int, default=8)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument(
        "--stage-ids",
        nargs="+",
        choices=ALL_760M_STAGE_IDS,
        default=list(ALL_760M_STAGE_IDS),
        help="Subset of 760M stages to benchmark.",
    )
    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    load_env_file(repo_root / ".env")

    registry = load_registry(args.registry.expanduser().resolve())
    stage_map = registry.stage_map()
    protocol_stage_map = build_protocol_r_760m_stage_map(
        revised_global_batch_size=args.revised_global_batch_size,
    )

    protocol_manifest = build_protocol_r_760m_manifest(
        paper_run_id=args.paper_run_id,
        exp_folder=args.exp_folder,
        revised_global_batch_size=args.revised_global_batch_size,
        save_milestone_freq=args.save_milestone_freq,
        seed=0,
    )
    protocol_out = (
        repo_root / "reports" / "paper" / args.paper_run_id / "launch" / "protocol_manifest.json"
    ).resolve()
    _write_json(protocol_out, protocol_manifest)

    budget = BudgetSpec(
        budget_id="760m_eta_bench",
        pretrain_steps=0,
        adapt_steps=args.steps,
        ext_steps=args.steps,
        seed=0,
    )
    eval_spec = registry.eval_profiles["default_longctx"]
    opts = OrchestratorOptions(
        deploy="interactive",
        runtime_mode="jax_train",
        exp_dir=args.exp_dir.expanduser().resolve(),
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        profile_root=args.profile_root.expanduser().resolve(),
        dclm_root=args.dclm_root.expanduser().resolve(),
        books_root=args.books_root.expanduser().resolve(),
        exp_folder=args.exp_folder,
        wandb_entity="none",
        wandb_project="none",
        wandb_key="env",
        global_batch_size=args.revised_global_batch_size,
        ext_global_batch_size=args.revised_global_batch_size,
        save_milestone_freq=args.save_milestone_freq,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.allow_missing_fingerprints and not args.dry_run),
    )

    rows: list[dict[str, Any]] = []
    s2_adapt_parent_ref = None
    s2_adapt_resume_root = None

    for stage_id in args.stage_ids:
        stage = stage_map[stage_id]
        spec = protocol_stage_map[stage_id]
        run_id = _run_id(stage.exp_name, args.steps)

        parent_refs_override = None
        explicit_resume_checkpoint_path = None
        explicit_resume_checkpoint_format = None
        extra_overrides = ["backend.distributed=false", "training.log_wandb=false"]
        seed_source = ""

        if stage_id in AUTHOR_SEED_SOURCES_760M:
            seed_source = AUTHOR_SEED_SOURCES_760M[stage_id]
            parent_refs_override = [author_seed_checkpoint_ref(args.artifact_root.expanduser().resolve(), seed_source)]
            explicit_resume_checkpoint_path = args.artifact_root.expanduser().resolve() / seed_source
            explicit_resume_checkpoint_format = "orbax"
            extra_overrides.append("training.load_part=params")
        elif stage_id == "S2":
            if s2_adapt_parent_ref is None or s2_adapt_resume_root is None:
                raise RuntimeError("S2 benchmark requires a completed S2_ADAPT benchmark parent.")
            parent_refs_override = [s2_adapt_parent_ref]
            explicit_resume_checkpoint_path = s2_adapt_resume_root
            explicit_resume_checkpoint_format = "orbax"
            seed_source = "s2_adapt_eta_parent"

        result = run_stage(
            stage=stage,
            stage_map=stage_map,
            opts=opts,
            budget=budget,
            eval_spec=eval_spec,
            repo_root=repo_root,
            run_id=run_id,
            explicit_resume_checkpoint_path=explicit_resume_checkpoint_path,
            explicit_resume_checkpoint_format=explicit_resume_checkpoint_format,
            parent_refs_override=parent_refs_override,
            extra_overrides=extra_overrides,
            extra_tags={
                "benchmark": "eta",
                "protocol": "revised",
                "paper_stage_label": PAPER_STAGE_LABELS_760M[stage_id],
                "seed_source": seed_source,
            },
        )

        metrics_rows = read_metrics_rows(_metrics_path(Path(result.run_dir)))
        estimate = None
        if metrics_rows:
            estimate = estimate_stage_training_wall_seconds(
                metrics_rows=metrics_rows,
                total_steps=spec.revised_total_steps,
                warmup_steps=args.warmup_steps,
            )

        row = {
            "stage_id": stage_id,
            "paper_stage_label": spec.paper_stage_label,
            "run_id": run_id,
            "status": result.status,
            "error_message": result.error_message,
            "kind": spec.kind,
            "seed_source": seed_source,
            "benchmark_steps": args.steps,
            "revised_total_steps": spec.revised_total_steps,
            "revised_global_batch_size": spec.revised_global_batch_size,
            "run_dir": result.run_dir,
            "metrics_path": result.metrics_path,
            "benchmark_wall_seconds": result.wall_seconds,
            "benchmark_gpu_hours": result.gpu_hours,
            "last_loss_ce": _read_last_loss(metrics_rows),
        }
        if estimate is not None:
            row.update(estimate)
            row["estimated_training_gpu_hours"] = (
                float(estimate["estimated_training_wall_seconds"]) * 8.0 / 3600.0
            )
        rows.append(row)

        if str(result.status).startswith("failed"):
            payload = {
                "schema_version": "1.0",
                "created_at_utc": utc_now_iso(),
                "paper_run_id": args.paper_run_id,
                "protocol_manifest": str(protocol_out),
                "rows": rows,
            }
            summary_out = args.summary_out.expanduser().resolve() if args.summary_out else (
                repo_root / "reports" / "paper" / args.paper_run_id / "eta" / "eta_summary.json"
            )
            _write_json(summary_out, payload)
            return 1

        if stage_id == "S2_ADAPT":
            if args.dry_run:
                s2_adapt_parent_ref = CheckpointRef(
                    checkpoint_id="S2_ADAPT",
                    exp_folder=opts.exp_folder,
                    exp_name=run_id,
                )
            else:
                s2_adapt_parent_ref = resolve_checkpoint_ref(
                    checkpoint_root=opts.checkpoint_root,
                    exp_folder=opts.exp_folder,
                    checkpoint_id="S2_ADAPT",
                    exp_name=run_id,
                    allow_missing=False,
                )
            s2_adapt_resume_root = opts.checkpoint_root / opts.exp_folder / run_id

    payload = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "paper_run_id": args.paper_run_id,
        "protocol_manifest": str(protocol_out),
        "rows": rows,
    }
    summary_out = args.summary_out.expanduser().resolve() if args.summary_out else (
        repo_root / "reports" / "paper" / args.paper_run_id / "eta" / "eta_summary.json"
    )
    _write_json(summary_out, payload)
    print(f"Wrote 760M ETA benchmark summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
