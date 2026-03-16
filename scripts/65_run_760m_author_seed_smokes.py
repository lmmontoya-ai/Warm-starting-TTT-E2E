#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ttt.research.author_checkpoints import author_seed_checkpoint_ref, load_env_file
from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec, utc_now_iso


STAGE_SOURCES: dict[str, str] = {
    "S0": "760m_fa",
    "S1": "760m_fa",
    "S2_ADAPT": "760m_fa",
    "S3": "760m_e2e",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_written(checkpoint_root: Path, exp_folder: str, run_id: str) -> bool:
    return (checkpoint_root / exp_folder / run_id / "latest.json").exists()


def _first_and_last_metrics(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}, {}
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {}, {}
    return rows[0], rows[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 2-step 760M author-seeded smoke gates for S0, S1, S2_ADAPT, and S3."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/author_checkpoints"))
    parser.add_argument("--paper-run-id", default="protocol_r_760m_author_seed_smokes_v1")
    parser.add_argument("--exp-folder", default="protocol_r_760m_author_seed_smokes_v1")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--stage-ids", default="S0,S1,S2_ADAPT,S3")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Optional training.global_batch_size override for reduced-batch feasibility probes.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional suffix appended to each run_id to keep reduced-batch probe runs distinct.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override(s) forwarded to each stage run.",
    )
    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    load_env_file(repo_root / ".env")

    registry = load_registry(args.registry.expanduser().resolve())
    stage_map = registry.stage_map()
    stage_ids = [item.strip() for item in args.stage_ids.split(",") if item.strip()]
    budget = BudgetSpec(
        budget_id="760m_author_seed_smoke",
        pretrain_steps=args.steps,
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
        profile_root=(repo_root / "artifacts" / "external_models").resolve(),
        dclm_root=args.dclm_root.expanduser().resolve(),
        books_root=args.books_root.expanduser().resolve(),
        exp_folder=args.exp_folder,
        wandb_entity="none",
        wandb_project="none",
        wandb_key="env",
        save_milestone_freq=args.save_milestone_freq,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.allow_missing_fingerprints and not args.dry_run),
    )

    rows: list[dict[str, Any]] = []
    artifact_root = args.artifact_root.expanduser().resolve()
    for stage_id in stage_ids:
        stage = stage_map[stage_id]
        checkpoint_key = STAGE_SOURCES[stage_id]
        parent_ref = author_seed_checkpoint_ref(artifact_root, checkpoint_key)
        run_id = f"{stage.exp_name}-author-smoke2{args.run_suffix}"
        extra_overrides = [
            "training.load_part=params",
            "training.log_wandb=false",
            "backend.distributed=false",
        ]
        if args.global_batch_size is not None:
            extra_overrides.append(f"training.global_batch_size={args.global_batch_size}")
        extra_overrides.extend(args.extra_override)
        result = run_stage(
            stage=stage,
            stage_map=stage_map,
            opts=opts,
            budget=budget,
            eval_spec=eval_spec,
            repo_root=repo_root,
            run_id=run_id,
            explicit_resume_checkpoint_path=artifact_root / checkpoint_key,
            explicit_resume_checkpoint_format="orbax",
            parent_refs_override=[parent_ref],
            extra_overrides=extra_overrides,
            extra_tags={"smoke_gate": "true", "seed_source": checkpoint_key},
        )
        run_dir = Path(result.run_dir).resolve()
        first_metrics, last_metrics = _first_and_last_metrics(run_dir)
        checkpoint_written = _checkpoint_written(opts.checkpoint_root, opts.exp_folder, run_id)
        rows.append(
            {
                "stage_id": stage_id,
                "checkpoint_key": checkpoint_key,
                "run_id": run_id,
                "status": result.status,
                "error_message": result.error_message,
                "wall_seconds": result.wall_seconds,
                "gpu_hours": result.gpu_hours,
                "tokens_seen": result.tokens_seen,
                "run_dir": str(run_dir),
                "metrics_path": str(run_dir / "metrics.jsonl"),
                "checkpoint_written": checkpoint_written,
                "first_metric_seen": bool(first_metrics),
                "first_metric_step": first_metrics.get("step") if first_metrics else None,
                "first_metric_loss_ce": first_metrics.get("loss_ce") if first_metrics else None,
                "last_metric_step": last_metrics.get("step") if last_metrics else None,
                "last_metric_loss_ce": last_metrics.get("loss_ce") if last_metrics else None,
                "parent_checkpoint": parent_ref.to_dict(),
            }
        )

    payload = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "paper_run_id": args.paper_run_id,
        "exp_folder": args.exp_folder,
        "steps": args.steps,
        "rows": rows,
    }
    if args.summary_out is None:
        summary_out = (
            repo_root
            / "reports"
            / "paper"
            / args.paper_run_id
            / "760m_author_seed_smoke_summary.json"
        )
    else:
        summary_out = args.summary_out.expanduser().resolve()
    _write_json(summary_out, payload)
    print(f"Wrote 760M author-seed smoke summary: {summary_out}")

    if any(str(row["status"]).startswith("failed") for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
