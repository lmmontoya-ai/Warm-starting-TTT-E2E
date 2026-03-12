#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.registry import load_registry, select_stages
from ttt.research.types import BudgetSpec


def _parse_csv(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _checkpoint_exists(checkpoint_root: Path, exp_folder: str, exp_name: str) -> bool:
    return (checkpoint_root / exp_folder / exp_name / "latest.json").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run warm-start staged experiments from the research registry "
            "(S0/S1/S2/S3 ladder) with strict lineage manifests."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("./configs/research/warmstart_registry.yaml"),
    )
    parser.add_argument("--paper-run-id", default="warmstart")

    parser.add_argument("--model-key", default="all")
    parser.add_argument("--path-group", default="all")
    parser.add_argument("--stage-ids", default="", help="Comma-separated stage ids")

    parser.add_argument("--deploy", default="interactive")
    parser.add_argument(
        "--runtime-mode",
        default="token_stats",
        choices=["simulate", "token_stats", "jax_train", "jax_eval"],
    )

    parser.add_argument("--exp-folder", default="warmstart_registry")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)

    parser.add_argument("--pretrain-steps", type=int, default=8)
    parser.add_argument("--adapt-steps", type=int, default=4)
    parser.add_argument("--ext-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--save-milestone-freq", type=int, default=2)

    parser.add_argument("--eval-profile", default="default_longctx")

    parser.add_argument("--wandb-entity", default="phase1")
    parser.add_argument("--wandb-project", default="phase1")
    parser.add_argument("--wandb-key", default="none")

    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    registry = load_registry(args.registry.expanduser().resolve())
    stage_ids = _parse_csv(args.stage_ids)
    selected = select_stages(
        registry,
        model_key=args.model_key,
        path_group=args.path_group,
        stage_ids=stage_ids if stage_ids else None,
    )
    if not selected:
        raise ValueError("No stages selected from registry.")

    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    profile_root = args.profile_root.expanduser().resolve()
    dclm_root = args.dclm_root.expanduser().resolve()
    books_root = args.books_root.expanduser().resolve()

    opts = OrchestratorOptions(
        deploy=args.deploy,
        runtime_mode=args.runtime_mode,
        exp_dir=exp_dir,
        checkpoint_root=checkpoint_root,
        profile_root=profile_root,
        dclm_root=dclm_root,
        books_root=books_root,
        exp_folder=args.exp_folder,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_key=args.wandb_key,
        global_batch_size=args.global_batch_size,
        accum_steps=args.accum_steps,
        seq_length=args.seq_length,
        save_milestone_freq=args.save_milestone_freq,
        dummy_dataset=args.dummy_dataset,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.allow_missing_fingerprints),
    )

    budget = BudgetSpec(
        budget_id="registry_budget",
        pretrain_steps=args.pretrain_steps,
        adapt_steps=args.adapt_steps,
        ext_steps=args.ext_steps,
        seed=args.seed,
    )

    eval_spec = registry.eval_profiles.get(args.eval_profile)
    if eval_spec is None:
        raise KeyError(f"Unknown eval profile: {args.eval_profile}")

    stage_map = registry.stage_map()
    repo_root = Path(__file__).resolve().parents[1]

    rows = []
    for stage in selected:
        if args.skip_existing and _checkpoint_exists(checkpoint_root, args.exp_folder, stage.exp_name):
            rows.append(
                {
                    "stage_id": stage.stage_id,
                    "exp_name": stage.exp_name,
                    "status": "skipped_existing",
                }
            )
            continue

        result = run_stage(
            stage=stage,
            stage_map=stage_map,
            opts=opts,
            budget=budget,
            eval_spec=eval_spec,
            repo_root=repo_root,
            run_id=stage.exp_name,
        )
        rows.append(
            {
                "stage_id": stage.stage_id,
                "exp_name": stage.exp_name,
                "status": result.status,
                "run_dir": result.run_dir,
                "gpu_hours": result.gpu_hours,
                "tokens_seen": result.tokens_seen,
                "error_message": result.error_message,
            }
        )
        if result.status.startswith("failed"):
            break

    if args.summary_out is None:
        summary_out = (
            Path("./reports/paper")
            / args.paper_run_id
            / "registry"
            / "registry_run_summary.json"
        ).resolve()
    else:
        summary_out = args.summary_out.expanduser().resolve()

    payload = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "registry": str(args.registry),
        "exp_folder": args.exp_folder,
        "runtime_mode": args.runtime_mode,
        "stages": [s.stage_id for s in selected],
        "rows": rows,
    }
    _write_json(summary_out, payload)
    print(f"Wrote registry run summary: {summary_out}")

    if any(str(row.get("status", "")).startswith("failed") for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
