#!/usr/bin/env python3
"""Run the warm-start stage ladder defined in the research registry.

Each selected stage is launched as ``<this interpreter> ttt/train.py ...`` so
the stages run in the same environment as this script (under ``uv run`` that
is the project environment). Stages run in registry order; a stage's parent
checkpoint is resolved from ``required_parent_checkpoint_ids`` under
``<checkpoint-root>/<exp-folder>/`` and is the checkpoint the stage restores.

``--exp-folder`` and ``--paper-run-id`` are folder names, not paths:
checkpoints go to ``<checkpoint-root>/<exp-folder>/<run>`` and run artifacts to
``<exp-dir>/<paper-run-id>/<stage>/<run>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ttt.infra.paths import path_name_error
from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.registry import load_registry, select_stages
from ttt.research.types import BudgetSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_csv(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _checkpoint_exists(checkpoint_root: Path, exp_folder: str, exp_name: str) -> bool:
    return (checkpoint_root / exp_folder / exp_name / "latest.json").exists()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run warm-start staged experiments from the research registry "
            "(S0/S1/S2/S3 ladder) with strict lineage manifests."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "configs" / "research" / "warmstart_registry.yaml",
    )
    parser.add_argument(
        "--paper-run-id",
        default="warmstart",
        help="Folder name for run artifacts under --exp-dir.",
    )

    parser.add_argument("--model-key", default="all")
    parser.add_argument("--path-group", default="all")
    parser.add_argument("--stage-ids", default="", help="Comma-separated stage ids")

    parser.add_argument("--deploy", default="interactive")
    parser.add_argument(
        "--runtime-mode",
        default="token_stats",
        choices=["simulate", "token_stats", "jax_train", "jax_eval"],
    )

    parser.add_argument(
        "--exp-folder",
        default="warmstart_registry",
        help=(
            "Folder name (not a path) that groups this ladder's checkpoints under "
            "--checkpoint-root. Use --checkpoint-root to put checkpoints elsewhere."
        ),
    )
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
    parser.add_argument("--ext-global-batch-size", type=int, default=None)
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

    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Defaults to <exp-dir>/<paper-run-id>/registry_run_summary.json.",
    )
    args = parser.parse_args(argv)
    for flag, value in (("--exp-folder", args.exp_folder), ("--paper-run-id", args.paper_run_id)):
        error = path_name_error(value, field=flag)
        if error is not None:
            parser.error(f"{error} Use --exp-dir and --checkpoint-root to choose output locations.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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
        ext_global_batch_size=args.ext_global_batch_size,
        accum_steps=args.accum_steps,
        seq_length=args.seq_length,
        save_milestone_freq=args.save_milestone_freq,
        dummy_dataset=args.dummy_dataset,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.allow_missing_fingerprints),
        # Launch stages with this interpreter instead of nesting `uv run`.
        python_executable=sys.executable,
        repo_root=REPO_ROOT,
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
    repo_root = REPO_ROOT

    if not args.skip_existing and not args.dry_run:
        # Re-running a finished stage overwrites its checkpoint, which breaks the
        # recorded lineage of every child that was trained from it. Refuse up
        # front instead of failing later on a parent hash mismatch.
        existing = [
            stage.stage_id
            for stage in selected
            if _checkpoint_exists(checkpoint_root, args.exp_folder, stage.exp_name)
        ]
        if existing:
            print(
                "ERROR: checkpoints already exist under "
                f"{checkpoint_root / args.exp_folder} for: {', '.join(existing)}. "
                "Pass --skip-existing to keep finished stages and run the rest, or use a "
                "new --exp-folder and --paper-run-id for a fresh ladder.",
                file=sys.stderr,
            )
            return 1

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

        print(f"=== {stage.stage_id} ({stage.exp_name})", flush=True)
        try:
            result = run_stage(
                stage=stage,
                stage_map=stage_map,
                opts=opts,
                budget=budget,
                eval_spec=eval_spec,
                repo_root=repo_root,
                run_id=stage.exp_name,
            )
        except (FileNotFoundError, ValueError) as exc:
            # Preconditions checked before launch: missing parent checkpoint,
            # dataset fingerprint or external profile.
            print(f"ERROR: {stage.stage_id} not started: {exc}", file=sys.stderr, flush=True)
            rows.append(
                {
                    "stage_id": stage.stage_id,
                    "exp_name": stage.exp_name,
                    "status": "failed_precondition",
                    "error_message": str(exc),
                }
            )
            break
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
            print(
                f"ERROR: {stage.stage_id} failed: {result.error_message} (run dir: {result.run_dir})",
                file=sys.stderr,
                flush=True,
            )
            break

    if args.summary_out is None:
        summary_out = exp_dir / args.paper_run_id / "registry_run_summary.json"
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
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print("Registry ladder completed: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
