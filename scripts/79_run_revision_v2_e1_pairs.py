#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec


DEFAULT_ARMS = ("S2_125M", "S2_MINUS_125M")


def _parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        if "-" in value:
            left, right = value.split("-", 1)
            start = int(left)
            stop = int(right)
            if stop < start:
                raise ValueError(f"Invalid seed range: {value}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(value))
    if not seeds:
        raise ValueError("At least one seed is required.")
    deduped = sorted(set(seeds))
    if deduped != seeds:
        raise ValueError(f"Duplicate seeds are not allowed: {raw}")
    return seeds


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _latest_step(checkpoint_dir: Path) -> int | None:
    latest_json = checkpoint_dir / "latest.json"
    if not latest_json.exists():
        return None
    payload = _load_json(latest_json)
    return int(payload["step"])


def _run_succeeded(run_dir: Path) -> bool:
    result_path = run_dir / "run_result.json"
    if not result_path.exists():
        return False
    try:
        payload = _load_json(result_path)
    except Exception:
        return False
    return str(payload.get("status", "")).strip() == "succeeded"


def _export_stage(
    *,
    repo_root: Path,
    paper_run_id: str,
    stage_id: str,
    run_id: str,
    repo_id: str,
    exp_dir: Path,
    checkpoint_root: Path,
    checkpoint_exp_folder: str,
    dry_run: bool,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/40_export_stage_to_hf.py",
        "--paper-run-id",
        paper_run_id,
        "--stage-id",
        stage_id,
        "--run-id",
        run_id,
        "--repo-id",
        repo_id,
        "--exp-dir",
        str(exp_dir),
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-exp-folder",
        checkpoint_exp_folder,
    ]
    if dry_run:
        return {"status": "dry_run", "command": cmd}
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return {"status": "succeeded" if completed.returncode == 0 else "failed", "returncode": completed.returncode}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered revision-v2 E1 paired S2/S2-minus seeds with "
            "seed-specific run ids and shared parent checkpoints."
        )
    )
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--deploy", default="revision_v2_prime_h100_2x")
    parser.add_argument("--runtime-mode", default="jax_train", choices=["simulate", "token_stats", "jax_train"])
    parser.add_argument("--paper-run-id", default="revision_v2_e1_paired_v1")
    parser.add_argument(
        "--exp-folder",
        default="",
        help="Checkpoint exp_folder. Defaults to --paper-run-id so HF export works without extra mapping.",
    )
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--seeds", default="1-5")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--ext-steps", type=int, default=480)
    parser.add_argument("--ext-global-batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=32768)
    parser.add_argument(
        "--save-milestone-freq",
        type=int,
        default=30,
        help="Checkpoint cadence for interruption safety; does not change the training objective.",
    )
    parser.add_argument("--wandb-entity", default="none")
    parser.add_argument("--wandb-project", default="none")
    parser.add_argument("--wandb-key", default="none")
    parser.add_argument("--eval-profile", default="default_longctx")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-to-hf", action="store_true")
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_125M_RESULTS_REPO", "Luxel/ttt-e2e-125m-results"))
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(args.registry.expanduser().resolve())
    stage_map = registry.stage_map()
    eval_spec = registry.eval_profiles[args.eval_profile]
    seeds = _parse_seed_spec(args.seeds)
    arms = _parse_csv(args.arms)
    unknown = [stage_id for stage_id in arms if stage_id not in DEFAULT_ARMS]
    if unknown:
        raise ValueError(f"E1 arms must be a subset of {DEFAULT_ARMS}; got {unknown}")
    exp_folder = args.exp_folder.strip() or args.paper_run_id

    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    opts = OrchestratorOptions(
        deploy=args.deploy,
        runtime_mode=args.runtime_mode,
        exp_dir=exp_dir,
        checkpoint_root=checkpoint_root,
        profile_root=args.profile_root.expanduser().resolve(),
        dclm_root=args.dclm_root.expanduser().resolve(),
        books_root=args.books_root.expanduser().resolve(),
        exp_folder=exp_folder,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_key=args.wandb_key,
        ext_global_batch_size=args.ext_global_batch_size,
        seq_length=args.seq_length,
        save_milestone_freq=args.save_milestone_freq,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.dry_run),
    )
    budget = BudgetSpec(
        budget_id="revision_v2_e1_pair",
        pretrain_steps=0,
        adapt_steps=0,
        ext_steps=args.ext_steps,
        seed=0,
    )

    rows: list[dict[str, Any]] = []
    failed = False
    for seed in seeds:
        for stage_id in arms:
            stage = stage_map[stage_id]
            run_id = f"{stage.exp_name}-seed{seed:03d}"
            run_dir = exp_dir / args.paper_run_id / stage.stage_id / run_id
            checkpoint_dir = checkpoint_root / exp_folder / run_id
            latest_step = _latest_step(checkpoint_dir)
            extra_overrides = [
                f"training.model_seed={seed}",
                f"training.data_seed={seed}",
            ]
            explicit_resume = None
            explicit_resume_format = None
            status_prefix = ""
            if latest_step is not None and latest_step >= args.ext_steps - 1 and args.skip_complete:
                row = {
                    "seed": seed,
                    "stage_id": stage_id,
                    "run_id": run_id,
                    "checkpoint_dir": str(checkpoint_dir),
                    "latest_step": latest_step,
                    "status": "skipped_complete_checkpoint",
                }
                rows.append(row)
                if args.export_to_hf and _run_succeeded(run_dir):
                    row["hf_export"] = _export_stage(
                        repo_root=repo_root,
                        paper_run_id=args.paper_run_id,
                        stage_id=stage_id,
                        run_id=run_id,
                        repo_id=args.hf_repo_id,
                        exp_dir=exp_dir,
                        checkpoint_root=checkpoint_root,
                        checkpoint_exp_folder=exp_folder,
                        dry_run=args.dry_run,
                    )
                continue
            if latest_step is not None and args.resume_existing:
                explicit_resume = checkpoint_dir
                explicit_resume_format = "orbax"
                extra_overrides.append("training.load_part=all")
                status_prefix = "resumed_"

            result = run_stage(
                stage=stage,
                stage_map=stage_map,
                opts=opts,
                budget=budget,
                eval_spec=eval_spec,
                repo_root=repo_root,
                run_id=run_id,
                explicit_resume_checkpoint_path=explicit_resume,
                explicit_resume_checkpoint_format=explicit_resume_format,
                extra_overrides=extra_overrides,
                extra_tags={"seed": str(seed), "revision_v2_arm": stage_id},
            )
            row = {
                "seed": seed,
                "stage_id": stage_id,
                "run_id": run_id,
                "checkpoint_dir": str(checkpoint_dir),
                "latest_step_before": latest_step,
                "status": status_prefix + result.status,
                "run_dir": result.run_dir,
                "gpu_hours": result.gpu_hours,
                "tokens_seen": result.tokens_seen,
                "error_message": result.error_message,
            }
            if args.export_to_hf and result.status == "succeeded":
                row["hf_export"] = _export_stage(
                    repo_root=repo_root,
                    paper_run_id=args.paper_run_id,
                    stage_id=stage_id,
                    run_id=run_id,
                    repo_id=args.hf_repo_id,
                    exp_dir=exp_dir,
                    checkpoint_root=checkpoint_root,
                    checkpoint_exp_folder=exp_folder,
                    dry_run=args.dry_run,
                )
            rows.append(row)
            if result.status.startswith("failed"):
                failed = True
                break
        if failed:
            break

    summary_out = (
        args.summary_out.expanduser().resolve()
        if args.summary_out is not None
        else (Path("./reports/revision_v2") / args.paper_run_id / "e1_pairs_summary.json").resolve()
    )
    _write_json(
        summary_out,
        {
            "schema_version": "1.0",
            "paper_run_id": args.paper_run_id,
            "exp_folder": exp_folder,
            "deploy": args.deploy,
            "runtime_mode": args.runtime_mode,
            "seeds": seeds,
            "arms": arms,
            "ext_steps": args.ext_steps,
            "ext_global_batch_size": args.ext_global_batch_size,
            "seq_length": args.seq_length,
            "save_milestone_freq": args.save_milestone_freq,
            "rows": rows,
        },
    )
    print(f"Wrote E1 pair summary: {summary_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
