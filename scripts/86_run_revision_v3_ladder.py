#!/usr/bin/env python3
"""Run the revision-v3 width-preserving 125M warm-start ladder.

Order of operations (each stage is skipped when a complete checkpoint exists):

  1. static gates: registry shape check, FA parent present;
  2. S0_125M   full-attention 32K extension of the seed (baseline);
  3. S1_125M   sliding-window-only conversion + 32K extension (baseline);
  4. S2_ADAPT_125M  10% bridge at 8K (batch 64, 480 steps);
  5. for each seed: S2_125M (bridge -> 32K) and S2_MINUS_125M (seed -> 32K),
     with explicit model/data seeds so the pairs match the E1 preregistration.

Every warm-started stage now runs behind the trainer's restore-coverage and
initial-loss gates, so a conversion that silently discards inherited weights
fails fast instead of producing a number.
"""

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
from ttt.research.warmstart_validity import check_registry_warmstart_shapes, render_shape_report


SINGLE_STAGES = ("S0_125M", "S1_125M", "S2_ADAPT_125M")
PAIRED_STAGES = ("S2_125M", "S2_MINUS_125M")
FA_PARENT_STAGE = "S0_PRETRAIN_FA_125M"
DEFAULT_PAPER_RUN_ID = "revision_v3_widthpreserving_v1"


def _parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        if "-" in value:
            left, right = value.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"Invalid seed range: {value}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(value))
    if not seeds:
        raise ValueError("At least one seed is required.")
    if sorted(set(seeds)) != seeds:
        raise ValueError(f"Seeds must be unique and ascending: {raw}")
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
    return int(_load_json(latest_json)["step"])


def _run_succeeded(run_dir: Path) -> bool:
    result_path = run_dir / "run_result.json"
    if not result_path.exists():
        return False
    try:
        return str(_load_json(result_path).get("status", "")).strip() == "succeeded"
    except Exception:
        return False


def _export_stage(*, repo_root: Path, paper_run_id: str, stage_id: str, run_id: str, repo_id: str, exp_dir: Path, checkpoint_root: Path, checkpoint_exp_folder: str, dry_run: bool) -> dict[str, Any]:
    cmd = [
        "uv", "run", "--exact", "python", "scripts/40_export_stage_to_hf.py",
        "--paper-run-id", paper_run_id,
        "--stage-id", stage_id,
        "--run-id", run_id,
        "--repo-id", repo_id,
        "--exp-dir", str(exp_dir),
        "--checkpoint-root", str(checkpoint_root),
        "--checkpoint-exp-folder", checkpoint_exp_folder,
    ]
    if dry_run:
        return {"status": "dry_run", "command": cmd}
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return {"status": "succeeded" if completed.returncode == 0 else "failed", "returncode": completed.returncode}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--deploy", default="revision_v3_vast_h100_8x")
    parser.add_argument("--runtime-mode", default="jax_train", choices=["simulate", "token_stats", "jax_train"])
    parser.add_argument("--paper-run-id", default=DEFAULT_PAPER_RUN_ID)
    parser.add_argument("--exp-folder", default="", help="Checkpoint exp_folder; defaults to --paper-run-id.")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--stages", default=",".join([*SINGLE_STAGES, *PAIRED_STAGES]))
    parser.add_argument("--seeds", default="1-5", help="Seeds for the paired S2/S2-minus extensions.")
    parser.add_argument("--adapt-steps", type=int, default=480, help="Bridge steps at 8K, batch 64 (10%% budget).")
    parser.add_argument("--ext-steps", type=int, default=480)
    parser.add_argument("--ext-global-batch-size", type=int, default=8)
    parser.add_argument("--save-milestone-freq", type=int, default=30)
    parser.add_argument("--wandb-entity", default="none")
    parser.add_argument("--wandb-project", default="none")
    parser.add_argument("--wandb-key", default="none")
    parser.add_argument("--eval-profile", default="default_longctx")
    parser.add_argument("--skip-shape-check", action="store_true", help="Not recommended; the check needs no GPU.")
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
    requested = _parse_csv(args.stages)
    unknown = [s for s in requested if s not in (*SINGLE_STAGES, *PAIRED_STAGES)]
    if unknown:
        raise ValueError(f"Unknown ladder stage(s) {unknown}; allowed={[*SINGLE_STAGES, *PAIRED_STAGES]}")
    exp_folder = args.exp_folder.strip() or args.paper_run_id
    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()

    validity: list[dict[str, str]] = []
    if not args.skip_shape_check:
        report = check_registry_warmstart_shapes(registry=registry, repo_root=repo_root, stage_ids=requested)
        print(render_shape_report(report))
        validity.append({"check": "registry_warmstart_shapes", "status": "PASS" if report.ok else "FAIL", "reason": "" if report.ok else render_shape_report(report)})
        if not report.ok:
            print("Refusing to launch: a warm-started stage would silently discard inherited weights.")
            return 2
    fa_parent = checkpoint_root / exp_folder / stage_map[FA_PARENT_STAGE].exp_name
    fa_step = _latest_step(fa_parent)
    validity.append({"check": "fa_parent_checkpoint_present", "status": "PASS" if fa_step is not None else "FAIL", "reason": str(fa_parent)})
    if fa_step is None and not args.dry_run:
        print(f"Refusing to launch: FA parent checkpoint missing at {fa_parent} (restore it with scripts/46_restore_stage_from_hf.py).")
        return 2

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
        # Extension stages run at batch 8 (one 32K sequence per GPU on 8 devices);
        # the bridge keeps its config batch of 64 at 8K. seq_length stays per-config.
        ext_global_batch_size=args.ext_global_batch_size,
        save_milestone_freq=args.save_milestone_freq,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.dry_run),
    )
    budget = BudgetSpec(
        budget_id="revision_v3_ladder",
        pretrain_steps=0,
        adapt_steps=args.adapt_steps,
        ext_steps=args.ext_steps,
        seed=0,
    )

    rows: list[dict[str, Any]] = []
    failed = False

    def _launch(stage_id: str, run_id: str, *, total_steps: int, seed: int | None) -> bool:
        stage = stage_map[stage_id]
        run_dir = exp_dir / args.paper_run_id / stage.stage_id / run_id
        checkpoint_dir = checkpoint_root / exp_folder / run_id
        latest_step = _latest_step(checkpoint_dir)
        extra_overrides: list[str] = []
        if seed is not None:
            extra_overrides += [f"training.model_seed={seed}", f"training.data_seed={seed}"]
        explicit_resume = None
        explicit_resume_format = None
        status_prefix = ""
        row: dict[str, Any] = {"seed": seed, "stage_id": stage_id, "run_id": run_id, "checkpoint_dir": str(checkpoint_dir)}
        if latest_step is not None and latest_step >= total_steps - 1 and args.skip_complete:
            row.update({"latest_step": latest_step, "status": "skipped_complete_checkpoint"})
            rows.append(row)
            if args.export_to_hf and _run_succeeded(run_dir):
                row["hf_export"] = _export_stage(repo_root=repo_root, paper_run_id=args.paper_run_id, stage_id=stage_id, run_id=run_id, repo_id=args.hf_repo_id, exp_dir=exp_dir, checkpoint_root=checkpoint_root, checkpoint_exp_folder=exp_folder, dry_run=args.dry_run)
            return True
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
            extra_tags={"revision_v3_stage": stage_id, **({"seed": str(seed)} if seed is not None else {})},
        )
        row.update({
            "latest_step_before": latest_step,
            "status": status_prefix + result.status,
            "run_dir": result.run_dir,
            "gpu_hours": result.gpu_hours,
            "tokens_seen": result.tokens_seen,
            "error_message": result.error_message,
        })
        if args.export_to_hf and result.status == "succeeded":
            row["hf_export"] = _export_stage(repo_root=repo_root, paper_run_id=args.paper_run_id, stage_id=stage_id, run_id=run_id, repo_id=args.hf_repo_id, exp_dir=exp_dir, checkpoint_root=checkpoint_root, checkpoint_exp_folder=exp_folder, dry_run=args.dry_run)
        rows.append(row)
        return not result.status.startswith("failed")

    for stage_id in SINGLE_STAGES:
        if stage_id not in requested:
            continue
        stage = stage_map[stage_id]
        total = args.adapt_steps if stage.kind == "adapt" else args.ext_steps
        if not _launch(stage_id, stage.exp_name, total_steps=total, seed=None):
            failed = True
            break

    if not failed:
        for seed in seeds:
            for stage_id in PAIRED_STAGES:
                if stage_id not in requested:
                    continue
                run_id = f"{stage_map[stage_id].exp_name}-seed{seed:03d}"
                if not _launch(stage_id, run_id, total_steps=args.ext_steps, seed=seed):
                    failed = True
                    break
            if failed:
                break

    summary_out = (
        args.summary_out.expanduser().resolve()
        if args.summary_out is not None
        else (Path("./reports/revision_v3") / args.paper_run_id / "ladder_summary.json").resolve()
    )
    _write_json(
        summary_out,
        {
            "schema_version": "1.0",
            "paper_run_id": args.paper_run_id,
            "exp_folder": exp_folder,
            "deploy": args.deploy,
            "runtime_mode": args.runtime_mode,
            "stages": requested,
            "seeds": seeds,
            "adapt_steps": args.adapt_steps,
            "ext_steps": args.ext_steps,
            "ext_global_batch_size": args.ext_global_batch_size,
            "save_milestone_freq": args.save_milestone_freq,
            "validity_checks": validity,
            "rows": rows,
        },
    )
    print(f"Wrote revision-v3 ladder summary: {summary_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
