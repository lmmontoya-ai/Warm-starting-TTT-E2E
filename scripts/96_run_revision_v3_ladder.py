#!/usr/bin/env python3
"""Run the revision-v3 125M warm-start ladder (width-preserving conversion, fixed restore and schedules).

What runs (each stage is skipped when a complete checkpoint exists):

  once      S3W_PRETRAIN_E2E_125M   parameter-matched scratch TTT-E2E pretrain (only if requested)
  per seed  S0_125M                 FA seed -> 32K full attention
            S1_125M                 FA seed -> 32K sliding window, no TTT
            S3_125M                 scratch TTT-E2E pretrain -> 32K (re-extension under the fixed schedule)
            S2_ADAPT_125M           FA seed -> 8K bridge; one independent bridge per seed
            S2_125M                 that seed's bridge -> 32K
            S2_MINUS_125M           FA seed -> 32K TTT-E2E directly (no bridge)
            S3W_125M                parameter-matched scratch -> 32K

Defaults follow the published recipe (Tandon et al. p.28): extensions run 120
steps at global batch 32 (4-way gradient accumulation on 8 GPUs, one 32K
sequence per GPU per micro-step), and every stage's LR schedule is rescaled to
its own step count (10% warmup, cosine to the last step). ``--variant`` tags
the warm-start runs (bridge, S2, S2-minus) so pilot arms such as a bridge
peak-LR sweep or the prime post-norm ablation can share one paper run id.

Every run goes through the trainer's gates: restore coverage, restore value
check, initial loss, and LR schedule fit. Use ``--eval-after`` to chain the
64-batch Books32K evaluation (and DCLM-8K for bridges) with
scripts/34_eval_matrix_jax.py before the box is released.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ttt.research.lineage import resolve_checkpoint_ref
from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec
from ttt.research.warmstart_validity import check_registry_warmstart_shapes, render_shape_report


SINGLE_STAGES = ("S3W_PRETRAIN_E2E_125M",)
PER_SEED_STAGES = ("S0_125M", "S1_125M", "S3_125M", "S2_ADAPT_125M", "S2_125M", "S2_MINUS_125M", "S3W_125M")
ALL_STAGES = (*SINGLE_STAGES, *PER_SEED_STAGES)
DEFAULT_STAGES = ("S0_125M", "S1_125M", "S3_125M", "S2_ADAPT_125M", "S2_125M", "S2_MINUS_125M")
# The bridge and its extension always carry the variant tag. S2-minus carries it
# only when a variant override changes it (the prime post-norm init), so an
# LR-only sweep shares one direct-conversion run across its arms.
VARIANT_STAGES = ("S2_ADAPT_125M", "S2_125M", "S2_MINUS_125M")
BRIDGE_STAGE = "S2_ADAPT_125M"
BRIDGED_EXT_STAGE = "S2_125M"
FA_PARENT_STAGE = "S0_PRETRAIN_FA_125M"
E2E_PARENT_STAGE = "S3_PRETRAIN_E2E_125M"
FA_CHILDREN = ("S0_125M", "S1_125M", "S2_ADAPT_125M", "S2_MINUS_125M")
EXT_EVAL_STAGES = ("S0_125M", "S1_125M", "S3_125M", "S2_125M", "S2_MINUS_125M", "S3W_125M")
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


def _sanitize_tag(raw: str) -> str:
    tag = raw.strip()
    if tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]*", tag):
        raise ValueError(f"--variant must be alphanumeric with '.' or '-' (got {raw!r})")
    return tag


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
        "uv", "run", "--exact", "python", "scripts/40_export_stage.py",
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


def stage_is_tagged(stage_id: str, args: argparse.Namespace) -> bool:
    if stage_id in ("S2_ADAPT_125M", "S2_125M"):
        return True
    if stage_id == "S2_MINUS_125M":
        return args.prime_post_norm_init is not None
    return False


def run_id_for(stage_exp_name: str, *, stage_id: str, seed: int | None, tag: str, shared_bridge: bool, tagged: bool = True) -> str:
    """Deterministic run ids: ``<exp_name>[-<variant>][-seedNNN]``."""

    parts = [stage_exp_name]
    if tag and tagged and stage_id in VARIANT_STAGES:
        parts.append(tag)
    if seed is not None and not (shared_bridge and stage_id == BRIDGE_STAGE):
        parts.append(f"seed{seed:03d}")
    return "-".join(parts)


def variant_overrides(stage_id: str, args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    if stage_id == BRIDGE_STAGE and args.adapt_peak_lr is not None:
        overrides.append(f"training.optimizer_outer.lr={args.adapt_peak_lr}")
    # The prime post-norm init only affects freshly created prime modules: the
    # bridge and the direct conversion. S2 restores its prime modules.
    if stage_id in (BRIDGE_STAGE, "S2_MINUS_125M") and args.prime_post_norm_init is not None:
        overrides.append(f"model.prime_post_norm_init={args.prime_post_norm_init}")
    return overrides


def plan_runs(*, requested: list[str], seeds: list[int], stage_map: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Expand the requested stages into an ordered list of concrete runs."""

    tag = _sanitize_tag(args.variant)
    runs: list[dict[str, Any]] = []
    for stage_id in SINGLE_STAGES:
        if stage_id in requested:
            runs.append({"stage_id": stage_id, "run_id": stage_map[stage_id].exp_name, "seed": None, "resume_exp_name": None, "extra": []})
    shared_bridge_done = False
    for seed in seeds:
        for stage_id in PER_SEED_STAGES:
            if stage_id not in requested:
                continue
            stage = stage_map[stage_id]
            if stage_id == BRIDGE_STAGE and args.shared_bridge:
                if shared_bridge_done:
                    continue
                shared_bridge_done = True
            run_seed = seed
            run_id = run_id_for(
                stage.exp_name,
                stage_id=stage_id,
                seed=run_seed,
                tag=tag,
                shared_bridge=args.shared_bridge,
                tagged=stage_is_tagged(stage_id, args),
            )
            extra = [*variant_overrides(stage_id, args)]
            if not (args.shared_bridge and stage_id == BRIDGE_STAGE):
                extra += [f"training.model_seed={seed}", f"training.data_seed={seed}"]
            resume_exp_name = None
            if stage_id == BRIDGED_EXT_STAGE:
                resume_exp_name = run_id_for(
                    stage_map[BRIDGE_STAGE].exp_name, stage_id=BRIDGE_STAGE, seed=seed, tag=tag, shared_bridge=args.shared_bridge
                )
                extra.append(f"training.resume_exp_name={resume_exp_name}")
            runs.append({"stage_id": stage_id, "run_id": run_id, "seed": seed, "resume_exp_name": resume_exp_name, "extra": extra})
    return runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES), help=f"Subset of {','.join(ALL_STAGES)}.")
    parser.add_argument("--seeds", default="1", help="Seeds, e.g. '1' or '1-3'. Every per-seed stage runs once per seed.")
    parser.add_argument("--variant", default="", help="Tag for bridge / S2 / S2-minus run ids, e.g. lr1e-3 or gain0.")
    parser.add_argument("--adapt-peak-lr", type=float, default=None, help="Bridge peak LR (config default 3e-3).")
    parser.add_argument("--prime-post-norm-init", type=float, default=None, help="Prime post-norm gain for fresh prime modules (default 1.0).")
    parser.add_argument("--shared-bridge", action="store_true", help="Legacy: one bridge feeds every seed's S2 (pseudo-replication).")
    parser.add_argument("--pretrain-steps", type=int, default=4800, help="S3W pretrain steps at 8K, batch 64.")
    parser.add_argument("--adapt-steps", type=int, default=480, help="Bridge steps at 8K, batch 64 (480 = 10%% of pretraining).")
    parser.add_argument("--ext-steps", type=int, default=120)
    parser.add_argument("--ext-global-batch-size", type=int, default=32)
    parser.add_argument("--ext-accum-steps", type=int, default=4)
    parser.add_argument("--schedule-policy", default="scaled", choices=["scaled", "config"])
    parser.add_argument("--save-milestone-freq", type=int, default=30)
    parser.add_argument("--wandb-entity", default="none")
    parser.add_argument("--wandb-project", default="none")
    parser.add_argument("--wandb-key", default="none")
    parser.add_argument("--eval-profile", default="default_longctx")
    parser.add_argument("--skip-shape-check", action="store_true", help="Not recommended; the check needs no GPU.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue with later runs after a failure.")
    parser.add_argument("--skip-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-after", action="store_true", help="Chain scripts/34_eval_matrix_jax.py after training.")
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--export-to-hf",
        action="store_true",
        help="Upload each succeeded stage with scripts/40_export_stage.py to --hf-repo-id.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=os.environ.get("HF_125M_RESULTS_REPO", ""),
        help="Your Hugging Face model repo for --export-to-hf (default: $HF_125M_RESULTS_REPO).",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args(argv)


def _eval_commands(*, args: argparse.Namespace, exp_folder: str, requested: list[str], report_dir: Path) -> list[list[str]]:
    base = [
        sys.executable, "scripts/34_eval_matrix_jax.py",
        "--paper-run-id", args.paper_run_id,
        "--exp-dir", str(args.exp_dir),
        "--checkpoint-root", str(args.checkpoint_root),
        "--exp-folder", exp_folder,
        "--dclm-root", str(args.dclm_root),
        "--books-root", str(args.books_root),
        "--eval-batches", str(args.eval_batches),
        "--eval-batch-size", str(args.eval_batch_size),
        "--strict",
    ]
    commands: list[list[str]] = []
    ext_stages = [s for s in EXT_EVAL_STAGES if s in requested]
    if ext_stages:
        commands.append([
            *base,
            "--eval-id", "jax_books32k_eval64",
            "--stages", ",".join(ext_stages),
            "--contexts", "32768",
            "--datasets", "books3",
            "--summary-json", str(report_dir / "books32k_eval64_summary.json"),
            "--summary-csv", str(report_dir / "books32k_eval64_summary.csv"),
        ])
    if BRIDGE_STAGE in requested:
        commands.append([
            *base,
            "--eval-id", "jax_dclm8k_eval64",
            "--stages", BRIDGE_STAGE,
            "--contexts", "8192",
            "--datasets", "dclm_filter_8k",
            "--summary-json", str(report_dir / "bridge_dclm8k_eval64_summary.json"),
            "--summary-csv", str(report_dir / "bridge_dclm8k_eval64_summary.csv"),
        ])
    return commands


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(args.registry.expanduser().resolve())
    stage_map = registry.stage_map()
    eval_spec = registry.eval_profiles[args.eval_profile]
    seeds = _parse_seed_spec(args.seeds)
    requested = _parse_csv(args.stages)
    unknown = [s for s in requested if s not in ALL_STAGES]
    if unknown:
        raise ValueError(f"Unknown ladder stage(s) {unknown}; allowed={list(ALL_STAGES)}")
    if args.export_to_hf and not args.hf_repo_id.strip():
        raise ValueError(
            "--export-to-hf needs --hf-repo-id (or HF_125M_RESULTS_REPO) naming a Hugging Face repo you own."
        )
    if BRIDGED_EXT_STAGE in requested and BRIDGE_STAGE not in requested:
        print(f"Note: {BRIDGED_EXT_STAGE} resumes from per-seed bridges; they must already exist (tag {args.variant!r}).")
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
    parents_needed = []
    if any(s in requested for s in FA_CHILDREN):
        parents_needed.append(FA_PARENT_STAGE)
    if "S3_125M" in requested:
        parents_needed.append(E2E_PARENT_STAGE)
    for parent_id in parents_needed:
        parent_dir = checkpoint_root / exp_folder / stage_map[parent_id].exp_name
        present = _latest_step(parent_dir) is not None
        validity.append({"check": f"parent_present:{parent_id}", "status": "PASS" if present else "FAIL", "reason": str(parent_dir)})
        if not present and not args.dry_run:
            print(
                f"Refusing to launch: parent checkpoint missing at {parent_dir} "
                f"(restore it with scripts/46_restore_stage.py --target-paper-run-id {exp_folder})."
            )
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
        ext_global_batch_size=args.ext_global_batch_size,
        ext_accum_steps=args.ext_accum_steps,
        schedule_policy=args.schedule_policy,
        save_milestone_freq=args.save_milestone_freq,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.dry_run),
    )
    budget = BudgetSpec(
        budget_id="revision_v3_ladder",
        pretrain_steps=args.pretrain_steps,
        adapt_steps=args.adapt_steps,
        ext_steps=args.ext_steps,
        seed=0,
    )
    steps_by_kind = {"pretrain": args.pretrain_steps, "adapt": args.adapt_steps, "ext": args.ext_steps}

    planned = plan_runs(requested=requested, seeds=seeds, stage_map=stage_map, args=args)
    rows: list[dict[str, Any]] = []
    failed = False
    for item in planned:
        stage = stage_map[item["stage_id"]]
        run_id = item["run_id"]
        total_steps = steps_by_kind[stage.kind]
        run_dir = exp_dir / args.paper_run_id / stage.stage_id / run_id
        checkpoint_dir = checkpoint_root / exp_folder / run_id
        latest_step = _latest_step(checkpoint_dir)
        row: dict[str, Any] = {k: item[k] for k in ("stage_id", "run_id", "seed", "resume_exp_name")}
        row["checkpoint_dir"] = str(checkpoint_dir)
        row["overrides"] = list(item["extra"])
        if latest_step is not None and latest_step >= total_steps - 1 and args.skip_complete:
            row.update({"latest_step": latest_step, "status": "skipped_complete_checkpoint"})
            if args.export_to_hf and _run_succeeded(run_dir):
                row["hf_export"] = _export_stage(repo_root=repo_root, paper_run_id=args.paper_run_id, stage_id=stage.stage_id, run_id=run_id, repo_id=args.hf_repo_id, exp_dir=exp_dir, checkpoint_root=checkpoint_root, checkpoint_exp_folder=exp_folder, dry_run=args.dry_run)
            rows.append(row)
            continue
        extra_overrides = list(item["extra"])
        explicit_resume = None
        explicit_resume_format = None
        status_prefix = ""
        if latest_step is not None and args.resume_existing:
            # Crash recovery within the same run id: restore weights and optimizer.
            explicit_resume = checkpoint_dir
            explicit_resume_format = "orbax"
            extra_overrides.append("training.load_part=all")
            status_prefix = "resumed_"
        parent_refs_override = None
        if item["resume_exp_name"]:
            parent_refs_override = [
                resolve_checkpoint_ref(
                    checkpoint_root=checkpoint_root,
                    exp_folder=exp_folder,
                    checkpoint_id=BRIDGE_STAGE,
                    exp_name=item["resume_exp_name"],
                    allow_missing=args.dry_run,
                )
            ]
        tags = {"revision_v3_stage": stage.stage_id}
        if item["seed"] is not None:
            tags["seed"] = str(item["seed"])
        if args.variant:
            tags["variant"] = args.variant
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
            parent_refs_override=parent_refs_override,
            extra_overrides=extra_overrides,
            extra_tags=tags,
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
            row["hf_export"] = _export_stage(repo_root=repo_root, paper_run_id=args.paper_run_id, stage_id=stage.stage_id, run_id=run_id, repo_id=args.hf_repo_id, exp_dir=exp_dir, checkpoint_root=checkpoint_root, checkpoint_exp_folder=exp_folder, dry_run=args.dry_run)
        rows.append(row)
        if result.status.startswith("failed"):
            failed = True
            if not args.keep_going:
                break

    report_dir = Path("./reports/revision_v3") / args.paper_run_id
    eval_rows: list[dict[str, Any]] = []
    if args.eval_after and not failed:
        for cmd in _eval_commands(args=args, exp_folder=exp_folder, requested=requested, report_dir=report_dir):
            if args.dry_run:
                eval_rows.append({"status": "dry_run", "command": cmd})
                continue
            completed = subprocess.run(cmd, cwd=repo_root, check=False)
            eval_rows.append({"status": "succeeded" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "command": cmd})
            if completed.returncode != 0:
                failed = True

    summary_out = args.summary_out.expanduser().resolve() if args.summary_out is not None else (report_dir / (f"ladder_summary_{args.variant}.json" if args.variant else "ladder_summary.json")).resolve()
    _write_json(
        summary_out,
        {
            "schema_version": "2.0",
            "paper_run_id": args.paper_run_id,
            "exp_folder": exp_folder,
            "deploy": args.deploy,
            "runtime_mode": args.runtime_mode,
            "stages": requested,
            "seeds": seeds,
            "variant": args.variant,
            "adapt_peak_lr": args.adapt_peak_lr,
            "prime_post_norm_init": args.prime_post_norm_init,
            "shared_bridge": bool(args.shared_bridge),
            "pretrain_steps": args.pretrain_steps,
            "adapt_steps": args.adapt_steps,
            "ext_steps": args.ext_steps,
            "ext_global_batch_size": args.ext_global_batch_size,
            "ext_accum_steps": args.ext_accum_steps,
            "schedule_policy": args.schedule_policy,
            "save_milestone_freq": args.save_milestone_freq,
            "validity_checks": validity,
            "rows": rows,
            "eval": eval_rows,
        },
    )
    print(f"Wrote revision-v3 ladder summary: {summary_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
