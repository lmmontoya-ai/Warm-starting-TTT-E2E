#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ttt.research.author_checkpoints import author_seed_checkpoint_ref, load_env_file
from ttt.research.orchestrator import OrchestratorOptions, run_stage
from ttt.research.protocol_760m import (
    ALL_760M_STAGE_IDS,
    CONTROL_760M_STAGE_IDS,
    AUTHOR_SEED_SOURCES_760M,
    CORE_760M_STAGE_IDS,
    PAPER_STAGE_LABELS_760M,
    PROTOCOL_R_760M_EXP_FOLDER,
    PROTOCOL_R_760M_PAPER_RUN_ID,
    build_protocol_r_760m_manifest,
    build_protocol_r_760m_stage_map,
)
from ttt.research.registry import load_registry
from ttt.research.types import BudgetSpec


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _env_default(name: str, fallback: str) -> str:
    value = str(os.environ.get(name, "")).strip()
    return value or fallback


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _checkpoint_exists(checkpoint_root: Path, exp_folder: str, run_id: str) -> bool:
    return (checkpoint_root / exp_folder / run_id / "latest.json").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the revised 760M author-seeded ladder on the local runtime. "
            "This launcher freezes the 8x H200 Protocol R-style batch reduction "
            "at global_batch_size=8 and scales stage steps to preserve token budget."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/author_checkpoints"))
    parser.add_argument("--paper-run-id", default=PROTOCOL_R_760M_PAPER_RUN_ID)
    parser.add_argument("--exp-folder", default=PROTOCOL_R_760M_EXP_FOLDER)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--revised-global-batch-size", type=int, default=8)
    parser.add_argument("--save-milestone-freq", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-entity", default=_env_default("WANDB_ENTITY", "none"))
    parser.add_argument("--wandb-project", default=_env_default("WANDB_PROJECT", "none"))
    parser.add_argument("--wandb-key", default="env")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument(
        "--phase",
        choices=("core", "controls", "all"),
        default="core",
        help=(
            "Execution tranche for the 760M author-seeded program. "
            "'core' runs S2_ADAPT, S2, S3; 'controls' runs S0, S1; "
            "'all' runs the full ladder."
        ),
    )
    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        seed=args.seed,
    )
    protocol_out = (
        repo_root / "reports" / "paper" / args.paper_run_id / "launch" / "protocol_manifest.json"
    ).resolve()
    _write_json(protocol_out, protocol_manifest)

    budget = BudgetSpec(
        budget_id="760m_author_seed_protocol_r",
        pretrain_steps=0,
        adapt_steps=protocol_stage_map["S2_ADAPT"].revised_total_steps,
        ext_steps=protocol_stage_map["S0"].revised_total_steps,
        seed=args.seed,
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
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_key=args.wandb_key,
        global_batch_size=args.revised_global_batch_size,
        ext_global_batch_size=args.revised_global_batch_size,
        save_milestone_freq=args.save_milestone_freq,
        dry_run=args.dry_run,
        paper_run_id=args.paper_run_id,
        require_dataset_fingerprint=(not args.allow_missing_fingerprints and not args.dry_run),
    )

    rows: list[dict[str, Any]] = []
    train_failed = False
    artifact_root = args.artifact_root.expanduser().resolve()
    selected_stage_ids = {
        "core": CORE_760M_STAGE_IDS,
        "controls": CONTROL_760M_STAGE_IDS,
        "all": ALL_760M_STAGE_IDS,
    }[args.phase]
    if not args.skip_train:
        for stage_id in selected_stage_ids:
            stage = stage_map[stage_id]
            run_id = stage.exp_name
            spec = protocol_stage_map[stage_id]
            if args.skip_existing and _checkpoint_exists(opts.checkpoint_root, opts.exp_folder, run_id):
                rows.append(
                    {
                        "step_id": f"train:{stage_id}",
                        "stage_id": stage_id,
                        "paper_stage_label": spec.paper_stage_label,
                        "run_id": run_id,
                        "status": "skipped_existing",
                        "revised_global_batch_size": spec.revised_global_batch_size,
                        "revised_total_steps": spec.revised_total_steps,
                    }
                )
                continue

            parent_refs_override = None
            explicit_resume_checkpoint_path = None
            explicit_resume_checkpoint_format = None
            extra_overrides = ["backend.distributed=false"]
            seed_source = ""

            if stage_id in AUTHOR_SEED_SOURCES_760M:
                seed_source = AUTHOR_SEED_SOURCES_760M[stage_id]
                parent_refs_override = [author_seed_checkpoint_ref(artifact_root, seed_source)]
                explicit_resume_checkpoint_path = artifact_root / seed_source
                explicit_resume_checkpoint_format = "orbax"
                extra_overrides.append("training.load_part=params")

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
                    "protocol": "revised",
                    "protocol_family": "760m_author_seed",
                    "paper_stage_label": PAPER_STAGE_LABELS_760M[stage_id],
                    "seed_source": seed_source or "stage_parent",
                },
            )
            rows.append(
                {
                    "step_id": f"train:{stage_id}",
                    "stage_id": stage_id,
                    "paper_stage_label": spec.paper_stage_label,
                    "run_id": run_id,
                    "status": result.status,
                    "error_message": result.error_message,
                    "gpu_hours": result.gpu_hours,
                    "tokens_seen": result.tokens_seen,
                    "run_dir": result.run_dir,
                    "revised_global_batch_size": spec.revised_global_batch_size,
                    "revised_total_steps": spec.revised_total_steps,
                    "seed_source": seed_source or "stage_parent",
                }
            )
            if str(result.status).startswith("failed"):
                train_failed = True
                break

    summary_out = (
        repo_root / "reports" / "paper" / args.paper_run_id / "launch" / "launcher_summary.json"
    ).resolve()
    if train_failed:
        _write_json(
            summary_out,
            {
                "schema_version": "1.0",
                "paper_run_id": args.paper_run_id,
                "phase": args.phase,
                "protocol_manifest": str(protocol_out),
                "rows": rows,
            },
        )
        return 1

    if not args.skip_eval:
        eval_stages = ",".join(stage_id for stage_id in selected_stage_ids if stage_id != "S2_ADAPT")
        cmd_eval = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/34_eval_matrix_jax.py",
            "--paper-run-id",
            args.paper_run_id,
            "--exp-dir",
            str(opts.exp_dir),
            "--checkpoint-root",
            str(opts.checkpoint_root),
            "--exp-folder",
            args.exp_folder,
            "--stages",
            eval_stages,
            "--contexts",
            "32768",
            "--datasets",
            "books3",
            "--dclm-root",
            str(opts.dclm_root),
            "--books-root",
            str(opts.books_root),
            "--eval-split",
            "val",
            "--eval-batches",
            str(args.eval_batches),
            "--eval-batch-size",
            str(args.eval_batch_size),
        ]
        if args.strict:
            cmd_eval.append("--strict")
        if args.dry_run:
            cmd_eval.append("--dry-run")
        rc = _run(cmd_eval, dry_run=args.dry_run)
        rows.append({"step_id": "eval:books32k", "command": cmd_eval, "returncode": rc})
        if rc != 0:
            _write_json(
                summary_out,
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "phase": args.phase,
                    "protocol_manifest": str(protocol_out),
                    "rows": rows,
                },
            )
            return rc

    if not args.skip_tables:
        if args.phase != "all":
            rows.append(
                {
                    "step_id": "tables",
                    "status": "skipped_phase",
                    "phase": args.phase,
                    "reason": "tables require the full S0/S1/S2/S3 comparison set",
                }
            )
        else:
            cmd_tables = [
                "uv",
                "run",
                "--exact",
                "python",
                "scripts/20_make_paper_tables.py",
                "--paper-run-id",
                args.paper_run_id,
                "--exp-dir",
                str(opts.exp_dir),
                "--s0-stage",
                "S0",
                "--s1-stage",
                "S1",
                "--s2-stage",
                "S2",
                "--s3-stage",
                "S3",
            ]
            rc = _run(cmd_tables, dry_run=args.dry_run)
            rows.append({"step_id": "tables", "command": cmd_tables, "returncode": rc})
            if rc != 0:
                _write_json(
                    summary_out,
                    {
                        "schema_version": "1.0",
                        "paper_run_id": args.paper_run_id,
                        "phase": args.phase,
                        "protocol_manifest": str(protocol_out),
                        "rows": rows,
                    },
                )
                return rc

    if not args.skip_figures:
        if args.phase != "all":
            rows.append(
                {
                    "step_id": "figures",
                    "status": "skipped_phase",
                    "phase": args.phase,
                    "reason": "figures require the full S0/S1/S2/S3 comparison set",
                }
            )
        else:
            cmd_figures = [
                "uv",
                "run",
                "--exact",
                "python",
                "scripts/21_make_paper_figures.py",
                "--paper-run-id",
                args.paper_run_id,
            ]
            rc = _run(cmd_figures, dry_run=args.dry_run)
            rows.append({"step_id": "figures", "command": cmd_figures, "returncode": rc})
            if rc != 0:
                _write_json(
                    summary_out,
                    {
                        "schema_version": "1.0",
                        "paper_run_id": args.paper_run_id,
                        "phase": args.phase,
                        "protocol_manifest": str(protocol_out),
                        "rows": rows,
                    },
                )
                return rc

    _write_json(
        summary_out,
        {
            "schema_version": "1.0",
            "paper_run_id": args.paper_run_id,
            "phase": args.phase,
            "protocol_manifest": str(protocol_out),
            "rows": rows,
        },
    )
    print(f"Wrote launcher summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
