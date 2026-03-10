#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


B1_B2_STAGES = "S0_PRETRAIN_FA,S0,S3_PRETRAIN_E2E,S3"


def _run(cmd: list[str], dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command real JAX runner for B1/B2 baseline ladder: "
            "pretrain-760m-fa -> ext-760m-fa-32K and "
            "pretrain-760m-e2e -> ext-760m-e2e-32K."
        )
    )
    parser.add_argument("--paper-run-id", default="b1_b2_real")
    parser.add_argument("--exp-folder", default="b1_b2_real")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))

    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)

    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--adapt-steps", type=int, default=0)
    parser.add_argument("--ext-steps", type=int, default=20)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--save-milestone-freq", type=int, default=10)

    parser.add_argument("--wandb-entity", default="phase1")
    parser.add_argument("--wandb-project", default="phase1")
    parser.add_argument("--wandb-key", default="none")

    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-ruler", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-bundle", action="store_true")

    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cmd_train = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/23_warmstart_registry.py",
        "--paper-run-id",
        args.paper_run_id,
        "--registry",
        "./configs/research/warmstart_registry.yaml",
        "--stage-ids",
        B1_B2_STAGES,
        "--runtime-mode",
        "jax_train",
        "--exp-folder",
        args.exp_folder,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--profile-root",
        str(args.profile_root),
        "--dclm-root",
        str(args.dclm_root),
        "--books-root",
        str(args.books_root),
        "--pretrain-steps",
        str(args.pretrain_steps),
        "--adapt-steps",
        str(args.adapt_steps),
        "--ext-steps",
        str(args.ext_steps),
        "--global-batch-size",
        str(args.global_batch_size),
        "--seq-length",
        str(args.seq_length),
        "--save-milestone-freq",
        str(args.save_milestone_freq),
        "--wandb-entity",
        args.wandb_entity,
        "--wandb-project",
        args.wandb_project,
        "--wandb-key",
        args.wandb_key,
    ]
    if args.allow_missing_fingerprints:
        cmd_train.append("--allow-missing-fingerprints")
    if args.dummy_dataset:
        cmd_train.append("--dummy-dataset")
    if args.skip_existing:
        cmd_train.append("--skip-existing")
    if args.dry_run:
        cmd_train.append("--dry-run")

    rc = _run(cmd_train, dry_run=args.dry_run)
    if rc != 0:
        return rc

    if not args.skip_eval:
        cmd_eval = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/18_eval_matrix.py",
            "--paper-run-id",
            args.paper_run_id,
            "--exp-dir",
            str(args.exp_dir),
            "--checkpoint-root",
            str(args.checkpoint_root),
            "--exp-folder",
            args.exp_folder,
            "--contexts",
            "8192,32768",
            "--datasets",
            "books3",
            "--dclm-root",
            str(args.dclm_root),
            "--books-root",
            str(args.books_root),
        ]
        if args.strict:
            cmd_eval.append("--strict")
        rc = _run(cmd_eval, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not args.skip_ruler:
        cmd_ruler = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/24_eval_ruler.py",
            "--paper-run-id",
            args.paper_run_id,
            "--exp-dir",
            str(args.exp_dir),
        ]
        if args.strict:
            cmd_ruler.append("--strict")
        rc = _run(cmd_ruler, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not args.skip_tables:
        cmd_tables = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/20_make_paper_tables.py",
            "--paper-run-id",
            args.paper_run_id,
            "--exp-dir",
            str(args.exp_dir),
        ]
        rc = _run(cmd_tables, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not args.skip_figures:
        cmd_fig = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/21_make_paper_figures.py",
            "--paper-run-id",
            args.paper_run_id,
        ]
        rc = _run(cmd_fig, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not args.skip_bundle:
        cmd_bundle = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/22_make_artifact_bundle.py",
            "--paper-run-id",
            args.paper_run_id,
            "--exp-dir",
            str(args.exp_dir),
        ]
        rc = _run(cmd_bundle, dry_run=args.dry_run)
        if rc != 0:
            return rc

    print("B1/B2 real JAX pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
