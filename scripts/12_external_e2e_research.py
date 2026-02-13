#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


FULL_STEPS = {
    "qwen2_5_0_5b": {"pretrain": 12000, "adapt": 12000, "ext": 300},
    "smollm2_360m": {"pretrain": 9600, "adapt": 9600, "ext": 240},
}


def _run(cmd: list[str], dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _models_for_selection(model: str) -> list[str]:
    if model == "all":
        return ["qwen2_5_0_5b", "smollm2_360m"]
    return [model]


def _prepare_profiles(args: argparse.Namespace) -> int:
    model_arg = args.model if args.model != "all" else "all"
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/07_prepare_external_models.py",
        "--model",
        model_arg,
        "--out-root",
        str(args.profile_root),
    ]
    return _run(cmd, dry_run=args.dry_run)


def _seed_imports(args: argparse.Namespace) -> int:
    model_arg = args.model if args.model != "all" else "all"
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/10_seed_external_import_checkpoints.py",
        "--model",
        model_arg,
        "--profile-root",
        str(args.profile_root),
        "--checkpoint-root",
        str(args.checkpoint_path),
        "--exp-folder",
        args.exp_folder,
    ]
    if args.seed_dataset_root:
        cmd.extend(["--dataset-root", str(args.seed_dataset_root)])
    if args.force_seed_imports:
        cmd.append("--force")
    return _run(cmd, dry_run=args.dry_run)


def _run_training_for(
    *,
    args: argparse.Namespace,
    model: str,
    pretrain_steps: int,
    adapt_steps: int,
    ext_steps: int,
) -> int:
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/09_external_pilot.py",
        "--model",
        model,
        "--path",
        args.path,
        "--deploy",
        args.deploy,
        "--runtime-mode",
        args.runtime_mode,
        "--exp-folder",
        args.exp_folder,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--profile-root",
        str(args.profile_root),
        "--dclm-root",
        str(args.dclm_root),
        "--books-root",
        str(args.books_root),
        "--pretrain-steps",
        str(pretrain_steps),
        "--adapt-steps",
        str(adapt_steps),
        "--ext-steps",
        str(ext_steps),
        "--save-milestone-freq",
        str(args.save_milestone_freq),
        "--global-batch-size",
        str(args.global_batch_size),
        "--seq-length",
        str(args.seq_length),
        "--wandb-entity",
        args.wandb_entity,
        "--wandb-project",
        args.wandb_project,
        "--wandb-key",
        args.wandb_key,
    ]
    if args.bootstrap_token_data:
        cmd.append("--bootstrap-token-data")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.dummy_dataset:
        cmd.append("--dummy-dataset")
    if args.dry_run:
        cmd.append("--dry-run")
    return _run(cmd, dry_run=args.dry_run)


def _run_training(args: argparse.Namespace) -> int:
    if args.budget == "pilot":
        return _run_training_for(
            args=args,
            model=args.model,
            pretrain_steps=args.pretrain_steps,
            adapt_steps=args.adapt_steps,
            ext_steps=args.ext_steps,
        )

    for model in _models_for_selection(args.model):
        steps = FULL_STEPS[model]
        rc = _run_training_for(
            args=args,
            model=model,
            pretrain_steps=steps["pretrain"],
            adapt_steps=steps["adapt"],
            ext_steps=steps["ext"],
        )
        if rc != 0:
            return rc
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/11_external_eval.py",
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_path),
        "--exp-folder",
        args.exp_folder,
        "--contexts",
        args.eval_contexts,
        "--datasets",
        args.eval_datasets,
        "--dclm-root",
        str(args.dclm_root),
        "--books-root",
        str(args.books_root),
        "--eval-split",
        args.eval_split,
        "--eval-batches",
        str(args.eval_batches),
        "--eval-seed",
        str(args.eval_seed),
        "--niah-examples",
        str(args.niah_examples),
        "--niah-candidates",
        str(args.niah_candidates),
        "--niah-positions",
        args.niah_positions,
        "--decode-steps",
        str(args.decode_steps),
        "--decode-prompts",
        str(args.decode_prompts),
    ]
    if args.eval_batch_size > 0:
        cmd.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.strict_eval:
        cmd.append("--strict")
    return _run(cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end orchestrator for external-model research: "
            "profiles -> import checkpoints -> training matrix -> eval suite."
        )
    )
    parser.add_argument("--model", default="all", choices=["all", "qwen2_5_0_5b", "smollm2_360m"])
    parser.add_argument("--path", default="all", choices=["all", "scratch", "adapter"])
    parser.add_argument("--budget", default="pilot", choices=["pilot", "full"])

    parser.add_argument("--deploy", default="interactive")
    parser.add_argument("--runtime-mode", default="token_stats", choices=["simulate", "token_stats"])

    parser.add_argument("--exp-folder", default="external_phase1_research")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--dclm-root", type=Path, default=Path("/tmp/phase1_token_data_dclm"))
    parser.add_argument("--books-root", type=Path, default=Path("/tmp/phase1_token_data_books"))

    parser.add_argument("--pretrain-steps", type=int, default=8)
    parser.add_argument("--adapt-steps", type=int, default=4)
    parser.add_argument("--ext-steps", type=int, default=4)
    parser.add_argument("--save-milestone-freq", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=128)

    parser.add_argument("--wandb-entity", default="phase1")
    parser.add_argument("--wandb-project", default="phase1")
    parser.add_argument("--wandb-key", default="none")

    parser.add_argument("--bootstrap-token-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dummy-dataset", action="store_true")

    parser.add_argument("--skip-prepare-profiles", action="store_true")
    parser.add_argument("--skip-seed-imports", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument(
        "--seed-dataset-root",
        type=Path,
        default=None,
        help="Optional token dataset root for import-checkpoint calibration seeding.",
    )
    parser.add_argument("--force-seed-imports", action="store_true")

    parser.add_argument("--eval-contexts", default="8192,32768,65536,131072")
    parser.add_argument("--eval-datasets", default="books3")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--niah-examples", type=int, default=64)
    parser.add_argument("--niah-candidates", type=int, default=16)
    parser.add_argument("--niah-positions", default="0.1,0.5,0.9")
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--decode-prompts", type=int, default=8)
    parser.add_argument("--strict-eval", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.exp_dir = args.exp_dir.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.profile_root = args.profile_root.expanduser().resolve()
    args.dclm_root = args.dclm_root.expanduser().resolve()
    args.books_root = args.books_root.expanduser().resolve()
    if args.seed_dataset_root is not None:
        args.seed_dataset_root = args.seed_dataset_root.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare_profiles:
        rc = _prepare_profiles(args)
        if rc != 0:
            return rc

    if not args.skip_seed_imports:
        rc = _seed_imports(args)
        if rc != 0:
            return rc

    if not args.skip_train:
        rc = _run_training(args)
        if rc != 0:
            return rc

    if not args.skip_eval:
        rc = _run_eval(args)
        if rc != 0:
            return rc

    print("External end-to-end research flow completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
