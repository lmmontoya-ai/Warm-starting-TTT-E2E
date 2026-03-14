#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reference 125M 32K FA extension smoke from the read-only "
            "ttte2e_reference snapshot using the same dataset/checkpoint inputs as "
            "the local parity runtime."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--resume-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, default=4560)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--wandb-key", default=os.environ.get("WANDB_API_KEY", ""))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "none"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "none"))
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--exp-folder", default="reference_smokes")
    parser.add_argument("--exp-name", default="ext-125m-fa-32K-ref-smoke")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ref_root = repo_root / "ttte2e_reference" / "e2e"
    log_path = args.log_path or (
        repo_root
        / "artifacts"
        / "reference_smokes"
        / args.exp_name
        / "reference_125m_32k_fa_smoke.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "--python",
        args.python_version,
        "--exact",
        "train",
        "+deploy=interactive",
        "+experiment=125m/extension/ext-125m-fa-32K",
        f"training.checkpoint_path={args.checkpoint_root}",
        f"training.exp_dir={args.exp_dir}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_name={args.exp_name}",
        f"training.total_steps={args.steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        "training.load_part=params",
        f"training.resume_step={args.resume_step}",
        "training.log_wandb=false",
        f"training.wandb_entity={args.wandb_entity}",
        f"training.wandb_project={args.wandb_project}",
        f"training.wandb_key={args.wandb_key}",
        f"checkpoint.resume_checkpoint_dir={args.resume_checkpoint_dir}",
        f"deploy_paths.data.books3={args.books_root}",
        f"deploy_paths.checkpoint={args.checkpoint_root}",
    ]
    if args.global_batch_size is not None:
        cmd.append(f"training.global_batch_size={args.global_batch_size}")
    if args.seq_length is not None:
        cmd.append(f"training.seq_length={args.seq_length}")

    if not args.wandb_key and not args.dry_run:
        raise SystemExit("WANDB_API_KEY must be available for the reference smoke wrapper.")

    redacted_cmd = [
        "training.wandb_key=<redacted>" if part.startswith("training.wandb_key=") else part
        for part in cmd
    ]

    manifest = {
        "cwd": str(ref_root),
        "command": redacted_cmd,
        "log_path": str(log_path),
    }
    _write_json(log_path.with_suffix(".manifest.json"), manifest)

    print(
        "$ (cd "
        + shlex.quote(str(ref_root))
        + " && "
        + " ".join(shlex.quote(part) for part in redacted_cmd)
        + ")"
    )
    if args.dry_run:
        return 0

    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=ref_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
