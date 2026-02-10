#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    id: str
    experiment: str
    exp_name: str
    kind: str
    notes: str
    extra_overrides: tuple[str, ...] = ()


STAGES: list[Stage] = [
    Stage(
        id="B1-pretrain",
        experiment="760m/pretrain/pretrain-760m-e2e",
        exp_name="pretrain-760m-e2e",
        kind="pretrain",
        notes="Paper-style E2E baseline stage A",
    ),
    Stage(
        id="B1-ext",
        experiment="760m/extension/ext-760m-e2e-32K",
        exp_name="ext-760m-e2e-32K",
        kind="ext",
        notes="Paper-style E2E baseline stage C",
        extra_overrides=(
            "training.load_part=params",
            "training.resume_exp_name=pretrain-760m-e2e",
        ),
    ),
    Stage(
        id="B2-pretrain",
        experiment="760m/pretrain/pretrain-760m-fa",
        exp_name="pretrain-760m-fa",
        kind="pretrain",
        notes="Non-TTT FA baseline stage A",
    ),
    Stage(
        id="B2-ext",
        experiment="760m/extension/ext-760m-fa-32K",
        exp_name="ext-760m-fa-32K",
        kind="ext",
        notes="Non-TTT FA baseline stage C",
        extra_overrides=(
            "training.load_part=params",
            "training.resume_exp_name=pretrain-760m-fa",
        ),
    ),
    Stage(
        id="P1-adapt",
        experiment="760m/pretrained/adapt-760m-e2e-8K-from-fa",
        exp_name="adapt-760m-e2e-8K-from-fa",
        kind="adapt",
        notes="Warm-start adaptation stage B",
    ),
    Stage(
        id="P1-ext-bridge",
        experiment="760m/pretrained/ext-760m-e2e-32K-from-fa-bridge",
        exp_name="ext-760m-e2e-32K-from-fa-bridge",
        kind="ext",
        notes="Warm-start extension from adaptation",
    ),
    Stage(
        id="P2-ext-direct",
        experiment="760m/pretrained/ext-760m-e2e-32K-from-fa-direct",
        exp_name="ext-760m-e2e-32K-from-fa-direct",
        kind="ext",
        notes="Warm-start extension direct from FA",
    ),
]


def _run_cmd(cmd: list[str], dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def _checkpoint_exists(checkpoint_root: Path, exp_folder: str, exp_name: str) -> bool:
    latest_path = checkpoint_root / exp_folder / exp_name / "latest.json"
    return latest_path.exists()


def _steps_for_stage(stage: Stage, args: argparse.Namespace) -> int:
    if stage.kind == "pretrain":
        return args.pretrain_steps
    if stage.kind == "adapt":
        return args.adapt_steps
    return args.ext_steps


def _build_train_command(stage: Stage, args: argparse.Namespace, steps: int) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--exact",
        "train",
        f"+deploy={args.deploy}",
        f"+experiment={stage.experiment}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_dir={args.exp_dir}",
        f"training.checkpoint_path={args.checkpoint_path}",
        f"training.total_steps={steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        f"training.runtime_mode={args.runtime_mode}",
        f"training.wandb_entity={args.wandb_entity}",
        f"training.wandb_project={args.wandb_project}",
        f"training.wandb_key={args.wandb_key}",
        f"deploy_paths.data.dclm_filter_8k={args.data_root}",
        f"deploy_paths.data.books3={args.data_root}",
        f"deploy_paths.checkpoint={args.checkpoint_path}",
    ]

    if args.global_batch_size is not None:
        cmd.append(f"training.global_batch_size={args.global_batch_size}")
    if args.seq_length is not None:
        cmd.append(f"training.seq_length={args.seq_length}")
    if args.dummy_dataset:
        cmd.append("training.dummy_dataset=true")

    cmd.extend(stage.extra_overrides)

    return cmd


def _bootstrap_token_data(args: argparse.Namespace) -> int:
    cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/04_make_token_data.py",
        "--out",
        str(args.data_root),
        "--train-tokens",
        str(args.bootstrap_train_tokens),
        "--val-tokens",
        str(args.bootstrap_val_tokens),
        "--vocab-size",
        str(args.bootstrap_vocab_size),
        "--seed",
        str(args.bootstrap_seed),
    ]
    return _run_cmd(cmd, dry_run=args.dry_run)


def _write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True))


def _run_report(args: argparse.Namespace) -> int:
    csv_cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/05_phase1_report.py",
        "--exp-dir",
        str(args.exp_dir),
        "--exp-folder",
        args.exp_folder,
        "--format",
        "csv",
        "--csv-out",
        str(args.csv_out),
    ]
    rc = _run_cmd(csv_cmd, dry_run=args.dry_run)
    if rc != 0:
        return rc

    json_cmd = [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/05_phase1_report.py",
        "--exp-dir",
        str(args.exp_dir),
        "--exp-folder",
        args.exp_folder,
        "--format",
        "json",
    ]
    print("$ " + " ".join(shlex.quote(part) for part in json_cmd), flush=True)
    if args.dry_run:
        return 0

    completed = subprocess.run(json_cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        return completed.returncode

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(completed.stdout)
    print(f"Wrote report json: {args.json_out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short-budget pilot matrix for pretrained warm-start experiments "
            "(B1/B2/P1/P2) and emit consolidated phase-1 reports."
        )
    )
    parser.add_argument("--deploy", default="interactive")
    parser.add_argument("--runtime-mode", default="token_stats", choices=["simulate", "token_stats"])
    parser.add_argument("--exp-folder", default="phase1_pilot")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/phase1_token_data"))

    parser.add_argument("--pretrain-steps", type=int, default=8)
    parser.add_argument("--adapt-steps", type=int, default=4)
    parser.add_argument("--ext-steps", type=int, default=4)
    parser.add_argument("--save-milestone-freq", type=int, default=2)

    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=128)

    parser.add_argument("--wandb-entity", default="phase1")
    parser.add_argument("--wandb-project", default="phase1")
    parser.add_argument("--wandb-key", default="none")

    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--bootstrap-token-data", action="store_true")
    parser.add_argument("--bootstrap-train-tokens", type=int, default=20000)
    parser.add_argument("--bootstrap-val-tokens", type=int, default=4000)
    parser.add_argument("--bootstrap-vocab-size", type=int, default=4096)
    parser.add_argument("--bootstrap-seed", type=int, default=0)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)

    args = parser.parse_args()

    if args.runtime_mode == "token_stats" and not args.dummy_dataset and not args.data_root:
        parser.error("--data-root is required when runtime-mode=token_stats and dummy dataset is disabled")

    if args.pretrain_steps <= 0 or args.adapt_steps <= 0 or args.ext_steps <= 0:
        parser.error("step counts must be positive")

    if args.global_batch_size is not None and args.global_batch_size <= 0:
        parser.error("--global-batch-size must be positive")

    if args.seq_length is not None and args.seq_length <= 0:
        parser.error("--seq-length must be positive")

    args.exp_dir = args.exp_dir.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.manifest_out is None:
        args.manifest_out = args.exp_dir / f"{args.exp_folder}_manifest.json"
    else:
        args.manifest_out = args.manifest_out.expanduser().resolve()

    if args.csv_out is None:
        args.csv_out = args.exp_dir / f"{args.exp_folder}_report.csv"
    else:
        args.csv_out = args.csv_out.expanduser().resolve()

    if args.json_out is None:
        args.json_out = args.exp_dir / f"{args.exp_folder}_report.json"
    else:
        args.json_out = args.json_out.expanduser().resolve()

    return args


def main() -> int:
    args = parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)

    if args.bootstrap_token_data and not args.dummy_dataset:
        rc = _bootstrap_token_data(args)
        if rc != 0:
            return rc

    if (
        not args.dummy_dataset
        and args.runtime_mode == "token_stats"
        and not args.data_root.exists()
        and not args.dry_run
    ):
        raise FileNotFoundError(
            f"Data root does not exist: {args.data_root}. "
            "Provide --data-root or use --bootstrap-token-data."
        )

    manifest_entries: list[dict] = []

    for stage in STAGES:
        steps = _steps_for_stage(stage, args)
        ckpt_exists = _checkpoint_exists(args.checkpoint_path, args.exp_folder, stage.exp_name)
        status = "planned"

        if args.skip_existing and ckpt_exists:
            status = "skipped_existing"
            print(f"[{stage.id}] skipping existing checkpoint for {stage.exp_name}")
        else:
            cmd = _build_train_command(stage, args, steps)
            rc = _run_cmd(cmd, dry_run=args.dry_run)
            if rc != 0:
                status = f"failed_exit_{rc}"
                manifest_entries.append(
                    {
                        "stage_id": stage.id,
                        "experiment": stage.experiment,
                        "exp_name": stage.exp_name,
                        "kind": stage.kind,
                        "steps": steps,
                        "status": status,
                        "notes": stage.notes,
                    }
                )
                _write_manifest(args.manifest_out, manifest_entries)
                return rc
            status = "completed"

        manifest_entries.append(
            {
                "stage_id": stage.id,
                "experiment": stage.experiment,
                "exp_name": stage.exp_name,
                "kind": stage.kind,
                "steps": steps,
                "status": status,
                "notes": stage.notes,
            }
        )

    _write_manifest(args.manifest_out, manifest_entries)
    print(f"Wrote pilot manifest: {args.manifest_out}")

    rc = _run_report(args)
    if rc != 0:
        return rc

    print("Pilot matrix completed.")
    print(f"Report CSV: {args.csv_out}")
    print(f"Report JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
