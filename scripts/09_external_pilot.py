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
    model_key: str
    experiment: str
    exp_name: str
    kind: str
    path_group: str
    notes: str
    requires_profile: bool = False


STAGES: list[Stage] = [
    # Qwen scratch
    Stage(
        id="q_scratch_pretrain",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/pretrain-fa-scratch-8K",
        exp_name="pretrain-qwen05-fa-scratch-8K",
        kind="pretrain",
        path_group="scratch",
        notes="Qwen scratch FA stage A",
    ),
    Stage(
        id="q_scratch_ext",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-fa-32K-from-scratch",
        exp_name="ext-qwen05-fa-32K-from-scratch",
        kind="ext",
        path_group="scratch",
        notes="Qwen scratch FA stage C",
    ),
    # Qwen adapter
    Stage(
        id="q_import_pretrain",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/pretrain-fa-import-8K",
        exp_name="pretrain-qwen05-fa-import-8K",
        kind="pretrain",
        path_group="adapter",
        notes="Qwen imported FA stage A",
        requires_profile=True,
    ),
    Stage(
        id="q_import_ext_fa",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-fa-32K-from-import",
        exp_name="ext-qwen05-fa-32K-from-import",
        kind="ext",
        path_group="adapter",
        notes="Qwen imported FA 32K control",
        requires_profile=True,
    ),
    Stage(
        id="q_import_adapt_swa",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/adapt-swa-8K-from-import",
        exp_name="adapt-qwen05-swa-8K-from-import",
        kind="adapt",
        path_group="adapter",
        notes="Qwen SWA bridge",
        requires_profile=True,
    ),
    Stage(
        id="q_import_adapt_e2e",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/adapt-e2e-8K-from-import",
        exp_name="adapt-qwen05-e2e-8K-from-import",
        kind="adapt",
        path_group="adapter",
        notes="Qwen TTT bridge",
        requires_profile=True,
    ),
    Stage(
        id="q_import_ext_e2e_bridge",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-e2e-32K-from-import-bridge",
        exp_name="ext-qwen05-e2e-32K-from-import-bridge",
        kind="ext",
        path_group="adapter",
        notes="Qwen 32K TTT from bridge",
        requires_profile=True,
    ),
    Stage(
        id="q_import_ext_e2e_direct",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-e2e-32K-from-import-direct",
        exp_name="ext-qwen05-e2e-32K-from-import-direct",
        kind="ext",
        path_group="adapter",
        notes="Qwen 32K TTT direct",
        requires_profile=True,
    ),
    # Smol scratch
    Stage(
        id="s_scratch_pretrain",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/pretrain-fa-scratch-8K",
        exp_name="pretrain-smol360-fa-scratch-8K",
        kind="pretrain",
        path_group="scratch",
        notes="Smol scratch FA stage A",
    ),
    Stage(
        id="s_scratch_ext",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-fa-32K-from-scratch",
        exp_name="ext-smol360-fa-32K-from-scratch",
        kind="ext",
        path_group="scratch",
        notes="Smol scratch FA stage C",
    ),
    # Smol adapter
    Stage(
        id="s_import_pretrain",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/pretrain-fa-import-8K",
        exp_name="pretrain-smol360-fa-import-8K",
        kind="pretrain",
        path_group="adapter",
        notes="Smol imported FA stage A",
        requires_profile=True,
    ),
    Stage(
        id="s_import_ext_fa",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-fa-32K-from-import",
        exp_name="ext-smol360-fa-32K-from-import",
        kind="ext",
        path_group="adapter",
        notes="Smol imported FA 32K control",
        requires_profile=True,
    ),
    Stage(
        id="s_import_adapt_swa",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/adapt-swa-8K-from-import",
        exp_name="adapt-smol360-swa-8K-from-import",
        kind="adapt",
        path_group="adapter",
        notes="Smol SWA bridge",
        requires_profile=True,
    ),
    Stage(
        id="s_import_adapt_e2e",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/adapt-e2e-8K-from-import",
        exp_name="adapt-smol360-e2e-8K-from-import",
        kind="adapt",
        path_group="adapter",
        notes="Smol TTT bridge",
        requires_profile=True,
    ),
    Stage(
        id="s_import_ext_e2e_bridge",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-e2e-32K-from-import-bridge",
        exp_name="ext-smol360-e2e-32K-from-import-bridge",
        kind="ext",
        path_group="adapter",
        notes="Smol 32K TTT from bridge",
        requires_profile=True,
    ),
    Stage(
        id="s_import_ext_e2e_direct",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-e2e-32K-from-import-direct",
        exp_name="ext-smol360-e2e-32K-from-import-direct",
        kind="ext",
        path_group="adapter",
        notes="Smol 32K TTT direct",
        requires_profile=True,
    ),
]


def _run_cmd(cmd: list[str], dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def _checkpoint_exists(checkpoint_root: Path, exp_folder: str, exp_name: str) -> bool:
    return (checkpoint_root / exp_folder / exp_name / "latest.json").exists()


def _steps_for_stage(stage: Stage, args: argparse.Namespace) -> int:
    if stage.kind == "pretrain":
        return args.pretrain_steps
    if stage.kind == "adapt":
        return args.adapt_steps
    return args.ext_steps


def _profile_path(profile_root: Path, model_key: str) -> Path:
    return profile_root / model_key / "model_profile.json"


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

    if stage.requires_profile:
        cmd.append(
            f"training.external_profile_path={_profile_path(args.profile_root, stage.model_key)}"
        )

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


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


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


def _selected_stages(model: str, path: str) -> list[Stage]:
    out: list[Stage] = []
    for stage in STAGES:
        if model != "all" and stage.model_key != model:
            continue
        if path != "all" and stage.path_group != path:
            continue
        out.append(stage)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run short-budget scratch/adapter pilots for Qwen2.5-0.5B and "
            "SmolLM2-360M, then emit consolidated phase-1 reports."
        )
    )
    parser.add_argument("--model", default="all", choices=["all", "qwen2_5_0_5b", "smollm2_360m"])
    parser.add_argument("--path", default="all", choices=["all", "scratch", "adapter"])
    parser.add_argument("--deploy", default="interactive")
    parser.add_argument("--runtime-mode", default="token_stats", choices=["simulate", "token_stats"])

    parser.add_argument("--exp-folder", default="external_phase1_pilot")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
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

    if args.pretrain_steps <= 0 or args.adapt_steps <= 0 or args.ext_steps <= 0:
        parser.error("All step counts must be positive")

    if args.global_batch_size <= 0:
        parser.error("--global-batch-size must be positive")

    if args.seq_length <= 0:
        parser.error("--seq-length must be positive")

    args.exp_dir = args.exp_dir.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.profile_root = args.profile_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()

    if args.manifest_out is None:
        args.manifest_out = args.exp_dir / f"{args.exp_folder}_external_manifest.json"
    else:
        args.manifest_out = args.manifest_out.expanduser().resolve()

    if args.csv_out is None:
        args.csv_out = args.exp_dir / f"{args.exp_folder}_external_report.csv"
    else:
        args.csv_out = args.csv_out.expanduser().resolve()

    if args.json_out is None:
        args.json_out = args.exp_dir / f"{args.exp_folder}_external_report.json"
    else:
        args.json_out = args.json_out.expanduser().resolve()

    return args


def main() -> int:
    args = parse_args()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)

    stages = _selected_stages(model=args.model, path=args.path)
    if not stages:
        print("No stages selected with the requested filters.")
        return 0

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

    if not args.dry_run:
        for stage in stages:
            if stage.requires_profile:
                profile_path = _profile_path(args.profile_root, stage.model_key)
                if not profile_path.exists():
                    raise FileNotFoundError(
                        f"Missing external profile for {stage.model_key}: {profile_path}. "
                        "Run scripts/07_prepare_external_models.py first."
                    )

    manifest_rows: list[dict] = []

    for stage in stages:
        steps = _steps_for_stage(stage, args)
        status = "planned"

        if args.skip_existing and _checkpoint_exists(args.checkpoint_path, args.exp_folder, stage.exp_name):
            status = "skipped_existing"
            print(f"[{stage.id}] skipping existing checkpoint for {stage.exp_name}")
        else:
            cmd = _build_train_command(stage=stage, args=args, steps=steps)
            rc = _run_cmd(cmd, dry_run=args.dry_run)
            if rc != 0:
                status = f"failed_exit_{rc}"
                manifest_rows.append(
                    {
                        "stage_id": stage.id,
                        "model_key": stage.model_key,
                        "path_group": stage.path_group,
                        "experiment": stage.experiment,
                        "exp_name": stage.exp_name,
                        "kind": stage.kind,
                        "steps": steps,
                        "status": status,
                        "notes": stage.notes,
                    }
                )
                _write_manifest(args.manifest_out, manifest_rows)
                return rc
            status = "completed"

        manifest_rows.append(
            {
                "stage_id": stage.id,
                "model_key": stage.model_key,
                "path_group": stage.path_group,
                "experiment": stage.experiment,
                "exp_name": stage.exp_name,
                "kind": stage.kind,
                "steps": steps,
                "status": status,
                "notes": stage.notes,
            }
        )

    _write_manifest(args.manifest_out, manifest_rows)
    print(f"Wrote external manifest: {args.manifest_out}")

    rc = _run_report(args)
    if rc != 0:
        return rc

    print("External pilot matrix completed.")
    print(f"Report CSV: {args.csv_out}")
    print(f"Report JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
