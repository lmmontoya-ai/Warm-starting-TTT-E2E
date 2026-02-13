#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentCommand:
    label: str
    model_key: str
    experiment: str
    notes: str
    path_group: str
    requires_profile: bool = False


COMMANDS: list[ExperimentCommand] = [
    # Qwen scratch path
    ExperimentCommand(
        label="Q-scratch-pretrain-fa",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/pretrain-fa-scratch-8K",
        notes="Qwen scratch FA stage A",
        path_group="scratch",
    ),
    ExperimentCommand(
        label="Q-scratch-ext-fa-32k",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-fa-32K-from-scratch",
        notes="Qwen scratch FA stage C",
        path_group="scratch",
    ),
    # Qwen adapter path
    ExperimentCommand(
        label="Q-import-pretrain-fa",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/pretrain-fa-import-8K",
        notes="Qwen imported FA stage A",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="Q-import-ext-fa-32k",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-fa-32K-from-import",
        notes="Qwen imported FA control stage C",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="Q-import-adapt-swa-8k",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/adapt-swa-8K-from-import",
        notes="Qwen FA->SWA bridge",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="Q-import-adapt-e2e-8k",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/adapt-e2e-8K-from-import",
        notes="Qwen TTT bridge stage",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="Q-import-ext-e2e-32k-bridge",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-e2e-32K-from-import-bridge",
        notes="Qwen 32K TTT extension from bridge",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="Q-import-ext-e2e-32k-direct",
        model_key="qwen2_5_0_5b",
        experiment="external/qwen2_5_0_5b/ext-e2e-32K-from-import-direct",
        notes="Qwen 32K TTT direct extension",
        path_group="adapter",
        requires_profile=True,
    ),
    # Smol scratch path
    ExperimentCommand(
        label="S-scratch-pretrain-fa",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/pretrain-fa-scratch-8K",
        notes="Smol scratch FA stage A",
        path_group="scratch",
    ),
    ExperimentCommand(
        label="S-scratch-ext-fa-32k",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-fa-32K-from-scratch",
        notes="Smol scratch FA stage C",
        path_group="scratch",
    ),
    # Smol adapter path
    ExperimentCommand(
        label="S-import-pretrain-fa",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/pretrain-fa-import-8K",
        notes="Smol imported FA stage A",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="S-import-ext-fa-32k",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-fa-32K-from-import",
        notes="Smol imported FA control stage C",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="S-import-adapt-swa-8k",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/adapt-swa-8K-from-import",
        notes="Smol FA->SWA bridge",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="S-import-adapt-e2e-8k",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/adapt-e2e-8K-from-import",
        notes="Smol TTT bridge stage",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="S-import-ext-e2e-32k-bridge",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-e2e-32K-from-import-bridge",
        notes="Smol 32K TTT extension from bridge",
        path_group="adapter",
        requires_profile=True,
    ),
    ExperimentCommand(
        label="S-import-ext-e2e-32k-direct",
        model_key="smollm2_360m",
        experiment="external/smollm2_360m/ext-e2e-32K-from-import-direct",
        notes="Smol 32K TTT direct extension",
        path_group="adapter",
        requires_profile=True,
    ),
]


def _profile_path(profile_root: Path, model_key: str) -> Path:
    return profile_root / model_key / "model_profile.json"


def _selected_commands(model: str, path_group: str) -> list[ExperimentCommand]:
    out: list[ExperimentCommand] = []
    for cmd in COMMANDS:
        if model != "all" and cmd.model_key != model:
            continue
        if path_group != "all" and cmd.path_group != path_group:
            continue
        out.append(cmd)
    return out


def _build_cli(exp: ExperimentCommand, args: argparse.Namespace) -> str:
    parts = [
        "uv run --exact train",
        f"+deploy={args.deploy}",
        f"+experiment={exp.experiment}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_dir={args.exp_dir}",
        f"training.checkpoint_path={args.checkpoint_path}",
        f"training.runtime_mode={args.runtime_mode}",
        f"training.wandb_entity={args.wandb_entity}",
        f"training.wandb_project={args.wandb_project}",
        f"training.wandb_key={args.wandb_key}",
        f"deploy_paths.data.dclm_filter_8k={args.dclm_path}",
        f"deploy_paths.data.books3={args.books_path}",
        f"deploy_paths.checkpoint={args.checkpoint_path}",
    ]

    if args.global_batch_size is not None:
        parts.append(f"training.global_batch_size={args.global_batch_size}")
    if args.seq_length is not None:
        parts.append(f"training.seq_length={args.seq_length}")
    if args.total_steps is not None:
        parts.append(f"training.total_steps={args.total_steps}")
    if args.dummy_dataset:
        parts.append("training.dummy_dataset=true")

    if exp.requires_profile:
        profile_path = _profile_path(Path(args.profile_root), exp.model_key)
        parts.append(f"training.external_profile_path={profile_path}")

    return " \\\n  ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print launch commands for external-model scratch/adapter experiments."
    )
    parser.add_argument("--model", default="all", choices=["all", "qwen2_5_0_5b", "smollm2_360m"])
    parser.add_argument("--path", default="all", choices=["all", "scratch", "adapter"])
    parser.add_argument("--deploy", default="interactive")
    parser.add_argument("--exp-folder", default="external_phase1")
    parser.add_argument("--exp-dir", default="./experiments")
    parser.add_argument("--checkpoint-path", default="./checkpoints")
    parser.add_argument("--profile-root", default="./artifacts/external_models")
    parser.add_argument("--wandb-entity", default="YOUR_ENTITY")
    parser.add_argument("--wandb-project", default="YOUR_PROJECT")
    parser.add_argument("--wandb-key", default="YOUR_KEY")
    parser.add_argument("--dclm-path", default="/path/to/dclm_filter_8k")
    parser.add_argument("--books-path", default="/path/to/books3")
    parser.add_argument("--runtime-mode", default="token_stats", choices=["simulate", "token_stats"])
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--dummy-dataset", action="store_true")
    args = parser.parse_args()

    selected = _selected_commands(model=args.model, path_group=args.path)
    if not selected:
        print("No experiments selected.")
        return 0

    print("# External pretrained matrix")
    for exp in selected:
        print(f"\n## {exp.label}  ({exp.notes})")
        print(_build_cli(exp, args))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
