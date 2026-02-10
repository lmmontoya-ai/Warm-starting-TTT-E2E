#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentCommand:
    label: str
    experiment: str
    notes: str


def build_matrix() -> list[ExperimentCommand]:
    return [
        ExperimentCommand(
            label="B1-e2e-baseline",
            experiment="760m/pretrain/pretrain-760m-e2e",
            notes="Baseline E2E pretrain",
        ),
        ExperimentCommand(
            label="B2-fa-baseline",
            experiment="760m/pretrain/pretrain-760m-fa",
            notes="Baseline non-TTT FA pretrain",
        ),
        ExperimentCommand(
            label="P1-adapt-bridge-stageB",
            experiment="760m/pretrained/adapt-760m-e2e-8K-from-fa",
            notes="8K adaptation from FA",
        ),
        ExperimentCommand(
            label="P1-adapt-bridge-stageC",
            experiment="760m/pretrained/ext-760m-e2e-32K-from-fa-bridge",
            notes="32K extension from adaptation checkpoint",
        ),
        ExperimentCommand(
            label="P2-direct",
            experiment="760m/pretrained/ext-760m-e2e-32K-from-fa-direct",
            notes="Direct 32K extension from FA checkpoint",
        ),
    ]


def make_cli_command(exp: ExperimentCommand, args) -> str:
    parts = [
        "uv run --exact train",
        f"+deploy={args.deploy}",
        f"+experiment={exp.experiment}",
        f"training.exp_folder={args.exp_folder}",
        f"training.wandb_entity={args.wandb_entity}",
        f"training.wandb_project={args.wandb_project}",
        f"training.wandb_key={args.wandb_key}",
        f"deploy_paths.data.dclm_filter_8k={args.dclm_path}",
        f"deploy_paths.data.books3={args.books_path}",
        f"deploy_paths.checkpoint={args.checkpoint_path}",
        f"training.runtime_mode={args.runtime_mode}",
    ]
    if args.dummy_dataset:
        parts.append("training.dummy_dataset=true")
    return " \\\n  ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print launch commands for the pretrained warm-start experiment matrix."
    )
    parser.add_argument("--deploy", default="interactive")
    parser.add_argument("--exp-folder", default="pretrained_phase1")
    parser.add_argument("--wandb-entity", default="YOUR_ENTITY")
    parser.add_argument("--wandb-project", default="YOUR_PROJECT")
    parser.add_argument("--wandb-key", default="YOUR_KEY")
    parser.add_argument("--dclm-path", default="/path/to/dclm_filter_8k")
    parser.add_argument("--books-path", default="/path/to/books3")
    parser.add_argument("--checkpoint-path", default="/path/to/checkpoints")
    parser.add_argument(
        "--runtime-mode",
        default="simulate",
        choices=["simulate", "token_stats"],
        help="Phase-1 runtime backend to use.",
    )
    parser.add_argument(
        "--dummy-dataset",
        action="store_true",
        help="Append training.dummy_dataset=true for local simulator runs.",
    )
    args = parser.parse_args()

    print("# Pretrained warm-start matrix")
    for exp in build_matrix():
        print(f"\n## {exp.label}  ({exp.notes})")
        print(make_cli_command(exp, args))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
