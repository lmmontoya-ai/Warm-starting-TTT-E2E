#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


STAGES_125M = ",".join(
    [
        "S0_PRETRAIN_FA_125M",
        "S0_125M",
        "S1_125M",
        "S2_ADAPT_125M",
        "S2_125M",
        "S3_PRETRAIN_E2E_125M",
        "S3_125M",
    ]
)


def _run(cmd: list[str], dry_run: bool) -> int:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _env_default(name: str, fallback: str) -> str:
    value = str(os.environ.get(name, "")).strip()
    return value or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command 125M ladder runner using the parity jax_train/jax_eval path "
            "plus stage-aware aggregation, tables, figures, and artifact bundling."
        )
    )
    parser.add_argument("--paper-run-id", default="warmstart_125m")
    parser.add_argument("--exp-folder", default="warmstart_125m")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))

    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)

    parser.add_argument("--pretrain-steps", type=int, default=4800)
    parser.add_argument("--adapt-steps", type=int, default=480)
    parser.add_argument("--ext-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--save-milestone-freq", type=int, default=120)

    parser.add_argument("--contexts", default="8192,32768")
    parser.add_argument("--datasets", default="books3")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)

    parser.add_argument("--wandb-entity", default=_env_default("WANDB_ENTITY", "none"))
    parser.add_argument("--wandb-project", default=_env_default("WANDB_PROJECT", "none"))
    parser.add_argument("--wandb-key", default="env")

    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-bundle", action="store_true")

    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_rows: list[dict[str, object]] = []

    def run_step(step_id: str, cmd: list[str]) -> int:
        rc = _run(cmd, dry_run=args.dry_run)
        summary_rows.append({"step_id": step_id, "command": cmd, "returncode": rc})
        return rc

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
        STAGES_125M,
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
        "--seed",
        str(args.seed),
        "--save-milestone-freq",
        str(args.save_milestone_freq),
        "--wandb-entity",
        args.wandb_entity,
        "--wandb-project",
        args.wandb_project,
        "--wandb-key",
        args.wandb_key,
    ]
    if args.global_batch_size is not None:
        cmd_train.extend(["--global-batch-size", str(args.global_batch_size)])
    if args.accum_steps is not None:
        cmd_train.extend(["--accum-steps", str(args.accum_steps)])
    if args.seq_length is not None:
        cmd_train.extend(["--seq-length", str(args.seq_length)])
    if args.allow_missing_fingerprints:
        cmd_train.append("--allow-missing-fingerprints")
    if args.dummy_dataset:
        cmd_train.append("--dummy-dataset")
    if args.skip_existing:
        cmd_train.append("--skip-existing")
    if args.dry_run:
        cmd_train.append("--dry-run")

    if not args.skip_train:
        rc = run_step("train", cmd_train)
        if rc != 0:
            summary_out = (
                Path("./reports/paper") / args.paper_run_id / "launch" / "launcher_summary.json"
            ).resolve()
            _write_json(
                summary_out,
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "wandb_entity": args.wandb_entity,
                    "wandb_project": args.wandb_project,
                    "rows": summary_rows,
                },
            )
            return rc

    if not args.skip_eval:
        cmd_eval = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/34_eval_matrix_jax.py",
            "--paper-run-id",
            args.paper_run_id,
            "--exp-dir",
            str(args.exp_dir),
            "--checkpoint-root",
            str(args.checkpoint_root),
            "--exp-folder",
            args.exp_folder,
            "--contexts",
            args.contexts,
            "--datasets",
            args.datasets,
            "--dclm-root",
            str(args.dclm_root),
            "--books-root",
            str(args.books_root),
            "--eval-split",
            args.eval_split,
            "--eval-batches",
            str(args.eval_batches),
            "--eval-batch-size",
            str(args.eval_batch_size),
        ]
        if args.strict:
            cmd_eval.append("--strict")
        if args.dry_run:
            cmd_eval.append("--dry-run")
        rc = run_step("eval", cmd_eval)
        if rc != 0:
            summary_out = (
                Path("./reports/paper") / args.paper_run_id / "launch" / "launcher_summary.json"
            ).resolve()
            _write_json(
                summary_out,
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "wandb_entity": args.wandb_entity,
                    "wandb_project": args.wandb_project,
                    "rows": summary_rows,
                },
            )
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
            "--s0-stage",
            "S0_125M",
            "--s1-stage",
            "S1_125M",
            "--s2-stage",
            "S2_125M",
            "--s3-stage",
            "S3_125M",
        ]
        rc = run_step("tables", cmd_tables)
        if rc != 0:
            summary_out = (
                Path("./reports/paper") / args.paper_run_id / "launch" / "launcher_summary.json"
            ).resolve()
            _write_json(
                summary_out,
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "wandb_entity": args.wandb_entity,
                    "wandb_project": args.wandb_project,
                    "rows": summary_rows,
                },
            )
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
        rc = run_step("figures", cmd_fig)
        if rc != 0:
            summary_out = (
                Path("./reports/paper") / args.paper_run_id / "launch" / "launcher_summary.json"
            ).resolve()
            _write_json(
                summary_out,
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "wandb_entity": args.wandb_entity,
                    "wandb_project": args.wandb_project,
                    "rows": summary_rows,
                },
            )
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
        rc = run_step("bundle", cmd_bundle)
        if rc != 0:
            summary_out = (
                Path("./reports/paper") / args.paper_run_id / "launch" / "launcher_summary.json"
            ).resolve()
            _write_json(
                summary_out,
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "wandb_entity": args.wandb_entity,
                    "wandb_project": args.wandb_project,
                    "rows": summary_rows,
                },
            )
            return rc

    summary_out = (
        Path("./reports/paper") / args.paper_run_id / "launch" / "launcher_summary.json"
    ).resolve()
    _write_json(
        summary_out,
        {
            "schema_version": "1.0",
            "paper_run_id": args.paper_run_id,
            "wandb_entity": args.wandb_entity,
            "wandb_project": args.wandb_project,
            "rows": summary_rows,
        },
    )

    print("125M parity ladder pipeline completed.")
    print(f"Wrote launcher summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
