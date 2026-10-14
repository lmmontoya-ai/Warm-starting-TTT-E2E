#!/usr/bin/env python3
"""Restore one exported ladder stage into the local experiment/checkpoint layout.

Reads the stage-export layout written by scripts/40_export_stage.py:

    <paper_run_id>/stages/<stage_id>/<run_id>/
        experiment/  checkpoint/latest.json  checkpoint/step_metadata_*.json  checkpoint/<step>/

from exactly one source:

    --repo-id USER/REPO   a Hugging Face repo holding exports
    --source-dir DIR      a local directory holding exports (extract a shared
                          tarball first: tar -xzf bundle.tar.gz -C DIR)

and writes

    <exp-dir>/<target-paper-run-id>/<target-stage-id>/<target-run-id>/   (experiment metadata)
    <checkpoint-root>/<target-paper-run-id>/<target-run-id>/             (latest.json + step dir)

plus <exp-dir>/.../restore_manifest.json recording the source. The warm-start
registry and scripts/96_run_revision_v3_ladder.py look for parents at
<checkpoint-root>/<exp_folder>/<run_id>, so restore seeds with
--target-paper-run-id set to the ladder's paper run id.

Example (125M FA seed from a local bundle into the revision-v3 ladder):

    python scripts/46_restore_stage.py --source-dir ./stage_bundle \\
        --source-paper-run-id protocol_r_125m_main_v1 \\
        --source-stage-id S0_PRETRAIN_FA_125M --source-run-id pretrain-125m-fa \\
        --target-paper-run-id revision_v3_widthpreserving_v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from ttt.research.author_checkpoints import load_env_file


RESTORE_MANIFEST_NAME = "restore_manifest.json"
PARAMS_ONLY_STEP_ENTRIES = ("_CHECKPOINT_METADATA", "model_weights")
EXPECTED_LAYOUT = (
    "<source-dir>/<paper_run_id>/stages/<stage_id>/<run_id>/"
    "{experiment/, checkpoint/latest.json, checkpoint/<step>/}"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_prefix(paper_run_id: str, stage_id: str, run_id: str) -> str:
    return f"{paper_run_id}/stages/{stage_id}/{run_id}"


def hf_allow_patterns(prefix: str, *, params_only: bool) -> list[str]:
    if not params_only:
        return [f"{prefix}/**"]
    return [
        f"{prefix}/experiment/**",
        f"{prefix}/checkpoint/latest.json",
        f"{prefix}/checkpoint/step_metadata_*.json",
        f"{prefix}/checkpoint/*/_CHECKPOINT_METADATA",
        f"{prefix}/checkpoint/*/model_weights/**",
        f"{prefix}/hf_export_manifest.json",
        f"{prefix}/export_manifest.json",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-id", default=None, help="Hugging Face repo holding stage exports.")
    source.add_argument("--source-dir", type=Path, default=None, help="Local directory holding stage exports.")
    parser.add_argument("--token", default="", help="HF token (else HF_TOKEN / HUGGINGFACE_HUB_TOKEN / cached login).")
    parser.add_argument("--source-paper-run-id", required=True)
    parser.add_argument("--source-stage-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--target-paper-run-id", required=True)
    parser.add_argument("--target-stage-id", default="")
    parser.add_argument("--target-run-id", default="")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--params-only",
        action="store_true",
        help=(
            "Restore experiment metadata and model_weights only. This is intended "
            "for warm starts and inference-only runs and deliberately omits optimizer state."
        ),
    )
    parser.add_argument(
        "--all-model-checkpoints",
        action="store_true",
        help=(
            "With --params-only, retain every exported numeric model_weights "
            "checkpoint instead of only latest. Used for descriptive T1b trajectories."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_step_dir(src: Path, dst: Path, *, params_only: bool, overwrite: bool) -> None:
    if not params_only:
        _copy_tree(src, dst, overwrite=overwrite)
        return
    if not (src / "model_weights").exists():
        raise FileNotFoundError(f"Params-only restore is missing model_weights: {src / 'model_weights'}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in PARAMS_ONLY_STEP_ENTRIES:
        entry = src / name
        if entry.is_dir():
            shutil.copytree(entry, dst / name)
        elif entry.exists():
            shutil.copy2(entry, dst / name)


def restore_from_stage_root(
    *,
    stage_root: Path,
    exp_target: Path,
    checkpoint_target: Path,
    params_only: bool,
    all_model_checkpoints: bool,
    overwrite: bool,
) -> dict:
    """Copy one stage export into the experiment/checkpoint targets; return what was restored."""

    experiment_src = stage_root / "experiment"
    checkpoint_src = stage_root / "checkpoint"
    latest_json = checkpoint_src / "latest.json"
    if not experiment_src.exists():
        raise FileNotFoundError(f"Missing experiment subtree in stage export: {experiment_src}")
    if not latest_json.exists():
        raise FileNotFoundError(f"Missing latest.json in stage export: {latest_json}")
    latest_step = int(_load_json(latest_json)["step"])
    step_dir_src = checkpoint_src / str(latest_step)
    if not step_dir_src.exists():
        raise FileNotFoundError(f"Missing latest checkpoint step dir in stage export: {step_dir_src}")
    if params_only and not (step_dir_src / "model_weights").exists():
        raise FileNotFoundError(f"Params-only restore is missing model_weights: {step_dir_src / 'model_weights'}")
    step_metadata_name = f"step_metadata_{latest_step:08d}.json"

    exp_target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_target.exists():
        if not overwrite:
            raise FileExistsError(f"Target checkpoint directory already exists: {checkpoint_target}")
        shutil.rmtree(checkpoint_target)
    _copy_tree(experiment_src, exp_target, overwrite=overwrite)
    checkpoint_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest_json, checkpoint_target / "latest.json")
    if all_model_checkpoints:
        step_dirs = sorted(
            (path for path in checkpoint_src.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        if not step_dirs:
            raise FileNotFoundError("No numeric model checkpoint directories in the stage export")
        for metadata_src in checkpoint_src.glob("step_metadata_*.json"):
            shutil.copy2(metadata_src, checkpoint_target / metadata_src.name)
        for source_step_dir in step_dirs:
            _copy_step_dir(
                source_step_dir,
                checkpoint_target / source_step_dir.name,
                params_only=True,
                overwrite=overwrite,
            )
    else:
        if (checkpoint_src / step_metadata_name).exists():
            shutil.copy2(checkpoint_src / step_metadata_name, checkpoint_target / step_metadata_name)
        _copy_step_dir(
            step_dir_src,
            checkpoint_target / str(latest_step),
            params_only=params_only,
            overwrite=overwrite,
        )

    return {
        "latest_step": latest_step,
        "restored_checkpoint_steps": sorted(
            int(path.name) for path in checkpoint_target.iterdir() if path.is_dir() and path.name.isdigit()
        ),
        "restored_files": {
            "experiment_dir": str(exp_target),
            "checkpoint_latest": str(checkpoint_target / "latest.json"),
            "checkpoint_step_dir": str(checkpoint_target / str(latest_step)),
            "step_metadata": str(checkpoint_target / step_metadata_name),
        },
    }


def _local_stage_root(source_dir: Path, prefix: str) -> Path:
    root = source_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"--source-dir {root} is not a directory. Extract a stage bundle first "
            "(tar -xzf bundle.tar.gz -C DIR) and pass DIR."
        )
    stage_root = root / prefix
    if not stage_root.is_dir():
        raise FileNotFoundError(f"No stage export at {stage_root}. Expected layout: {EXPECTED_LAYOUT}")
    return stage_root


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args(argv)
    if args.all_model_checkpoints and not args.params_only:
        raise ValueError("--all-model-checkpoints requires --params-only")

    target_stage_id = args.target_stage_id.strip() or args.source_stage_id
    target_run_id = args.target_run_id.strip() or args.source_run_id
    exp_target = args.exp_dir.expanduser().resolve() / args.target_paper_run_id / target_stage_id / target_run_id
    checkpoint_target = args.checkpoint_root.expanduser().resolve() / args.target_paper_run_id / target_run_id
    prefix = stage_prefix(args.source_paper_run_id, args.source_stage_id, args.source_run_id)

    if args.source_dir is not None:
        source = {"kind": "local", "source_dir": str(args.source_dir.expanduser().resolve())}
    else:
        source = {
            "kind": "hf",
            "repo_id": args.repo_id,
            "repo_type": args.repo_type,
            "revision": args.revision,
            "allow_patterns": hf_allow_patterns(prefix, params_only=args.params_only),
        }
    restore_manifest = {
        "schema_version": "1.1",
        "source": source,
        "repo_id": args.repo_id,
        "source_paper_run_id": args.source_paper_run_id,
        "source_stage_id": args.source_stage_id,
        "source_run_id": args.source_run_id,
        "target_paper_run_id": args.target_paper_run_id,
        "target_stage_id": target_stage_id,
        "target_run_id": target_run_id,
        "stage_prefix": prefix,
        "params_only": bool(args.params_only),
        "all_model_checkpoints": bool(args.all_model_checkpoints),
        "exp_target": str(exp_target),
        "checkpoint_target": str(checkpoint_target),
    }

    if args.source_dir is not None:
        stage_root = _local_stage_root(args.source_dir, prefix)
        if args.dry_run:
            print(json.dumps(restore_manifest, indent=2, sort_keys=True))
            return 0
        restore_manifest.update(
            restore_from_stage_root(
                stage_root=stage_root,
                exp_target=exp_target,
                checkpoint_target=checkpoint_target,
                params_only=args.params_only,
                all_model_checkpoints=args.all_model_checkpoints,
                overwrite=args.overwrite,
            )
        )
    else:
        if args.dry_run:
            print(json.dumps(restore_manifest, indent=2, sort_keys=True))
            return 0
        from huggingface_hub import snapshot_download

        token = (
            args.token.strip()
            or os.environ.get("HF_TOKEN", "").strip()
            or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
            or None
        )
        with tempfile.TemporaryDirectory(prefix="hf_stage_restore_") as tmp:
            snapshot_root = Path(tmp) / "snapshot"
            snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                allow_patterns=source["allow_patterns"],
                token=token,
                local_dir=snapshot_root,
            )
            restore_manifest.update(
                restore_from_stage_root(
                    stage_root=snapshot_root / prefix,
                    exp_target=exp_target,
                    checkpoint_target=checkpoint_target,
                    params_only=args.params_only,
                    all_model_checkpoints=args.all_model_checkpoints,
                    overwrite=args.overwrite,
                )
            )

    _write_json(exp_target / RESTORE_MANIFEST_NAME, restore_manifest)
    print(json.dumps(restore_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
