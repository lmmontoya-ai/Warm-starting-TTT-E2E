#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from ttt.research.author_checkpoints import load_env_file


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore one exported ladder stage from HF into the local experiment/checkpoint "
            "layout expected by the warm-start registry."
        )
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default="")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()

    token = (
        args.token.strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        or None
    )
    target_stage_id = args.target_stage_id.strip() or args.source_stage_id
    target_run_id = args.target_run_id.strip() or args.source_run_id

    exp_target = (
        args.exp_dir.expanduser().resolve()
        / args.target_paper_run_id
        / target_stage_id
        / target_run_id
    )
    checkpoint_target = (
        args.checkpoint_root.expanduser().resolve()
        / args.target_paper_run_id
        / target_run_id
    )

    stage_prefix = (
        f"{args.source_paper_run_id}/stages/{args.source_stage_id}/{args.source_run_id}"
    )
    allow_patterns = [f"{stage_prefix}/**"]

    restore_manifest = {
        "schema_version": "1.0",
        "repo_id": args.repo_id,
        "source_paper_run_id": args.source_paper_run_id,
        "source_stage_id": args.source_stage_id,
        "source_run_id": args.source_run_id,
        "target_paper_run_id": args.target_paper_run_id,
        "target_stage_id": target_stage_id,
        "target_run_id": target_run_id,
        "stage_prefix": stage_prefix,
        "allow_patterns": allow_patterns,
        "exp_target": str(exp_target),
        "checkpoint_target": str(checkpoint_target),
    }
    if args.dry_run:
        print(json.dumps(restore_manifest, indent=2, sort_keys=True))
        return 0

    with tempfile.TemporaryDirectory(prefix="hf_stage_restore_") as tmp:
        local_snapshot_root = Path(tmp) / "snapshot"
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            allow_patterns=allow_patterns,
            token=token,
            local_dir=local_snapshot_root,
        )

        stage_root = local_snapshot_root / stage_prefix
        experiment_src = stage_root / "experiment"
        checkpoint_src = stage_root / "checkpoint"
        latest_json = checkpoint_src / "latest.json"
        if not experiment_src.exists():
            raise FileNotFoundError(f"Missing experiment subtree in HF snapshot: {experiment_src}")
        if not latest_json.exists():
            raise FileNotFoundError(f"Missing latest.json in HF snapshot: {latest_json}")

        latest_payload = _load_json(latest_json)
        latest_step = int(latest_payload["step"])
        step_dir_src = checkpoint_src / str(latest_step)
        if not step_dir_src.exists():
            raise FileNotFoundError(f"Missing latest checkpoint step dir in HF snapshot: {step_dir_src}")

        step_metadata_name = f"step_metadata_{latest_step:08d}.json"
        step_metadata_src = checkpoint_src / step_metadata_name

        exp_target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_target.exists():
            if not args.overwrite:
                raise FileExistsError(f"Target checkpoint directory already exists: {checkpoint_target}")
            shutil.rmtree(checkpoint_target)
        _copy_tree(experiment_src, exp_target, overwrite=args.overwrite)
        checkpoint_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_json, checkpoint_target / "latest.json")
        if step_metadata_src.exists():
            shutil.copy2(step_metadata_src, checkpoint_target / step_metadata_name)
        _copy_tree(step_dir_src, checkpoint_target / str(latest_step), overwrite=args.overwrite)

        restore_manifest.update(
            {
                "latest_step": latest_step,
                "restored_files": {
                    "experiment_dir": str(exp_target),
                    "checkpoint_latest": str(checkpoint_target / "latest.json"),
                    "checkpoint_step_dir": str(checkpoint_target / str(latest_step)),
                    "step_metadata": str(checkpoint_target / step_metadata_name),
                },
            }
        )
        _write_json(exp_target / "hf_restore_manifest.json", restore_manifest)

    print(json.dumps(restore_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
