#!/usr/bin/env python3
"""Export one completed ladder stage to a Hugging Face repo or a local directory.

Both destinations use the same stage-export layout, which
scripts/46_restore_stage.py reads back:

    <paper_run_id>/stages/<stage_id>/<run_id>/
        experiment/            run metadata (run_result.json, resolved_config.yaml, metrics, ...)
        checkpoint/latest.json
        checkpoint/step_metadata_<step>.json
        checkpoint/<step>/     the latest Orbax step (model_weights/, opt_state/, ...)

Exactly one destination:

    --repo-id USER/REPO   upload to a Hugging Face model repo you control (created if missing)
    --dest-dir DIR        write the tree under DIR, e.g. to tar it up and share it

Example (local bundle of a pretrained seed, weights only):

    python scripts/40_export_stage.py --paper-run-id protocol_r_125m_main_v1 \\
        --stage-id S0_PRETRAIN_FA_125M --run-id pretrain-125m-fa \\
        --dest-dir ./stage_bundle --params-only
    tar -C ./stage_bundle -czf seed_bundle.tar.gz protocol_r_125m_main_v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from ttt.research.author_checkpoints import load_env_file


EXPERIMENT_FILES = [
    "command.sh",
    "environment_manifest.json",
    "events.jsonl",
    "metrics.jsonl",
    "resolved_config.yaml",
    "unresolved_config.yaml",
    "phase1_resolved_config.yaml",
    "phase1_unresolved_config.yaml",
    "run_manifest.json",
    "run_result.json",
    "stage_manifest.json",
    "checkpoint_manifest.json",
    "budget_manifest.json",
    "eval_manifest.json",
    "per_position_nll.npy",
    "loss_curve.npy",
    "token_nll_curve.npy",
]

# Entries of an Orbax step directory needed to restore model weights only.
PARAMS_ONLY_STEP_ENTRIES = ("_CHECKPOINT_METADATA", "model_weights")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_prefix(paper_run_id: str, stage_id: str, run_id: str) -> str:
    return f"{paper_run_id}/stages/{stage_id}/{run_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--run-id", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--repo-id", default=None, help="Hugging Face model repo to upload to.")
    destination.add_argument("--dest-dir", type=Path, default=None, help="Local directory to write the export tree into.")
    parser.add_argument("--token", default="", help="HF token (else HF_TOKEN / HUGGINGFACE_HUB_TOKEN / cached login).")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument(
        "--checkpoint-exp-folder",
        default="",
        help="Checkpoint folder under --checkpoint-root. Defaults to --paper-run-id.",
    )
    parser.add_argument(
        "--params-only",
        action="store_true",
        help="Export only model weights from the checkpoint step (no optimizer state).",
    )
    parser.add_argument("--overwrite", action="store_true", help="--dest-dir only: replace an existing export.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional env file with HF_TOKEN.")
    parser.add_argument("--require-eval-success", action="store_true")
    return parser.parse_args(argv)


def collect_stage(args: argparse.Namespace) -> dict:
    """Validate the local run and list what the export contains."""

    experiment_dir = (args.exp_dir / args.paper_run_id / args.stage_id / args.run_id).expanduser().resolve()
    checkpoint_exp_folder = args.checkpoint_exp_folder.strip() or args.paper_run_id
    checkpoint_dir = (args.checkpoint_root / checkpoint_exp_folder / args.run_id).expanduser().resolve()
    run_result_path = experiment_dir / "run_result.json"
    if not run_result_path.exists():
        raise FileNotFoundError(f"Missing run_result.json for stage export: {run_result_path}")
    status = str(_load_json(run_result_path).get("status", "unknown"))
    if status not in {"succeeded", "dry_run"}:
        raise ValueError(f"Refusing to export stage with status={status}: {run_result_path}")

    if args.require_eval_success:
        eval_manifest_path = experiment_dir / "eval_manifest.json"
        if not eval_manifest_path.exists():
            raise FileNotFoundError(f"Missing eval_manifest.json for strict stage export: {eval_manifest_path}")
        eval_status = str(_load_json(eval_manifest_path).get("status", "unknown")).strip()
        if eval_status != "succeeded":
            raise ValueError(
                f"Refusing to export stage without successful eval manifest status={eval_status}: {eval_manifest_path}"
            )

    latest_json = checkpoint_dir / "latest.json"
    if not latest_json.exists():
        raise FileNotFoundError(f"Missing latest.json for checkpoint export: {latest_json}")
    latest_step = int(_load_json(latest_json)["step"])
    step_dir = checkpoint_dir / str(latest_step)
    if not step_dir.exists():
        raise FileNotFoundError(f"Missing latest step directory: {step_dir}")
    if args.params_only and not (step_dir / "model_weights").exists():
        raise FileNotFoundError(f"--params-only export needs {step_dir / 'model_weights'}")

    step_entries = sorted(path.name for path in step_dir.iterdir())
    if args.params_only:
        step_entries = [name for name in step_entries if name in PARAMS_ONLY_STEP_ENTRIES]
    return {
        "experiment_dir": experiment_dir,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_exp_folder": checkpoint_exp_folder,
        "latest_json": latest_json,
        "latest_step": latest_step,
        "step_dir": step_dir,
        "step_entries": step_entries,
        "step_metadata": checkpoint_dir / f"step_metadata_{latest_step:08d}.json",
        "experiment_files": [name for name in EXPERIMENT_FILES if (experiment_dir / name).exists()],
    }


def _base_manifest(args: argparse.Namespace, stage: dict, destination: dict) -> dict:
    return {
        "schema_version": "1.1",
        "paper_run_id": args.paper_run_id,
        "stage_id": args.stage_id,
        "run_id": args.run_id,
        "destination": destination,
        "repo_id": destination.get("repo_id"),
        "checkpoint_exp_folder": stage["checkpoint_exp_folder"],
        "latest_step": stage["latest_step"],
        "params_only": bool(args.params_only),
    }


def export_to_dir(args: argparse.Namespace, stage: dict) -> dict:
    dest_root = args.dest_dir.expanduser().resolve()
    stage_root = dest_root / stage_prefix(args.paper_run_id, args.stage_id, args.run_id)
    if stage_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Export already exists: {stage_root} (use --overwrite)")
        shutil.rmtree(stage_root)
    exported: list[str] = []
    for name in stage["experiment_files"]:
        target = stage_root / "experiment" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage["experiment_dir"] / name, target)
        exported.append(f"experiment/{name}")
    checkpoint_root = stage_root / "checkpoint"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage["latest_json"], checkpoint_root / "latest.json")
    exported.append("checkpoint/latest.json")
    if stage["step_metadata"].exists():
        shutil.copy2(stage["step_metadata"], checkpoint_root / stage["step_metadata"].name)
        exported.append(f"checkpoint/{stage['step_metadata'].name}")
    step_target = checkpoint_root / str(stage["latest_step"])
    step_target.mkdir(parents=True, exist_ok=True)
    for name in stage["step_entries"]:
        source = stage["step_dir"] / name
        if source.is_dir():
            shutil.copytree(source, step_target / name)
        else:
            shutil.copy2(source, step_target / name)
    exported.append(f"checkpoint/{stage['latest_step']}/")
    manifest = _base_manifest(args, stage, {"kind": "local", "dest_dir": str(dest_root)})
    manifest["exported_files"] = exported
    _write_json(stage_root / "export_manifest.json", manifest)
    return manifest


def export_to_hf(args: argparse.Namespace, stage: dict) -> dict:
    from huggingface_hub import HfApi

    token = (
        args.token.strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        or None
    )
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    prefix = stage_prefix(args.paper_run_id, args.stage_id, args.run_id)
    label = f"{args.paper_run_id} {args.stage_id}/{args.run_id}"
    uploaded: list[str] = []
    for name in stage["experiment_files"]:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(stage["experiment_dir"] / name),
            path_in_repo=f"{prefix}/experiment/{name}",
            commit_message=f"Upload {label} experiment artifacts",
        )
        uploaded.append(f"experiment/{name}")
    for src, rel_name in [
        (stage["latest_json"], "checkpoint/latest.json"),
        (stage["step_metadata"], f"checkpoint/{stage['step_metadata'].name}"),
    ]:
        if not src.exists():
            continue
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(src),
            path_in_repo=f"{prefix}/{rel_name}",
            commit_message=f"Upload {label} checkpoint sidecars",
        )
        uploaded.append(rel_name)
    allow_patterns = None
    if args.params_only:
        allow_patterns = ["_CHECKPOINT_METADATA", "model_weights/*"]
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(stage["step_dir"]),
        path_in_repo=f"{prefix}/checkpoint/{stage['latest_step']}",
        allow_patterns=allow_patterns,
        commit_message=f"Upload {label} checkpoint step {stage['latest_step']}",
    )
    uploaded.append(f"checkpoint/{stage['latest_step']}/")
    manifest = _base_manifest(args, stage, {"kind": "hf", "repo_id": args.repo_id})
    manifest["uploaded_files"] = uploaded
    manifest_path = stage["experiment_dir"] / "hf_export_manifest.json"
    _write_json(manifest_path, manifest)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(manifest_path),
        path_in_repo=f"{prefix}/hf_export_manifest.json",
        commit_message=f"Upload {label} export manifest",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file((repo_root / args.env_file) if not args.env_file.is_absolute() else args.env_file)
    stage = collect_stage(args)
    if args.dest_dir is not None:
        manifest = export_to_dir(args, stage)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(
            f"Exported to {args.dest_dir.expanduser().resolve()}; share it with e.g. "
            f"tar -C {args.dest_dir} -czf stage_bundle.tar.gz {args.paper_run_id}"
        )
    else:
        manifest = export_to_hf(args, stage)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
