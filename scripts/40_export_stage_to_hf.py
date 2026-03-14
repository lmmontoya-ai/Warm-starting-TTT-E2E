#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a completed ladder stage to HF from the pod.")
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--require-eval-success", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = args.token or None
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    experiment_dir = (args.exp_dir / args.paper_run_id / args.stage_id / args.run_id).expanduser().resolve()
    checkpoint_dir = (args.checkpoint_root / args.paper_run_id / args.run_id).expanduser().resolve()
    run_result_path = experiment_dir / "run_result.json"
    if not run_result_path.exists():
        raise FileNotFoundError(f"Missing run_result.json for stage export: {run_result_path}")
    run_result = _load_json(run_result_path)
    status = str(run_result.get("status", "unknown"))
    if status not in {"succeeded", "dry_run"}:
        raise ValueError(f"Refusing to export stage with status={status}: {run_result_path}")

    eval_manifest_path = experiment_dir / "eval_manifest.json"
    if args.require_eval_success:
        if not eval_manifest_path.exists():
            raise FileNotFoundError(f"Missing eval_manifest.json for strict stage export: {eval_manifest_path}")
        eval_manifest = _load_json(eval_manifest_path)
        eval_status = str(eval_manifest.get("status", "unknown")).strip()
        if eval_status != "succeeded":
            raise ValueError(
                f"Refusing to export stage without successful eval manifest status={eval_status}: {eval_manifest_path}"
            )

    latest_json = checkpoint_dir / "latest.json"
    if not latest_json.exists():
        raise FileNotFoundError(f"Missing latest.json for checkpoint export: {latest_json}")
    latest_payload = _load_json(latest_json)
    latest_step = int(latest_payload["step"])
    step_dir = checkpoint_dir / str(latest_step)
    if not step_dir.exists():
        raise FileNotFoundError(f"Missing latest step directory: {step_dir}")

    path_prefix = f"{args.paper_run_id}/stages/{args.stage_id}/{args.run_id}"
    uploaded_files: list[str] = []
    for rel_name in EXPERIMENT_FILES:
        src = experiment_dir / rel_name
        if not src.exists():
            continue
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(src),
            path_in_repo=f"{path_prefix}/experiment/{rel_name}",
            commit_message=f"Upload {args.paper_run_id} {args.stage_id}/{args.run_id} experiment artifacts",
        )
        uploaded_files.append(f"experiment/{rel_name}")

    latest_step_metadata = checkpoint_dir / f"step_metadata_{latest_step:08d}.json"
    for src, rel_name in [
        (latest_json, "checkpoint/latest.json"),
        (latest_step_metadata, f"checkpoint/step_metadata_{latest_step:08d}.json"),
    ]:
        if not src.exists():
            continue
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(src),
            path_in_repo=f"{path_prefix}/{rel_name}",
            commit_message=f"Upload {args.paper_run_id} {args.stage_id}/{args.run_id} checkpoint sidecars",
        )
        uploaded_files.append(rel_name)

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(step_dir),
        path_in_repo=f"{path_prefix}/checkpoint/{latest_step}",
        commit_message=f"Upload {args.paper_run_id} {args.stage_id}/{args.run_id} checkpoint step {latest_step}",
    )
    uploaded_files.append(f"checkpoint/{latest_step}/")

    export_manifest = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "stage_id": args.stage_id,
        "run_id": args.run_id,
        "repo_id": args.repo_id,
        "latest_step": latest_step,
        "uploaded_files": uploaded_files,
    }
    export_manifest_path = experiment_dir / "hf_export_manifest.json"
    _write_json(export_manifest_path, export_manifest)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(export_manifest_path),
        path_in_repo=f"{path_prefix}/hf_export_manifest.json",
        commit_message=f"Upload {args.paper_run_id} {args.stage_id}/{args.run_id} export manifest",
    )
    print(json.dumps(export_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
