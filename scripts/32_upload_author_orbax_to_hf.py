#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from ttt.research.author_checkpoints import (
    artifact_root,
    load_env_file,
    manifest_path,
    select_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload standardized raw author Orbax artifacts to a Hugging Face repo. "
            "This preserves the raw checkpoint tree plus local manifests."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="all",
        help="Comma-separated checkpoint keys or 'all'. Supported: 760m_fa,760m_e2e",
    )
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face repo id, e.g. user/repo.")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("./artifacts/author_checkpoints"),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--staging-subdir",
        default="author_checkpoints",
        help="Folder name used in the temporary upload bundle root.",
    )
    parser.add_argument("--token", default="", help="Optional HF token. Falls back to env or cached login.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()

    artifact_root_path = args.artifact_root.expanduser().resolve()
    token = args.token.strip() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    api = HfApi(token=token)

    if not args.dry_run:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=bool(args.private),
            exist_ok=True,
        )

    selected = select_specs(args.checkpoint)
    for spec in selected:
        manifest_file = manifest_path(artifact_root_path, spec)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Missing artifact manifest: {manifest_file}")

    with tempfile.TemporaryDirectory(prefix="hf_author_orbax_") as tmp_dir:
        bundle_root = Path(tmp_dir) / args.staging_subdir
        bundle_root.mkdir(parents=True, exist_ok=True)
        for spec in selected:
            local_dir = artifact_root(artifact_root_path, spec)
            target = bundle_root / spec.key
            print(f"Staging {local_dir} -> {target}")
            if not args.dry_run:
                shutil.copytree(local_dir, target, dirs_exist_ok=False)

        print(f"Uploading bundle {bundle_root} -> {args.repo_id}")
        if not args.dry_run:
            api.upload_large_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                folder_path=bundle_root,
                private=bool(args.private),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
