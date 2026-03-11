#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from ttt.research.author_checkpoints import load_env_file, select_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw author Orbax artifacts from a Hugging Face repo into a local "
            "directory while preserving the staged checkpoint subdirectories."
        )
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument(
        "--checkpoint",
        default="all",
        help="Comma-separated checkpoint keys or 'all'. Supported: 760m_fa,760m_e2e",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("./artifacts/hf_author_checkpoints"),
    )
    parser.add_argument(
        "--repo-subdir",
        default="",
        help="Optional root subdirectory in the HF repo that contains the staged author checkpoints.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default="", help="Optional HF token. Falls back to env or cached login.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()

    token = args.token.strip() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    dest_root = args.dest_root.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    allow_patterns = []
    repo_subdir = args.repo_subdir.strip("/")
    for spec in select_specs(args.checkpoint):
        prefix = f"{repo_subdir}/{spec.key}/**" if repo_subdir else f"{spec.key}/**"
        allow_patterns.append(prefix)

    print(f"Downloading {args.repo_id} -> {dest_root}")
    print(f"allow_patterns={allow_patterns}")
    if args.dry_run:
        return 0

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        allow_patterns=allow_patterns,
        token=token,
        local_dir=dest_root,
    )
    print(f"Downloaded snapshot to {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
