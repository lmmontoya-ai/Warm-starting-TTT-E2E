#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ttt.research.author_checkpoints import (
    build_base_manifest,
    ensure_dir,
    find_latest_step,
    gcloud_cat,
    gcloud_cp,
    gcloud_du,
    load_env_file,
    manifest_path,
    raw_step_dir,
    select_specs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch author-shared Orbax checkpoints from requester-pays GCS into a "
            "standardized local raw-artifact layout."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="all",
        help="Comma-separated checkpoint keys or 'all'. Supported: 760m_fa,760m_e2e",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("./artifacts/author_checkpoints"),
        help="Local root for standardized author-checkpoint artifacts.",
    )
    parser.add_argument(
        "--billing-project",
        default="",
        help="Requester-pays billing project. Falls back to GCS_BILLING_PROJECT.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional fixed step to fetch. Defaults to latest numeric step per checkpoint.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Write manifests only.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifests.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()

    artifact_root = ensure_dir(args.artifact_root.expanduser().resolve())
    billing_project = args.billing_project.strip() or os.environ.get("GCS_BILLING_PROJECT") or None

    for spec in select_specs(args.checkpoint):
        step = int(args.step) if args.step is not None else find_latest_step(spec, billing_project=billing_project)
        local_step_dir = raw_step_dir(artifact_root, spec, step)
        manifest_file = manifest_path(artifact_root, spec)
        if manifest_file.exists() and not args.force:
            raise FileExistsError(f"Manifest already exists: {manifest_file}. Use --force to overwrite.")

        ensure_dir(local_step_dir.parent)
        remote_step_uri = f"{spec.source_uri}/{step}"
        remote_bytes = gcloud_du(remote_step_uri, billing_project=billing_project)

        if not args.skip_download:
            ensure_dir(local_step_dir.parent)
            gcloud_cp(
                remote_step_uri,
                local_step_dir.parent,
                recursive=True,
                billing_project=billing_project,
                dry_run=args.dry_run,
            )

        root_metadata: dict[str, object] = {}
        if not args.dry_run:
            try:
                root_metadata = {
                    "checkpoint_metadata": gcloud_cat(
                        f"{remote_step_uri}/_CHECKPOINT_METADATA",
                        billing_project=billing_project,
                    )
                }
            except Exception:
                root_metadata = {}

        manifest = build_base_manifest(
            spec=spec,
            step=step,
            billing_project=billing_project,
            remote_bytes=remote_bytes,
            local_step_dir=local_step_dir,
        )
        if root_metadata:
            try:
                manifest["raw_checkpoint_metadata"] = json.loads(root_metadata["checkpoint_metadata"])
            except Exception:
                manifest["raw_checkpoint_metadata_text"] = root_metadata["checkpoint_metadata"]
        manifest["download_status"] = "skipped" if args.skip_download else ("dry_run" if args.dry_run else "completed")
        write_json(manifest_file, manifest)
        print(f"Wrote manifest: {manifest_file}")
        if not args.dry_run:
            print(f"Local raw Orbax step: {local_step_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
