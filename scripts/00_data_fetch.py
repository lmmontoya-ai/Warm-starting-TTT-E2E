#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASETS = {
    "dclm_filter_8k": "gs://llama3-dclm-filter-8k/",
    "books3": "gs://llama3-books3/",
}

SPLITS = ["train", "val"]


def _find_tool() -> tuple[str, list[str]]:
    gcloud = shutil.which("gcloud")
    if gcloud:
        return ("gcloud", [gcloud, "storage", "cp", "-r"])

    gsutil = shutil.which("gsutil")
    if gsutil:
        return ("gsutil", [gsutil, "-m", "cp", "-r"])

    return ("", [])


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _has_zarr_split(dataset_path: Path, split: str) -> bool:
    split_path = dataset_path / split
    if not split_path.exists():
        return False

    # Zarr arrays are usually directories containing .zarray
    if (split_path / ".zarray").exists() or (split_path / ".zgroup").exists():
        return True

    # Fallback: treat any directory as a potential Zarr store
    return split_path.is_dir()


def _check_dataset(dataset_path: Path) -> bool:
    ok = True
    if not dataset_path.exists():
        print(f"Missing dataset path: {dataset_path}")
        return False

    for split in SPLITS:
        if not _has_zarr_split(dataset_path, split):
            print(f"Missing split '{split}' under {dataset_path}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify TTT-E2E datasets (Zarr) to a local or cloud-mounted path."
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        required=True,
        help="Destination root directory (datasets will be placed under this folder).",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS.keys()),
        help=f"Comma-separated dataset list. Options: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument(
        "--billing-project",
        default=None,
        help="GCP billing project for Requester Pays buckets (if needed).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify local dataset layout, do not download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target path exists.",
    )
    args = parser.parse_args()

    dest_root = args.dest_root.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in selected:
        if d not in DATASETS:
            print(f"Unknown dataset: {d}")
            return 2

    if not args.check_only:
        tool_name, base_cmd = _find_tool()
        if not base_cmd:
            print("Neither 'gcloud' nor 'gsutil' was found in PATH.")
            return 3

        for d in selected:
            target = dest_root / d
            if target.exists() and any(target.iterdir()) and not args.force:
                print(f"Skipping download (already exists): {target}")
                continue

            bucket = DATASETS[d]
            if tool_name == "gcloud":
                cmd = base_cmd[:]
                if args.billing_project:
                    cmd += ["--billing-project", args.billing_project]
                cmd += [bucket, str(target)]
            else:
                cmd = base_cmd[:]
                if args.billing_project:
                    cmd += ["-u", args.billing_project]
                cmd += [bucket, str(target)]

            _run(cmd)

    print("\n=== Dataset Check ===")
    ok = True
    for d in selected:
        target = dest_root / d
        if not _check_dataset(target):
            ok = False
        else:
            print(f"OK: {d} @ {target}")

    if not ok:
        print("\nOne or more datasets failed validation.")
        return 4

    print("\nAll datasets look good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
