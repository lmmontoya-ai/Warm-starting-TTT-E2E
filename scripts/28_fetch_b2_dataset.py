#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


DATASETS = {"dclm_filter_8k", "books3"}
DEFAULT_SPLITS = ("train", "val")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _find_aws(*, allow_missing: bool = False) -> str:
    aws = shutil.which("aws")
    if not aws:
        if allow_missing:
            return "aws"
        raise FileNotFoundError("The AWS CLI was not found in PATH.")
    return aws


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _run_capture(cmd: list[str]) -> str:
    print("+", shlex.join(cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def _has_split(dataset_path: Path, split: str) -> bool:
    split_path = dataset_path / split
    if not split_path.exists():
        return False
    if (
        (split_path / ".zarray").exists()
        or (split_path / ".zgroup").exists()
        or (split_path / "zarr.json").exists()
    ):
        return True
    return (split_path / "c").is_dir()


def _validate_local(dataset_root: Path, splits: list[str]) -> bool:
    ok = True
    for split in splits:
        if not _has_split(dataset_root, split):
            print(f"Missing split '{split}' under {dataset_root}")
            ok = False
    return ok


def _parse_csv(raw: str, *, allowed: set[str] | None = None) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if allowed is None:
        return values
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown values: {', '.join(unknown)}")
    return values


def _s3_split_uri(bucket: str, prefix: str, dataset_name: str, split: str) -> str:
    cleaned_prefix = prefix.strip("/")
    if cleaned_prefix:
        return f"s3://{bucket}/{cleaned_prefix}/{dataset_name}/{split}"
    return f"s3://{bucket}/{dataset_name}/{split}"


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _local_split_stats(local_root: Path) -> tuple[int, int]:
    if not local_root.exists():
        return 0, 0

    file_count = 0
    total_bytes = 0
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return file_count, total_bytes


def _remote_split_stats(
    *,
    aws: str,
    remote_uri: str,
    endpoint_url: str,
    region: str | None,
) -> tuple[int, int] | None:
    cmd = [
        aws,
        "s3",
        "ls",
        remote_uri,
        "--recursive",
        "--summarize",
        "--endpoint-url",
        endpoint_url,
    ]
    if region:
        cmd += ["--region", region]

    try:
        output = _run_capture(cmd)
    except subprocess.CalledProcessError:
        return None

    total_files: int | None = None
    total_bytes: int | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total Objects:"):
            total_files = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Total Size:"):
            total_bytes = int(stripped.split(":", 1)[1].strip())

    if total_files is None or total_bytes is None:
        return None
    return total_files, total_bytes


def _render_progress(
    *,
    label: str,
    local_files: int,
    local_bytes: int,
    total_files: int,
    total_bytes: int,
) -> None:
    clamped_files = min(local_files, total_files)
    clamped_bytes = min(local_bytes, total_bytes)
    pct = 100.0 if total_bytes <= 0 else (clamped_bytes / total_bytes) * 100.0
    line = (
        f"\r[{label}] {pct:6.2f}% | "
        f"files {clamped_files}/{total_files} | "
        f"bytes {_human_bytes(clamped_bytes)}/{_human_bytes(total_bytes)}"
    )
    print(line, end="", file=sys.stderr, flush=True)


def _sync_split_from_b2(
    *,
    aws: str,
    remote_uri: str,
    local_root: Path,
    endpoint_url: str,
    region: str | None,
    dry_run: bool,
    show_progress: bool,
    progress_label: str,
) -> None:
    local_root.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        aws,
        "s3",
        "sync",
        remote_uri,
        str(local_root),
        "--endpoint-url",
        endpoint_url,
        "--only-show-errors",
    ]
    if region:
        cmd += ["--region", region]
    if dry_run:
        _run(cmd, dry_run=dry_run)
        return

    remote_stats = None
    if show_progress:
        remote_stats = _remote_split_stats(
            aws=aws,
            remote_uri=remote_uri,
            endpoint_url=endpoint_url,
            region=region,
        )
        if remote_stats is not None:
            total_files, total_bytes = remote_stats
            start_files, start_bytes = _local_split_stats(local_root)
            _render_progress(
                label=progress_label,
                local_files=start_files,
                local_bytes=start_bytes,
                total_files=total_files,
                total_bytes=total_bytes,
            )
            print("", file=sys.stderr)

    print("+", shlex.join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        while True:
            rc = proc.poll()
            if remote_stats is not None:
                total_files, total_bytes = remote_stats
                local_files, local_bytes = _local_split_stats(local_root)
                _render_progress(
                    label=progress_label,
                    local_files=local_files,
                    local_bytes=local_bytes,
                    total_files=total_files,
                    total_bytes=total_bytes,
                )
            if rc is not None:
                break
            time.sleep(1.0)
    finally:
        if remote_stats is not None:
            print("", file=sys.stderr)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download mirrored TTT-E2E datasets from Backblaze B2 into the runtime dataset layout."
    )
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default="dclm_filter_8k,books3",
        help="Comma-separated datasets to fetch (default: dclm_filter_8k,books3).",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split names to fetch (default: train,val).",
    )
    parser.add_argument(
        "--b2-bucket",
        default=None,
        help="Backblaze B2 bucket name. Falls back to B2_BUCKET.",
    )
    parser.add_argument(
        "--b2-prefix",
        default=None,
        help="Backblaze B2 prefix for dataset roots. Falls back to B2_DATASET_PREFIX.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Backblaze S3 endpoint URL. Falls back to B2_ENDPOINT_URL.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional env file to load before resolving credentials.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing local split directories before syncing from B2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable split-level progress reporting and rely on the underlying CLI only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file.expanduser().resolve())

    datasets = _parse_csv(args.datasets, allowed=DATASETS)
    splits = _parse_csv(args.splits)

    bucket = args.b2_bucket or os.environ.get("B2_BUCKET")
    prefix = args.b2_prefix or os.environ.get("B2_DATASET_PREFIX", "ttt-e2e-datasets")
    endpoint_url = args.endpoint_url or os.environ.get("B2_ENDPOINT_URL")
    region = os.environ.get("AWS_DEFAULT_REGION")

    if not bucket:
        raise ValueError("Missing Backblaze bucket. Set B2_BUCKET or pass --b2-bucket.")
    if not endpoint_url:
        raise ValueError("Missing Backblaze endpoint URL. Set B2_ENDPOINT_URL or pass --endpoint-url.")

    aws = _find_aws(allow_missing=args.dry_run)
    dest_root = args.dest_root.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        dataset_root = dest_root / dataset_name
        for split in splits:
            local_split = dataset_root / split
            if local_split.exists() and args.force:
                if args.dry_run:
                    print(f"+ rm -rf {local_split}")
                else:
                    shutil.rmtree(local_split)
            elif local_split.exists():
                print(f"Resuming existing split with aws s3 sync: {local_split}")

            remote_uri = _s3_split_uri(bucket, prefix, dataset_name, split)
            _sync_split_from_b2(
                aws=aws,
                remote_uri=remote_uri,
                local_root=local_split,
                endpoint_url=endpoint_url,
                region=region,
                dry_run=args.dry_run,
                show_progress=not args.no_progress,
                progress_label=f"{dataset_name}/{split}",
            )

        if args.dry_run:
            print(f"Dry run: skipped local validation for {dataset_name}.")
        else:
            if not _validate_local(dataset_root, splits):
                print(f"Validation failed for {dataset_name} under {dataset_root}")
                return 2
            print(f"Validated local dataset root: {dataset_root}")

    print("Backblaze dataset fetch finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
