#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


DATASETS = {
    "dclm_filter_8k": "gs://llama3-dclm-filter-8k/data.zarr",
    "books3": "gs://llama3-books3/data.zarr",
}

PACKAGE_TOKEN_TARGETS = {
    "125m": {
        "dclm_filter_8k": 2_520_000_000,
        "books3": 126_000_000,
    },
    "760m": {
        "dclm_filter_8k": 15_200_000_000,
        "books3": 759_000_000,
    },
}

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


def _find_gcs_tool(*, preferred: str | None = None, allow_missing: bool = False) -> tuple[str, list[str]]:
    gcloud = shutil.which("gcloud")
    gsutil = shutil.which("gsutil")

    if preferred == "gcloud":
        if gcloud:
            return ("gcloud", [gcloud, "storage"])
        raise FileNotFoundError("Requested gcloud, but 'gcloud' was not found in PATH.")

    if preferred == "gsutil":
        if gsutil:
            return ("gsutil", [gsutil])
        raise FileNotFoundError("Requested gsutil, but 'gsutil' was not found in PATH.")

    if gcloud:
        return ("gcloud", [gcloud, "storage"])

    if gsutil:
        return ("gsutil", [gsutil])

    if allow_missing:
        return ("gcloud", ["gcloud", "storage"])

    raise FileNotFoundError("Neither 'gcloud' nor 'gsutil' was found in PATH.")


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


def _gcs_cp(
    *,
    tool_name: str,
    base_cmd: list[str],
    sources: list[str],
    dest: str,
    billing_project: str | None,
    recursive: bool,
    dry_run: bool,
) -> None:
    if tool_name == "gcloud":
        cmd = base_cmd[:] + ["cp"]
        if recursive:
            cmd.append("-r")
        if billing_project:
            cmd += ["--billing-project", billing_project]
        cmd += sources + [dest]
    else:
        cmd = base_cmd[:]
        if recursive:
            recursive_args = ["-r"]
        else:
            recursive_args = []
        if billing_project:
            cmd += ["-u", billing_project]
        cmd += ["cp"] + recursive_args
        cmd += sources + [dest]
    _run(cmd, dry_run=dry_run)


def _gcs_cat(
    *,
    tool_name: str,
    base_cmd: list[str],
    uri: str,
    billing_project: str | None,
) -> str:
    if tool_name == "gcloud":
        cmd = base_cmd[:] + ["cat", uri]
        if billing_project:
            cmd += ["--billing-project", billing_project]
    else:
        cmd = base_cmd[:]
        if billing_project:
            cmd += ["-u", billing_project]
        cmd += ["cat", uri]
    return _run_capture(cmd)


def _has_zarr_split(dataset_path: Path, split: str) -> bool:
    split_path = dataset_path / split
    if not split_path.exists():
        return False
    return (split_path / "zarr.json").exists() or (split_path / ".zarray").exists() or split_path.is_dir()


def _validate_local(dataset_path: Path, splits: list[str]) -> bool:
    ok = True
    if not dataset_path.exists():
        print(f"Missing dataset path: {dataset_path}")
        return False
    for split in splits:
        if not _has_zarr_split(dataset_path, split):
            print(f"Missing split '{split}' under {dataset_path}")
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


def _s3_uri(bucket: str, prefix: str) -> str:
    cleaned_prefix = prefix.strip("/")
    if cleaned_prefix:
        return f"s3://{bucket}/{cleaned_prefix}"
    return f"s3://{bucket}"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_staging_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return _ensure_dir(explicit.expanduser().resolve())

    candidates: list[Path] = []
    env_value = os.environ.get("DATA_STAGING_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(Path.home() / "ttt-e2e-datasets")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _ensure_dir(candidate.resolve())
        except OSError as exc:
            last_error = exc
            print(f"Skipping unusable staging root candidate {candidate}: {exc}")

    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not resolve a writable staging root.")


def _upload_path(
    *,
    aws: str,
    local_root: Path,
    bucket: str,
    prefix: str,
    endpoint_url: str,
    region: str | None,
    dry_run: bool,
    delete: bool,
) -> None:
    remote_root = _s3_uri(bucket, prefix)
    cmd = [
        aws,
        "s3",
        "sync",
        str(local_root),
        remote_root,
        "--endpoint-url",
        endpoint_url,
        "--only-show-errors",
    ]
    if region:
        cmd += ["--region", region]
    if delete:
        cmd.append("--delete")
    _run(cmd, dry_run=dry_run)


def _write_json(path: Path, payload: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        print(f"+ write-json {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_remote_json(
    *,
    tool_name: str,
    base_cmd: list[str],
    uri: str,
    billing_project: str | None,
) -> dict[str, Any]:
    return json.loads(
        _gcs_cat(
            tool_name=tool_name,
            base_cmd=base_cmd,
            uri=uri,
            billing_project=billing_project,
        )
    )


def _infer_shape_len(payload: dict[str, Any]) -> int | None:
    shape = payload.get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        return int(shape[0])
    return None


def _infer_chunk_len(payload: dict[str, Any]) -> int | None:
    chunk_grid = payload.get("chunk_grid")
    if isinstance(chunk_grid, dict):
        chunk_shape = chunk_grid.get("configuration", {}).get("chunk_shape")
        if isinstance(chunk_shape, list) and len(chunk_shape) == 1:
            return int(chunk_shape[0])

    chunks = payload.get("chunks")
    if isinstance(chunks, list) and len(chunks) == 1:
        return int(chunks[0])
    return None


def _batched(values: Iterable[int], batch_size: int) -> Iterable[list[int]]:
    batch: list[int] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_split_metadata(source_meta: dict[str, Any], target_shape: int) -> dict[str, Any]:
    payload = json.loads(json.dumps(source_meta))
    payload["shape"] = [int(target_shape)]
    return payload


def _copy_chunk_batch(
    *,
    tool_name: str,
    base_cmd: list[str],
    chunk_uris: list[str],
    local_chunk_dir: Path,
    billing_project: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"+ mkdir -p {local_chunk_dir}")
    else:
        local_chunk_dir.mkdir(parents=True, exist_ok=True)

    total = len(chunk_uris)
    for index, chunk_uri in enumerate(chunk_uris, start=1):
        chunk_name = chunk_uri.rstrip("/").rsplit("/", 1)[-1]
        print(f"Copying chunk {index}/{total}: {chunk_uri}")
        _gcs_cp(
            tool_name=tool_name,
            base_cmd=base_cmd,
            sources=[chunk_uri],
            dest=str(local_chunk_dir),
            billing_project=billing_project,
            recursive=False,
            dry_run=dry_run,
        )

        if dry_run:
            continue

        final_path = local_chunk_dir / chunk_name
        temp_path = local_chunk_dir / f"{chunk_name}_.gstmp"
        if temp_path.exists():
            raise RuntimeError(
                f"Chunk copy left a temporary file behind: {temp_path}. "
                "Delete it and rerun the exporter."
            )
        if not final_path.exists():
            raise RuntimeError(f"Chunk copy did not produce the expected file: {final_path}")


def _existing_chunk_ids(local_chunk_dir: Path) -> set[int]:
    if not local_chunk_dir.exists():
        return set()

    found: set[int] = set()
    for path in local_chunk_dir.iterdir():
        if not path.is_file():
            continue
        try:
            found.add(int(path.name))
        except ValueError:
            continue
    return found


def _remove_stale_temp_chunks(local_chunk_dir: Path, chunk_ids: Iterable[int], *, dry_run: bool) -> None:
    for chunk_id in chunk_ids:
        temp_path = local_chunk_dir / f"{chunk_id}_.gstmp"
        if not temp_path.exists():
            continue
        if dry_run:
            print(f"+ rm -f {temp_path}")
        else:
            temp_path.unlink()
            print(f"Removed stale temp chunk: {temp_path}")


def _export_split(
    *,
    tool_name: str,
    base_cmd: list[str],
    remote_root_uri: str,
    local_dataset_root: Path,
    split: str,
    target_shape: int,
    selection_policy: str,
    billing_project: str | None,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    split_uri = f"{remote_root_uri.rstrip('/')}/{split}"
    source_meta = _load_remote_json(
        tool_name=tool_name,
        base_cmd=base_cmd,
        uri=f"{split_uri}/zarr.json",
        billing_project=billing_project,
    )
    source_shape = _infer_shape_len(source_meta)
    chunk_len = _infer_chunk_len(source_meta)
    if source_shape is None or chunk_len is None:
        raise ValueError(f"Could not infer shape/chunk size for {split_uri}")
    if target_shape > source_shape:
        raise ValueError(
            f"Requested shape {target_shape} exceeds source shape {source_shape} for {split_uri}"
        )

    chunk_count = int(math.ceil(target_shape / chunk_len))
    if chunk_count <= 0:
        raise ValueError(f"Requested shape too small for export: {target_shape}")

    split_root = local_dataset_root / split
    local_meta = _build_split_metadata(source_meta, target_shape)
    _write_json(split_root / "zarr.json", local_meta, dry_run=dry_run)

    requested_chunk_ids = list(range(chunk_count))
    local_chunk_dir = split_root / "c"
    existing_chunk_ids = set() if dry_run else _existing_chunk_ids(local_chunk_dir)
    missing_chunk_ids = [chunk_id for chunk_id in requested_chunk_ids if chunk_id not in existing_chunk_ids]

    if existing_chunk_ids:
        print(
            f"Skipping {len(existing_chunk_ids & set(requested_chunk_ids))} existing chunks for {local_chunk_dir}"
        )

    for chunk_ids in _batched(missing_chunk_ids, batch_size):
        _remove_stale_temp_chunks(local_chunk_dir, chunk_ids, dry_run=dry_run)
        chunk_uris = [f"{split_uri}/c/{chunk_id}" for chunk_id in chunk_ids]
        _copy_chunk_batch(
            tool_name=tool_name,
            base_cmd=base_cmd,
            chunk_uris=chunk_uris,
            local_chunk_dir=local_chunk_dir,
            billing_project=billing_project,
            dry_run=dry_run,
        )

    return {
        "split": split,
        "source_uri": split_uri,
        "source_shape": source_shape,
        "exported_shape": target_shape,
        "chunk_len": chunk_len,
        "selected_chunk_start": 0,
        "selected_chunk_end": chunk_count - 1,
        "selected_chunk_count": chunk_count,
        "selection_policy": selection_policy,
    }


def _export_package(
    *,
    package_name: str,
    datasets: list[str],
    staging_root: Path,
    tool_name: str,
    base_cmd: list[str],
    billing_project: str | None,
    val_policy: str,
    dry_run: bool,
    batch_size: int,
) -> tuple[Path, dict[str, Any]]:
    if package_name not in PACKAGE_TOKEN_TARGETS:
        raise ValueError(f"Unknown package: {package_name}")

    package_root = staging_root / f"paper_budget_{package_name}_val-{val_policy}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "package_name": package_name,
        "package_root": str(package_root),
        "source_kind": "gcs_zarr_v3",
        "billing_project": billing_project,
        "train_selection_policy": "contiguous_prefix_chunks",
        "val_policy": val_policy,
        "datasets": [],
    }

    for dataset_name in datasets:
        remote_root_uri = DATASETS[dataset_name]
        local_dataset_root = package_root / dataset_name
        root_meta = _load_remote_json(
            tool_name=tool_name,
            base_cmd=base_cmd,
            uri=f"{remote_root_uri.rstrip('/')}/zarr.json",
            billing_project=billing_project,
        )
        _write_json(local_dataset_root / "zarr.json", root_meta, dry_run=dry_run)

        train_target = PACKAGE_TOKEN_TARGETS[package_name][dataset_name]
        exported_splits = [
            _export_split(
                tool_name=tool_name,
                base_cmd=base_cmd,
                remote_root_uri=remote_root_uri,
                local_dataset_root=local_dataset_root,
                split="train",
                target_shape=train_target,
                selection_policy="contiguous_prefix_chunks",
                billing_project=billing_project,
                dry_run=dry_run,
                batch_size=batch_size,
            )
        ]

        split_names = ["train"]
        if val_policy == "full":
            val_meta = _load_remote_json(
                tool_name=tool_name,
                base_cmd=base_cmd,
                uri=f"{remote_root_uri.rstrip('/')}/val/zarr.json",
                billing_project=billing_project,
            )
            val_shape = _infer_shape_len(val_meta)
            if val_shape is None:
                raise ValueError(f"Could not infer val shape for {remote_root_uri}")
            exported_splits.append(
                _export_split(
                    tool_name=tool_name,
                    base_cmd=base_cmd,
                    remote_root_uri=remote_root_uri,
                    local_dataset_root=local_dataset_root,
                    split="val",
                    target_shape=val_shape,
                    selection_policy="full_split_copy",
                    billing_project=billing_project,
                    dry_run=dry_run,
                    batch_size=batch_size,
                )
            )
            split_names.append("val")

        if dry_run:
            print(f"Dry run: skipped local validation for {local_dataset_root}")
        else:
            if not _validate_local(local_dataset_root, split_names):
                raise ValueError(f"Validation failed for exported dataset root: {local_dataset_root}")

        manifest["datasets"].append(
            {
                "dataset_name": dataset_name,
                "source_root_uri": remote_root_uri,
                "local_root": str(local_dataset_root),
                "splits": exported_splits,
            }
        )

    _write_json(package_root / "export_manifest.json", manifest, dry_run=dry_run)
    return package_root, manifest


def _run_full_mirror(
    *,
    datasets: list[str],
    splits: list[str],
    staging_root: Path,
    tool_name: str,
    base_cmd: list[str],
    billing_project: str | None,
    aws: str | None,
    bucket: str | None,
    prefix: str,
    endpoint_url: str | None,
    region: str | None,
    force_download: bool,
    skip_upload: bool,
    delete_remote_extra: bool,
    cleanup_local: bool,
    check_only: bool,
    dry_run: bool,
) -> int:
    for dataset_name in datasets:
        dataset_root = staging_root / dataset_name
        if not check_only:
            for split in splits:
                split_root = dataset_root / split
                if split_root.exists() and not force_download:
                    print(f"Skipping download (already exists): {split_root}")
                    continue
                if split_root.exists() and force_download and not dry_run:
                    shutil.rmtree(split_root)
                _gcs_cp(
                    tool_name=tool_name,
                    base_cmd=base_cmd,
                    sources=[f"{DATASETS[dataset_name].rstrip('/')}/{split}"],
                    dest=str(dataset_root),
                    billing_project=billing_project,
                    recursive=True,
                    dry_run=dry_run,
                )

        if dry_run:
            print(f"Dry run: skipped local validation for {dataset_name}.")
        else:
            if not _validate_local(dataset_root, splits):
                print(f"Validation failed for {dataset_name} under {dataset_root}")
                return 2
            print(f"Validated local dataset root: {dataset_root}")

        if skip_upload or check_only:
            continue

        assert aws is not None
        assert bucket is not None
        assert endpoint_url is not None
        _upload_path(
            aws=aws,
            local_root=dataset_root,
            bucket=bucket,
            prefix=f"{prefix.rstrip('/')}/{dataset_name}",
            endpoint_url=endpoint_url,
            region=region,
            dry_run=dry_run,
            delete=delete_remote_extra,
        )
        print(f"Mirrored {dataset_name} to {_s3_uri(bucket, f'{prefix.rstrip('/')}/{dataset_name}')}")

        if cleanup_local:
            if dry_run:
                print(f"+ rm -rf {dataset_root}")
            else:
                shutil.rmtree(dataset_root)
                print(f"Removed local staging dataset: {dataset_root}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror TTT-E2E datasets from GCS to Backblaze B2, either as full roots or compact paper-budget exports."
    )
    parser.add_argument(
        "--gcs-tool",
        choices=("gcloud", "gsutil"),
        default=os.environ.get("GCS_TOOL"),
        help="Force the GCS CLI transport instead of auto-detecting gcloud first.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "package"),
        default="package",
        help="`full` mirrors entire split roots. `package` exports compact paper-budget subsets.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="Local staging root used while mirroring datasets.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS.keys()),
        help=f"Comma-separated datasets. Options: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split names for full-mirror mode (default: train,val).",
    )
    parser.add_argument(
        "--packages",
        default="760m",
        help="Comma-separated compact package presets to export in package mode. Options: 125m,760m",
    )
    parser.add_argument(
        "--val-policy",
        choices=("full", "none"),
        default="full",
        help="Whether to include the full val split in compact package exports.",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=64,
        help="How many chunk objects to copy per GCS batch command in package mode.",
    )
    parser.add_argument(
        "--billing-project",
        default=None,
        help="Requester-pays billing project. Falls back to GCS_BILLING_PROJECT.",
    )
    parser.add_argument(
        "--b2-bucket",
        default=None,
        help="Backblaze B2 bucket name. Falls back to B2_BUCKET.",
    )
    parser.add_argument(
        "--b2-prefix",
        default=None,
        help="Backblaze B2 prefix for exported roots. Falls back to B2_DATASET_PREFIX.",
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
        "--force-download",
        action="store_true",
        help="Re-download selected splits even if the target path exists (full mode only).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build the local export only; do not sync to Backblaze.",
    )
    parser.add_argument(
        "--delete-remote-extra",
        action="store_true",
        help="Pass --delete during the B2 sync so remote files not present locally are removed.",
    )
    parser.add_argument(
        "--cleanup-local",
        action="store_true",
        help="Delete local staged exports after a successful upload.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify local dataset layout in full mode; package mode does not support this.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing copy/upload or local writes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file.expanduser().resolve())

    datasets = _parse_csv(args.datasets, allowed=set(DATASETS))
    splits = _parse_csv(args.splits)
    packages = _parse_csv(args.packages, allowed=set(PACKAGE_TOKEN_TARGETS))

    staging_root = _resolve_staging_root(args.staging_root)

    billing_project = args.billing_project or os.environ.get("GCS_BILLING_PROJECT")
    bucket = args.b2_bucket or os.environ.get("B2_BUCKET")
    prefix = args.b2_prefix or os.environ.get("B2_DATASET_PREFIX", "ttt-e2e-datasets")
    endpoint_url = args.endpoint_url or os.environ.get("B2_ENDPOINT_URL")
    region = os.environ.get("AWS_DEFAULT_REGION")

    tool_name, base_cmd = _find_gcs_tool(
        preferred=args.gcs_tool,
        allow_missing=(args.dry_run and args.mode == "full"),
    )

    aws: str | None = None
    if not args.skip_upload and not args.check_only:
        if not bucket:
            raise ValueError("Missing Backblaze bucket. Set B2_BUCKET or pass --b2-bucket.")
        if not endpoint_url:
            raise ValueError("Missing Backblaze endpoint URL. Set B2_ENDPOINT_URL or pass --endpoint-url.")
        aws = _find_aws(allow_missing=args.dry_run)

    if args.mode == "full":
        return _run_full_mirror(
            datasets=datasets,
            splits=splits,
            staging_root=staging_root,
            tool_name=tool_name,
            base_cmd=base_cmd,
            billing_project=billing_project,
            aws=aws,
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
            region=region,
            force_download=args.force_download,
            skip_upload=args.skip_upload,
            delete_remote_extra=args.delete_remote_extra,
            cleanup_local=args.cleanup_local,
            check_only=args.check_only,
            dry_run=args.dry_run,
        )

    if args.check_only:
        raise ValueError("--check-only is only supported in --mode full.")

    for package_name in packages:
        package_root, manifest = _export_package(
            package_name=package_name,
            datasets=datasets,
            staging_root=staging_root,
            tool_name=tool_name,
            base_cmd=base_cmd,
            billing_project=billing_project,
            val_policy=args.val_policy,
            dry_run=args.dry_run,
            batch_size=args.chunk_batch_size,
        )

        if args.dry_run:
            print(f"Dry run: built package plan {package_root.name}")
        else:
            print(f"Built compact package: {package_root}")
            print(
                f"  train_selection_policy={manifest['train_selection_policy']} val_policy={manifest['val_policy']}"
            )

        if not args.skip_upload:
            assert aws is not None
            assert bucket is not None
            assert endpoint_url is not None
            _upload_path(
                aws=aws,
                local_root=package_root,
                bucket=bucket,
                prefix=f"{prefix.rstrip('/')}/{package_root.name}",
                endpoint_url=endpoint_url,
                region=region,
                dry_run=args.dry_run,
                delete=args.delete_remote_extra,
            )
            print(f"Uploaded compact package to {_s3_uri(bucket, f'{prefix.rstrip('/')}/{package_root.name}')}")

        if args.cleanup_local:
            if args.dry_run:
                print(f"+ rm -rf {package_root}")
            else:
                shutil.rmtree(package_root)
                print(f"Removed local package root: {package_root}")

    print("Mirror workflow finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
