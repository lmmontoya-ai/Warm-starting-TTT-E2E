#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
from botocore.config import Config
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

from ttt.research.author_checkpoints import load_env_file


@dataclass(frozen=True)
class MirrorSpec:
    stage_id: str
    run_id: str
    b2_checkpoint_prefix: str
    b2_experiment_prefix: str


PAPER_RUN_ID = "protocol_r_760m_author_seed_v1"
B2_ROOT = "ttt-e2e-artifacts"
STAGE_SPECS = {
    "S2_ADAPT": MirrorSpec(
        stage_id="S2_ADAPT",
        run_id="adapt-760m-e2e-8K-from-fa",
        b2_checkpoint_prefix=f"{B2_ROOT}/checkpoints/{PAPER_RUN_ID}/adapt-760m-e2e-8K-from-fa/",
        b2_experiment_prefix=f"{B2_ROOT}/experiments/{PAPER_RUN_ID}/S2_ADAPT/adapt-760m-e2e-8K-from-fa/",
    ),
    "S2": MirrorSpec(
        stage_id="S2",
        run_id="ext-760m-e2e-32K-from-fa-bridge",
        b2_checkpoint_prefix=f"{B2_ROOT}/checkpoints/{PAPER_RUN_ID}/ext-760m-e2e-32K-from-fa-bridge/",
        b2_experiment_prefix=f"{B2_ROOT}/experiments/{PAPER_RUN_ID}/S2/ext-760m-e2e-32K-from-fa-bridge/",
    ),
    "S3": MirrorSpec(
        stage_id="S3",
        run_id="ext-760m-e2e-32K",
        b2_checkpoint_prefix=f"{B2_ROOT}/checkpoints/{PAPER_RUN_ID}/ext-760m-e2e-32K/",
        b2_experiment_prefix=f"{B2_ROOT}/experiments/{PAPER_RUN_ID}/S3/ext-760m-e2e-32K/",
    ),
}


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _b2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT_URL"],
        region_name=os.environ.get("AWS_DEFAULT_REGION") or "us-east-005",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


def _iter_b2_objects(client, *, bucket: str, prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            yield key, int(obj["Size"])


def _read_b2_json(client, *, bucket: str, key: str) -> dict:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def _iter_latest_checkpoint_objects(client, *, bucket: str, prefix: str):
    latest_key = f"{prefix}latest.json"
    latest = _read_b2_json(client, bucket=bucket, key=latest_key)
    latest_path = str(latest["path"]).strip("/")
    wanted_prefixes = (latest_key, f"{prefix}{latest_path}/")
    for key, size in _iter_b2_objects(client, bucket=bucket, prefix=prefix):
        if key == wanted_prefixes[0] or key.startswith(wanted_prefixes[1]):
            yield key, size


def _existing_hf_sizes(api: HfApi, *, repo_id: str, repo_type: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    try:
        rows = api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, recursive=True, expand=True)
        for row in rows:
            path = getattr(row, "path", "")
            if not path:
                continue
            out[path] = getattr(row, "size", None)
    except Exception:
        return out
    return out


def _target_path(spec: MirrorSpec, *, source_key: str, source_prefix: str, kind: str) -> str:
    rel = source_key.removeprefix(source_prefix)
    return f"{PAPER_RUN_ID}/stages/{spec.stage_id}/{spec.run_id}/{kind}/{rel}"


def _retry_delay_seconds(error: HfHubHTTPError) -> int | None:
    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None) != 429:
        return None
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return max(1, int(retry_after))
    match = re.search(r"Retry after (\d+) seconds", str(error))
    if match:
        return max(1, int(match.group(1)))
    if "rate limit" in str(error).lower():
        return 3700
    return None


def _mirror_prefix(
    *,
    client,
    api: HfApi,
    bucket: str,
    repo_id: str,
    repo_type: str,
    spec: MirrorSpec,
    source_prefix: str,
    kind: str,
    existing: dict[str, int | None],
    dry_run: bool,
    limit: int,
    latest_checkpoint_only: bool = False,
    max_batch_files: int = 8,
    max_batch_bytes: int = 4 * 1024**3,
) -> tuple[int, int, int]:
    uploaded = 0
    skipped = 0
    bytes_seen = 0
    pending: list[tuple[str, int, str]] = []
    if latest_checkpoint_only:
        objects = list(_iter_latest_checkpoint_objects(client, bucket=bucket, prefix=source_prefix))
    else:
        objects = list(_iter_b2_objects(client, bucket=bucket, prefix=source_prefix))
    def flush_batch() -> int:
        if not pending:
            return 0
        with TemporaryDirectory(prefix="b2_hf_batch_") as td:
            operations: list[CommitOperationAdd] = []
            for index, (source_key, _size, target) in enumerate(pending):
                local_file = Path(td) / f"{index:05d}_{Path(source_key).name}"
                client.download_file(bucket, source_key, str(local_file))
                operations.append(CommitOperationAdd(path_in_repo=target, path_or_fileobj=str(local_file)))
            while True:
                try:
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        operations=operations,
                        commit_message=f"Upload {PAPER_RUN_ID} {spec.stage_id}/{spec.run_id} {kind} batch",
                    )
                    break
                except HfHubHTTPError as error:
                    delay = _retry_delay_seconds(error)
                    if delay is None:
                        raise
                    print(f"HF rate limit hit; sleeping {delay + 10} seconds before retrying batch.", flush=True)
                    time.sleep(delay + 10)
        for _source_key, size, target in pending:
            existing[target] = size
        count = len(pending)
        pending.clear()
        return count

    batch_bytes = 0
    for source_key, size in objects:
        target = _target_path(spec, source_key=source_key, source_prefix=source_prefix, kind=kind)
        bytes_seen += size
        if existing.get(target) == size:
            skipped += 1
            continue
        print(f"{source_key} -> {target} ({size} bytes)", flush=True)
        if dry_run:
            uploaded += 1
            if limit and uploaded >= limit:
                break
            continue
        if pending and (len(pending) >= max_batch_files or batch_bytes + size > max_batch_bytes):
            uploaded += flush_batch()
            batch_bytes = 0
        pending.append((source_key, size, target))
        batch_bytes += size
        if limit and uploaded >= limit:
            break
    if not dry_run:
        uploaded += flush_batch()
    return uploaded, skipped, bytes_seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror 760M Protocol R stage artifacts from Backblaze B2 into an HF stage-restore repo."
    )
    parser.add_argument("--repo-id", default=os.environ.get("HF_760M_RESULTS_REPO", "Luxel/ttt-e2e-760m-results"))
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--bucket", default=os.environ.get("B2_BUCKET", "TTTE2E"))
    parser.add_argument("--stages", default="S2_ADAPT,S2,S3")
    parser.add_argument("--env-file", type=Path, default=Path(".env.backblaze"))
    parser.add_argument("--hf-env-file", type=Path, default=Path(".env.hf"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N missing objects per prefix; for smoke tests.")
    parser.add_argument("--include-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-experiments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-batch-files", type=int, default=8)
    parser.add_argument("--max-batch-bytes", type=int, default=4 * 1024**3)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["latest", "all"],
        default="latest",
        help="For checkpoint prefixes, mirror only latest.json plus latest.path by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / args.env_file)
    load_env_file(repo_root / args.hf_env_file)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN or provide .env.hf.")

    client = _b2_client()
    api = HfApi(token=token)
    if not args.dry_run:
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
    existing = _existing_hf_sizes(api, repo_id=args.repo_id, repo_type=args.repo_type)

    selected = _parse_csv(args.stages)
    for stage_id in selected:
        if stage_id not in STAGE_SPECS:
            raise ValueError(f"Unknown stage: {stage_id}. Supported: {','.join(STAGE_SPECS)}")
        spec = STAGE_SPECS[stage_id]
        print(f"== {spec.stage_id}/{spec.run_id} ==", flush=True)
        if args.include_experiments:
            uploaded, skipped, bytes_seen = _mirror_prefix(
                client=client,
                api=api,
                bucket=args.bucket,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                spec=spec,
                source_prefix=spec.b2_experiment_prefix,
                kind="experiment",
                existing=existing,
                dry_run=args.dry_run,
                limit=args.limit,
                max_batch_files=args.max_batch_files,
                max_batch_bytes=args.max_batch_bytes,
            )
            print(f"experiment: uploaded={uploaded} skipped={skipped} bytes_seen={bytes_seen}", flush=True)
        if args.include_checkpoints:
            uploaded, skipped, bytes_seen = _mirror_prefix(
                client=client,
                api=api,
                bucket=args.bucket,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                spec=spec,
                source_prefix=spec.b2_checkpoint_prefix,
                kind="checkpoint",
                existing=existing,
                dry_run=args.dry_run,
                limit=args.limit,
                latest_checkpoint_only=args.checkpoint_selection == "latest",
                max_batch_files=args.max_batch_files,
                max_batch_bytes=args.max_batch_bytes,
            )
            print(f"checkpoint: uploaded={uploaded} skipped={skipped} bytes_seen={bytes_seen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
