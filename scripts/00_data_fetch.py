#!/usr/bin/env python3
"""Download the original TTT-E2E tokenized datasets (Zarr v3) from GCS.

The original authors publish two tokenized corpora:

    dclm_filter_8k  gs://llama3-dclm-filter-8k/data.zarr   (pretraining, bridge)
    books3          gs://llama3-books3/data.zarr           (32K extension, eval)

Both buckets are Requester Pays: every read is billed to a Google Cloud
project you name with --billing-project (or GCS_BILLING_PROJECT). Without one,
GCS rejects even a listing.

Two download modes:

  --budget full   copy the whole bucket for each dataset (several TB).
  --budget 125m   copy only what the 125M ladder reads: the first train chunks
                  up to the 125M token budget (2.52B DCLM tokens, 126M Books3
                  tokens) plus the full val split. About 40 GB in total:
                  DCLM train 10.4, DCLM val 20.4, Books3 train 0.8,
                  Books3 val 8.4.
  --budget 760m   the same selection at the 760M token budget.

Budget mode reproduces the maintainer's `paper_budget_<budget>_val-full`
package exactly: train keeps a contiguous prefix of whole source chunks
(`ceil(target / chunk_len)` chunks) and its zarr.json records shape = target,
so readers see exactly `target` tokens; val is copied whole. The selection is
written to <dest-root>/export_manifest.json.

Output layout (what --dclm-root / --books-root and
scripts/13_dataset_fingerprint.py expect):

    <dest-root>/dclm_filter_8k/{zarr.json,train/,val/}
    <dest-root>/books3/{zarr.json,train/,val/}

Example:

    python scripts/00_data_fetch.py --dest-root /data/ttt-e2e --budget 125m \\
        --billing-project my-gcp-project
"""

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

SPLITS = ("train", "val")

# Train-split token targets per model budget. Changing a value changes the
# selected data; keep them equal to the values that built the published
# paper_budget_<budget>_val-full packages.
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

BILLING_ENV_VAR = "GCS_BILLING_PROJECT"
TRAIN_SELECTION_POLICY = "contiguous_prefix_chunks"


class MissingBillingProjectError(ValueError):
    pass


def resolve_billing_project(cli_value: str | None, environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    value = (cli_value or "").strip() or str(env.get(BILLING_ENV_VAR, "")).strip()
    if not value:
        raise MissingBillingProjectError(
            "The TTT-E2E datasets live in Requester Pays buckets "
            f"({', '.join(DATASETS.values())}); GCS refuses reads without a billing project. "
            f"Pass --billing-project <your-gcp-project> or set {BILLING_ENV_VAR}. "
            "Downloads are billed to that project."
        )
    return value


# --------------------------------------------------------------------------- GCS transport


class GcsTransport:
    """Thin wrapper over `gcloud storage` or `gsutil` with a billing project."""

    def __init__(self, *, tool_name: str, base_cmd: list[str], billing_project: str, dry_run: bool = False):
        self.tool_name = tool_name
        self.base_cmd = list(base_cmd)
        self.billing_project = billing_project
        self.dry_run = dry_run

    def cp_command(self, sources: list[str], dest: str, *, recursive: bool) -> list[str]:
        return build_cp_command(
            tool_name=self.tool_name,
            base_cmd=self.base_cmd,
            sources=sources,
            dest=dest,
            billing_project=self.billing_project,
            recursive=recursive,
        )

    def cp(self, sources: list[str], dest: str, *, recursive: bool) -> None:
        cmd = self.cp_command(sources, dest, recursive=recursive)
        print("+", shlex.join(cmd), flush=True)
        if self.dry_run:
            return
        subprocess.run(cmd, check=True)

    def cat(self, uri: str) -> str:
        if self.tool_name == "gcloud":
            cmd = self.base_cmd + ["cat", uri, "--billing-project", self.billing_project]
        else:
            cmd = self.base_cmd + ["-u", self.billing_project, "cat", uri]
        print("+", shlex.join(cmd), flush=True)
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def build_cp_command(
    *,
    tool_name: str,
    base_cmd: list[str],
    sources: list[str],
    dest: str,
    billing_project: str,
    recursive: bool,
) -> list[str]:
    if tool_name == "gcloud":
        cmd = list(base_cmd) + ["cp"]
        if recursive:
            cmd.append("-r")
        cmd += [f"--billing-project={billing_project}"]
        return cmd + list(sources) + [dest]
    cmd = list(base_cmd) + ["-u", billing_project, "cp"]
    if recursive:
        cmd.append("-r")
    return cmd + list(sources) + [dest]


def find_gcs_tool(preferred: str | None = None) -> tuple[str, list[str]]:
    gcloud = shutil.which("gcloud")
    gsutil = shutil.which("gsutil")
    if preferred == "gcloud" or (preferred is None and gcloud):
        if not gcloud:
            raise FileNotFoundError("Requested gcloud, but 'gcloud' was not found in PATH.")
        return "gcloud", [gcloud, "storage"]
    if preferred == "gsutil" or (preferred is None and gsutil):
        if not gsutil:
            raise FileNotFoundError("Requested gsutil, but 'gsutil' was not found in PATH.")
        return "gsutil", [gsutil, "-m"]
    raise FileNotFoundError(
        "Neither 'gcloud' nor 'gsutil' was found in PATH. Install the Google Cloud CLI "
        "(https://cloud.google.com/sdk/docs/install) and run `gcloud auth login`."
    )


# --------------------------------------------------------------------------- selection


def infer_shape_len(meta: dict[str, Any]) -> int | None:
    shape = meta.get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        return int(shape[0])
    return None


def infer_chunk_len(meta: dict[str, Any]) -> int | None:
    chunk_grid = meta.get("chunk_grid")
    if isinstance(chunk_grid, dict):
        chunk_shape = chunk_grid.get("configuration", {}).get("chunk_shape")
        if isinstance(chunk_shape, list) and len(chunk_shape) == 1:
            return int(chunk_shape[0])
    chunks = meta.get("chunks")
    if isinstance(chunks, list) and len(chunks) == 1:
        return int(chunks[0])
    return None


def plan_split(*, source_meta: dict[str, Any], target_shape: int, selection_policy: str) -> dict[str, Any]:
    """Pure selection math for one split: which leading chunks to copy."""

    source_shape = infer_shape_len(source_meta)
    chunk_len = infer_chunk_len(source_meta)
    if source_shape is None or chunk_len is None:
        raise ValueError("Could not infer a 1-D shape and chunk length from the split metadata.")
    if target_shape > source_shape:
        raise ValueError(f"Requested shape {target_shape} exceeds source shape {source_shape}")
    chunk_count = int(math.ceil(target_shape / chunk_len))
    if chunk_count <= 0:
        raise ValueError(f"Requested shape too small for export: {target_shape}")
    selection_warning = ""
    if selection_policy == TRAIN_SELECTION_POLICY and target_shape < chunk_len:
        selection_warning = (
            "target_shape is smaller than the source chunk length, so the exported split "
            "is necessarily a strict prefix of the first source chunk. This preserves token "
            "budget but may distort train-distribution representativeness for ordered corpora."
        )
    return {
        "source_shape": source_shape,
        "exported_shape": int(target_shape),
        "chunk_len": chunk_len,
        "source_chunk_count": int(math.ceil(source_shape / chunk_len)),
        "selected_chunk_start": 0,
        "selected_chunk_end": chunk_count - 1,
        "selected_chunk_count": chunk_count,
        "selection_policy": selection_policy,
        "selection_warning": selection_warning,
    }


def local_split_metadata(source_meta: dict[str, Any], target_shape: int) -> dict[str, Any]:
    payload = json.loads(json.dumps(source_meta))
    payload["shape"] = [int(target_shape)]
    return payload


# --------------------------------------------------------------------------- local helpers


def _write_json(path: Path, payload: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        print(f"+ write-json {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _batched(values: Iterable[int], batch_size: int) -> Iterable[list[int]]:
    batch: list[int] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _existing_chunk_ids(local_chunk_dir: Path) -> set[int]:
    if not local_chunk_dir.exists():
        return set()
    found: set[int] = set()
    for path in local_chunk_dir.iterdir():
        if path.is_file():
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


def has_zarr_split(dataset_path: Path, split: str) -> bool:
    split_path = dataset_path / split
    if not split_path.exists():
        return False
    return (split_path / "zarr.json").exists() or (split_path / ".zarray").exists() or split_path.is_dir()


def check_dataset(dataset_path: Path, splits: Iterable[str] = SPLITS) -> bool:
    if not dataset_path.exists():
        print(f"Missing dataset path: {dataset_path}")
        return False
    ok = True
    for split in splits:
        if not has_zarr_split(dataset_path, split):
            print(f"Missing split '{split}' under {dataset_path}")
            ok = False
    return ok


# --------------------------------------------------------------------------- budget export


def export_split(
    *,
    transport: Any,
    remote_root_uri: str,
    local_dataset_root: Path,
    split: str,
    target_shape: int | None,
    selection_policy: str,
    batch_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    split_uri = f"{remote_root_uri.rstrip('/')}/{split}"
    source_meta = json.loads(transport.cat(f"{split_uri}/zarr.json"))
    if target_shape is None:
        target_shape = infer_shape_len(source_meta)
        if target_shape is None:
            raise ValueError(f"Could not infer shape for {split_uri}")
    plan = plan_split(source_meta=source_meta, target_shape=target_shape, selection_policy=selection_policy)
    if plan["selection_warning"]:
        print(f"WARNING: {plan['selection_warning']}")

    split_root = local_dataset_root / split
    _write_json(split_root / "zarr.json", local_split_metadata(source_meta, target_shape), dry_run=dry_run)

    requested = list(range(plan["selected_chunk_count"]))
    local_chunk_dir = split_root / "c"
    existing = set() if dry_run else _existing_chunk_ids(local_chunk_dir)
    missing = [chunk_id for chunk_id in requested if chunk_id not in existing]
    if existing:
        print(f"Skipping {len(existing & set(requested))} existing chunks for {local_chunk_dir}")
    if not dry_run:
        local_chunk_dir.mkdir(parents=True, exist_ok=True)
    total = len(missing)
    copied = 0
    for chunk_ids in _batched(missing, batch_size):
        _remove_stale_temp_chunks(local_chunk_dir, chunk_ids, dry_run=dry_run)
        for chunk_id in chunk_ids:
            copied += 1
            chunk_uri = f"{split_uri}/c/{chunk_id}"
            print(f"Copying chunk {copied}/{total}: {chunk_uri}")
            transport.cp([chunk_uri], str(local_chunk_dir), recursive=False)
            if dry_run:
                continue
            final_path = local_chunk_dir / str(chunk_id)
            temp_path = local_chunk_dir / f"{chunk_id}_.gstmp"
            if temp_path.exists():
                raise RuntimeError(
                    f"Chunk copy left a temporary file behind: {temp_path}. Delete it and rerun."
                )
            if not final_path.exists():
                raise RuntimeError(f"Chunk copy did not produce the expected file: {final_path}")

    return {"split": split, "source_uri": split_uri, **plan}


def export_budget(
    *,
    budget: str,
    datasets: list[str],
    dest_root: Path,
    transport: Any,
    val_policy: str,
    batch_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    if budget not in PACKAGE_TOKEN_TARGETS:
        raise ValueError(f"Unknown budget: {budget}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "package_name": budget,
        "package_id": f"paper_budget_{budget}_val-{val_policy}",
        "package_root": str(dest_root),
        "source_kind": "gcs_zarr_v3",
        "billing_project": getattr(transport, "billing_project", None),
        "train_selection_policy": TRAIN_SELECTION_POLICY,
        "val_policy": val_policy,
        "datasets": [],
    }
    for dataset_name in datasets:
        remote_root_uri = DATASETS[dataset_name]
        local_dataset_root = dest_root / dataset_name
        root_meta = json.loads(transport.cat(f"{remote_root_uri.rstrip('/')}/zarr.json"))
        _write_json(local_dataset_root / "zarr.json", root_meta, dry_run=dry_run)

        exported = [
            export_split(
                transport=transport,
                remote_root_uri=remote_root_uri,
                local_dataset_root=local_dataset_root,
                split="train",
                target_shape=PACKAGE_TOKEN_TARGETS[budget][dataset_name],
                selection_policy=TRAIN_SELECTION_POLICY,
                batch_size=batch_size,
                dry_run=dry_run,
            )
        ]
        split_names = ["train"]
        if val_policy == "full":
            exported.append(
                export_split(
                    transport=transport,
                    remote_root_uri=remote_root_uri,
                    local_dataset_root=local_dataset_root,
                    split="val",
                    target_shape=None,
                    selection_policy="full_split_copy",
                    batch_size=batch_size,
                    dry_run=dry_run,
                )
            )
            split_names.append("val")

        if not dry_run and not check_dataset(local_dataset_root, split_names):
            raise ValueError(f"Validation failed for exported dataset root: {local_dataset_root}")
        manifest["datasets"].append(
            {
                "dataset_name": dataset_name,
                "source_root_uri": remote_root_uri,
                "local_root": str(local_dataset_root),
                "splits": exported,
            }
        )
    _write_json(dest_root / "export_manifest.json", manifest, dry_run=dry_run)
    return manifest


def fetch_full(*, datasets: list[str], dest_root: Path, transport: GcsTransport, force: bool) -> None:
    for name in datasets:
        target = dest_root / name
        if target.exists() and any(target.iterdir()) and not force:
            print(f"Skipping download (already exists): {target}")
            continue
        transport.cp([f"{DATASETS[name]}/"], str(target), recursive=True)


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest-root", type=Path, required=True, help="Datasets are written to <dest-root>/<dataset>.")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help=f"Comma-separated dataset list. Options: {', '.join(DATASETS)}",
    )
    parser.add_argument(
        "--budget",
        choices=("full", *PACKAGE_TOKEN_TARGETS),
        default="full",
        help="full = whole buckets; 125m / 760m = token-budget train prefix plus val (see above).",
    )
    parser.add_argument(
        "--val-policy",
        choices=("full", "none"),
        default="full",
        help="Budget mode only: copy the full val split (default) or skip it.",
    )
    parser.add_argument(
        "--billing-project",
        default=None,
        help=f"GCP project billed for Requester Pays reads. Falls back to ${BILLING_ENV_VAR}. Required to download.",
    )
    parser.add_argument("--gcs-tool", choices=("gcloud", "gsutil"), default=None, help="Force the GCS CLI.")
    parser.add_argument("--chunk-batch-size", type=int, default=64, help="Budget mode: chunks handled per batch.")
    parser.add_argument("--check-only", action="store_true", help="Only verify the local layout; no download.")
    parser.add_argument("--force", action="store_true", help="Full mode: re-download even if the target exists.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read remote metadata and print the copy plan without copying chunks or writing files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest_root = args.dest_root.expanduser().resolve()
    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in selected if d not in DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {', '.join(unknown)}")
        return 2

    if not args.check_only:
        try:
            billing_project = resolve_billing_project(args.billing_project)
        except MissingBillingProjectError as exc:
            print(f"ERROR: {exc}")
            return 2
        try:
            tool_name, base_cmd = find_gcs_tool(args.gcs_tool)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 3
        transport = GcsTransport(
            tool_name=tool_name, base_cmd=base_cmd, billing_project=billing_project, dry_run=args.dry_run
        )
        if not args.dry_run:
            dest_root.mkdir(parents=True, exist_ok=True)
        if args.budget == "full":
            fetch_full(datasets=selected, dest_root=dest_root, transport=transport, force=args.force)
        else:
            manifest = export_budget(
                budget=args.budget,
                datasets=selected,
                dest_root=dest_root,
                transport=transport,
                val_policy=args.val_policy,
                batch_size=args.chunk_batch_size,
                dry_run=args.dry_run,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
        if args.dry_run:
            print("Dry run: nothing was copied.")
            return 0

    print("\n=== Dataset Check ===")
    ok = True
    splits = SPLITS if args.budget == "full" or args.val_policy == "full" else ("train",)
    for name in selected:
        target = dest_root / name
        if check_dataset(target, splits):
            print(f"OK: {name} @ {target}")
        else:
            ok = False
    if not ok:
        print("\nOne or more datasets failed validation.")
        return 4
    print("\nAll datasets look good. Next: write fingerprints with scripts/13_dataset_fingerprint.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
