#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
from typing import Any


DATASETS = {
    "dclm_filter_8k": "gs://llama3-dclm-filter-8k/data.zarr",
    "books3": "gs://llama3-books3/data.zarr",
}

MODEL_TOKEN_TARGETS = {
    "125m": {
        "dclm_filter_8k": 2_520_000_000,
        "books3": 126_000_000,
    },
    "760m": {
        "dclm_filter_8k": 15_200_000_000,
        "books3": 759_000_000,
    },
}


def _find_tool() -> tuple[str, list[str]]:
    gcloud = shutil.which("gcloud")
    if gcloud:
        return ("gcloud", [gcloud, "storage"])

    gsutil = shutil.which("gsutil")
    if gsutil:
        return ("gsutil", [gsutil])

    raise FileNotFoundError("Neither 'gcloud' nor 'gsutil' was found in PATH.")


def _run_capture(cmd: list[str]) -> str:
    print("+", shlex.join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def _cat_object(tool_name: str, base_cmd: list[str], uri: str, billing_project: str | None) -> str:
    if tool_name == "gcloud":
        cmd = base_cmd[:] + ["cat", uri]
        if billing_project:
            cmd += ["--billing-project", billing_project]
        return _run_capture(cmd)

    cmd = base_cmd[:] + ["cat"]
    if billing_project:
        cmd += ["-u", billing_project]
    cmd += [uri]
    return _run_capture(cmd)


def _ls_prefix(
    tool_name: str,
    base_cmd: list[str],
    uri: str,
    billing_project: str | None,
    *,
    recursive: bool,
) -> list[str]:
    if tool_name == "gcloud":
        cmd = base_cmd[:] + ["ls"]
        if recursive:
            cmd.append("--recursive")
        if billing_project:
            cmd += ["--billing-project", billing_project]
        cmd.append(uri)
        output = _run_capture(cmd)
    else:
        cmd = base_cmd[:] + ["ls"]
        if recursive:
            cmd.append("-r")
        if billing_project:
            cmd += ["-u", billing_project]
        cmd.append(uri)
        output = _run_capture(cmd)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _load_zarr_metadata(
    tool_name: str,
    base_cmd: list[str],
    split_uri: str,
    billing_project: str | None,
) -> tuple[str, dict[str, Any]]:
    candidates = [".zarray", "zarr.json"]
    errors: list[str] = []
    for name in candidates:
        uri = f"{split_uri.rstrip('/')}/{name}"
        try:
            payload = json.loads(_cat_object(tool_name, base_cmd, uri, billing_project))
            return name, payload
        except subprocess.CalledProcessError as exc:
            errors.append(f"{name}: exit={exc.returncode}")
        except json.JSONDecodeError as exc:
            errors.append(f"{name}: invalid json ({exc})")
    raise RuntimeError(f"Could not load Zarr metadata for {split_uri}. Tried: {', '.join(errors)}")


def _infer_chunk_len(metadata_name: str, payload: dict[str, Any]) -> int | None:
    if metadata_name == ".zarray":
        chunks = payload.get("chunks")
        if isinstance(chunks, list) and len(chunks) == 1:
            return int(chunks[0])
        return None

    # Zarr v3 style
    chunk_grid = payload.get("chunk_grid")
    if isinstance(chunk_grid, dict):
        chunk_shape = chunk_grid.get("configuration", {}).get("chunk_shape")
        if isinstance(chunk_shape, list) and len(chunk_shape) == 1:
            return int(chunk_shape[0])
    return None


def _infer_shape_len(metadata_name: str, payload: dict[str, Any]) -> int | None:
    if metadata_name == ".zarray":
        shape = payload.get("shape")
        if isinstance(shape, list) and len(shape) == 1:
            return int(shape[0])
        return None

    shape = payload.get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        return int(shape[0])
    return None


def _estimate_needed_chunks(token_target: int, chunk_len: int) -> int:
    return int(math.ceil(token_target / chunk_len))


def _sample_chunk_objects(
    tool_name: str,
    base_cmd: list[str],
    split_uri: str,
    billing_project: str | None,
    *,
    limit: int,
) -> list[str]:
    entries = _ls_prefix(tool_name, base_cmd, f"{split_uri.rstrip('/')}/*", billing_project, recursive=False)
    filtered = [item for item in entries if not item.endswith("/")]
    return filtered[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Requester Pays GCS-hosted Zarr datasets to determine whether direct subset export is practical."
    )
    parser.add_argument(
        "--datasets",
        default="dclm_filter_8k,books3",
        help=f"Comma-separated datasets. Options: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated split names to inspect (default: train,val).",
    )
    parser.add_argument(
        "--billing-project",
        default=None,
        help="Requester Pays billing project to pass through the GCS CLI.",
    )
    parser.add_argument(
        "--sample-objects",
        type=int,
        default=8,
        help="How many top-level split objects to sample while inferring chunk layout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool_name, base_cmd = _find_tool()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    report: list[dict[str, Any]] = []
    direct_subset_feasible = True

    for dataset_name in datasets:
        bucket_uri = DATASETS[dataset_name]
        for split in splits:
            split_uri = f"{bucket_uri.rstrip('/')}/{split}"
            metadata_name, metadata = _load_zarr_metadata(
                tool_name,
                base_cmd,
                split_uri,
                args.billing_project,
            )
            shape_len = _infer_shape_len(metadata_name, metadata)
            chunk_len = _infer_chunk_len(metadata_name, metadata)
            sample_objects = _sample_chunk_objects(
                tool_name,
                base_cmd,
                split_uri,
                args.billing_project,
                limit=args.sample_objects,
            )

            model_estimates: dict[str, dict[str, int] | str] = {}
            if chunk_len is None:
                direct_subset_feasible = False
                model_estimates["status"] = "non-1d-or-unknown-chunking"
            else:
                for model_key, targets in MODEL_TOKEN_TARGETS.items():
                    token_target = targets.get(dataset_name)
                    if token_target is None:
                        continue
                    model_estimates[model_key] = {
                        "token_target": token_target,
                        "estimated_chunks_needed": _estimate_needed_chunks(token_target, chunk_len),
                    }

            if shape_len is not None and chunk_len is not None:
                total_chunks = _estimate_needed_chunks(shape_len, chunk_len)
            else:
                total_chunks = None

            row = {
                "dataset": dataset_name,
                "split": split,
                "uri": split_uri,
                "metadata_object": metadata_name,
                "shape_len": shape_len,
                "chunk_len": chunk_len,
                "total_chunks_estimate": total_chunks,
                "sample_objects": sample_objects,
                "model_estimates": model_estimates,
            }
            report.append(row)

    print(json.dumps(report, indent=2, sort_keys=True))

    print("\n=== Verdict ===")
    if direct_subset_feasible:
        print(
            "Direct subset export looks feasible: the inspected metadata exposes a 1D array with readable chunk geometry."
        )
        print(
            "Next step: implement a chunk-aligned exporter that reads only the needed chunk range from GCS and writes compact local/B2 dataset roots."
        )
    else:
        print(
            "Direct subset export could not be confirmed from metadata alone. Chunk geometry is missing or not 1D."
        )
        print("Next step: inspect the sample objects more closely or fall back to a fuller mirror strategy.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
