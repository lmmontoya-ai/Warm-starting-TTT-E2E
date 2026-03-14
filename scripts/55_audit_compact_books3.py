#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zarr.storage


def _open_split(root: Path, split: str) -> zarr.Array:
    store = zarr.storage.LocalStore(str(root), read_only=True)
    return zarr.open_array(store, path=f"/{split}")


def _load_meta(root: Path, split: str) -> dict[str, Any]:
    split_root = root / split
    for name in ("zarr.json", ".zarray"):
        path = split_root / name
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"Missing Zarr metadata under {split_root}")


def _infer_chunk_len(meta: dict[str, Any]) -> int:
    chunk_grid = meta.get("chunk_grid")
    if isinstance(chunk_grid, dict):
        chunk_shape = chunk_grid.get("configuration", {}).get("chunk_shape")
        if isinstance(chunk_shape, list) and len(chunk_shape) == 1:
            return int(chunk_shape[0])
    chunks = meta.get("chunks")
    if isinstance(chunks, list) and len(chunks) == 1:
        return int(chunks[0])
    raise ValueError("Could not infer 1D chunk length from metadata.")


def _prefix_checks(small: zarr.Array, large: zarr.Array) -> dict[str, Any]:
    head_n = min(2_000_000, int(small.shape[0]), int(large.shape[0]))
    tail_n = min(100_000, int(small.shape[0]), int(large.shape[0]))
    head_match = bool(
        np.array_equal(
            np.asarray(small[:head_n], dtype=np.uint32),
            np.asarray(large[:head_n], dtype=np.uint32),
        )
    )
    tail_start = int(small.shape[0]) - tail_n
    tail_match = bool(
        np.array_equal(
            np.asarray(small[tail_start : tail_start + tail_n], dtype=np.uint32),
            np.asarray(large[tail_start : tail_start + tail_n], dtype=np.uint32),
        )
    )
    return {
        "head_tokens_checked": head_n,
        "tail_tokens_checked": tail_n,
        "head_match": head_match,
        "tail_match": tail_match,
        "is_prefix_of_large_train": head_match and tail_match,
    }


def _strided_sample(arr: zarr.Array, *, blocks: int, block_size: int) -> np.ndarray:
    n = int(arr.shape[0])
    if n <= block_size:
        return np.asarray(arr[:], dtype=np.int32)
    starts = np.linspace(0, max(0, n - block_size), num=blocks, dtype=int)
    parts = [np.asarray(arr[start : start + block_size], dtype=np.int32) for start in starts]
    return np.concatenate(parts)


def _jsd(a: np.ndarray, b: np.ndarray, *, vocab_size: int) -> float:
    count_a = np.bincount(a, minlength=vocab_size).astype(np.float64)
    count_b = np.bincount(b, minlength=vocab_size).astype(np.float64)
    p_a = count_a / count_a.sum()
    p_b = count_b / count_b.sum()
    mean = 0.5 * (p_a + p_b)
    mask_a = p_a > 0
    mask_b = p_b > 0
    kl_a = np.sum(p_a[mask_a] * np.log2(p_a[mask_a] / mean[mask_a]))
    kl_b = np.sum(p_b[mask_b] * np.log2(p_b[mask_b] / mean[mask_b]))
    return float(0.5 * (kl_a + kl_b))


def _top_tokens(sample: np.ndarray, *, k: int = 20) -> list[list[int]]:
    values, counts = np.unique(sample, return_counts=True)
    order = np.argsort(counts)[::-1][:k]
    return [[int(values[idx]), int(counts[idx])] for idx in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a compact Books3 train package is a strict prefix of a larger "
            "train stream and how representative it is of the shared val split."
        )
    )
    parser.add_argument("--small-root", type=Path, required=True, help="Compact Books3 root to audit.")
    parser.add_argument("--large-root", type=Path, required=True, help="Larger Books3 root to compare against.")
    parser.add_argument("--blocks", type=int, default=20, help="How many evenly spaced blocks to sample.")
    parser.add_argument("--block-size", type=int, default=100_000, help="Tokens per sampled block.")
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    small_root = args.small_root.expanduser().resolve()
    large_root = args.large_root.expanduser().resolve()

    small_train = _open_split(small_root, "train")
    large_train = _open_split(large_root, "train")
    large_val = _open_split(large_root, "val")

    small_meta = _load_meta(small_root, "train")
    large_meta = _load_meta(large_root, "train")
    small_chunk_len = _infer_chunk_len(small_meta)
    large_chunk_len = _infer_chunk_len(large_meta)

    prefix = _prefix_checks(small_train, large_train)
    small_sample = _strided_sample(small_train, blocks=args.blocks, block_size=args.block_size)
    large_train_sample = _strided_sample(large_train, blocks=args.blocks, block_size=args.block_size)
    large_val_sample = _strided_sample(large_val, blocks=args.blocks, block_size=args.block_size)

    payload = {
        "schema_version": "1.0",
        "small_root": str(small_root),
        "large_root": str(large_root),
        "small_train_tokens": int(small_train.shape[0]),
        "large_train_tokens": int(large_train.shape[0]),
        "large_val_tokens": int(large_val.shape[0]),
        "small_train_chunk_len": small_chunk_len,
        "large_train_chunk_len": large_chunk_len,
        "small_train_lt_chunk_len": int(small_train.shape[0]) < small_chunk_len,
        "prefix_checks": prefix,
        "sample_config": {
            "blocks": int(args.blocks),
            "block_size": int(args.block_size),
            "sample_tokens": int(small_sample.shape[0]),
        },
        "distribution": {
            "jsd_small_train_vs_large_val": _jsd(small_sample, large_val_sample, vocab_size=args.vocab_size),
            "jsd_large_train_vs_large_val": _jsd(
                large_train_sample, large_val_sample, vocab_size=args.vocab_size
            ),
            "jsd_small_train_vs_large_train": _jsd(
                small_sample, large_train_sample, vocab_size=args.vocab_size
            ),
            "small_train_top20": _top_tokens(small_sample),
            "large_train_top20": _top_tokens(large_train_sample),
            "large_val_top20": _top_tokens(large_val_sample),
        },
        "conclusion": {
            "likely_issue": (
                "Compact Books3 train package is a strict prefix of the larger train stream "
                "and is less representative of val than the larger train package."
            ),
            "recommended_fix": (
                "Use a broader Books3 train root for 125M extension stages, or rechunk/resample "
                "the compact package. Do not rely on contiguous prefix export when target tokens "
                "are smaller than one source chunk."
            ),
        },
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
