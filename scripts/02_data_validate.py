#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
import zarr.codecs
import zarr.storage


def _open_split(dataset_path: Path, split: str):
    codec = zarr.codecs.BloscCodec(
        cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
    )
    store = zarr.storage.LocalStore(str(dataset_path), read_only=True)
    return zarr.open_array(store, path=f"/{split}", codec=codec)


def _check_split(arr: zarr.Array, seq_len: int, samples: int) -> list[str]:
    errors: list[str] = []
    if arr.ndim != 1:
        errors.append(f"Expected 1D token array, got shape {arr.shape}")
        return errors

    if arr.shape[0] <= seq_len + 1:
        errors.append(
            f"Array too small for seq_len={seq_len}: length={arr.shape[0]}"
        )
        return errors

    # Sample a few sequences to verify slicing logic
    max_idx = (arr.shape[0] - 1) // seq_len
    sample_idxs = np.linspace(0, max_idx - 1, num=min(samples, max_idx), dtype=int)
    for idx in sample_idxs:
        sample = arr[idx * seq_len : (idx + 1) * seq_len + 1]
        if len(sample) != seq_len + 1:
            errors.append(
                f"Bad sample length at idx={idx}: got {len(sample)}, expected {seq_len + 1}"
            )
            break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Zarr dataset layout and seq_len slicing for TTT-E2E."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset root (contains train/val Zarr arrays).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length to validate against (default: 8192).",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated split names to check (default: train,val).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of random/linearly spaced samples to validate per split.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        return 2

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    ok = True
    for split in splits:
        try:
            arr = _open_split(dataset_path, split)
        except Exception as exc:
            print(f"Failed to open split '{split}': {exc}")
            ok = False
            continue

        errors = _check_split(arr, args.seq_len, args.samples)
        if errors:
            ok = False
            print(f"Split '{split}' errors:")
            for err in errors:
                print(f"- {err}")
        else:
            print(
                f"OK: {split} shape={arr.shape} dtype={arr.dtype}"
            )

    if not ok:
        print("\nValidation failed.")
        return 3

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
