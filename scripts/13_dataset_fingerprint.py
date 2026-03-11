#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ttt.research.types import DatasetRef


def _tokens_sha256(tokens: list[int]) -> str:
    h = hashlib.sha256()
    for token in tokens:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def _load_tokens_from_json(path: Path) -> list[int] | None:
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected token list in {path}, got {type(data)}")
    return [int(x) for x in data]


def _load_tokens_from_txt(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    return [int(part) for part in path.read_text().split()]


def _load_zarr_metadata(split_path: Path) -> tuple[str, dict[str, Any]] | None:
    for name in ("zarr.json", ".zarray"):
        meta_path = split_path / name
        if meta_path.exists():
            return name, json.loads(meta_path.read_text())
    return None


def _infer_shape_len(metadata_name: str, payload: dict[str, Any]) -> int:
    shape = payload.get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        return int(shape[0])
    raise ValueError(f"Could not infer 1D shape from {metadata_name}")


def _infer_chunk_len(metadata_name: str, payload: dict[str, Any]) -> int:
    if metadata_name == ".zarray":
        chunks = payload.get("chunks")
        if isinstance(chunks, list) and len(chunks) == 1:
            return int(chunks[0])
        raise ValueError("Could not infer chunks from .zarray metadata")

    chunk_grid = payload.get("chunk_grid")
    if isinstance(chunk_grid, dict):
        chunk_shape = chunk_grid.get("configuration", {}).get("chunk_shape")
        if isinstance(chunk_shape, list) and len(chunk_shape) == 1:
            return int(chunk_shape[0])
    raise ValueError("Could not infer chunk_shape from zarr.json metadata")


def _infer_dtype(metadata_name: str, payload: dict[str, Any]) -> np.dtype:
    if metadata_name == ".zarray":
        dtype_name = payload.get("dtype")
        if not isinstance(dtype_name, str):
            raise ValueError("Missing dtype in .zarray metadata")
        return np.dtype(dtype_name)

    dtype_name = payload.get("data_type")
    if not isinstance(dtype_name, str):
        raise ValueError("Missing data_type in zarr.json metadata")

    endian = "little"
    for codec in payload.get("codecs", []):
        if codec.get("name") == "bytes":
            endian = str(codec.get("configuration", {}).get("endian", "little"))
            break

    base = {
        "uint8": "u1",
        "uint16": "u2",
        "uint32": "u4",
        "uint64": "u8",
        "int8": "i1",
        "int16": "i2",
        "int32": "i4",
        "int64": "i8",
    }.get(dtype_name)
    if base is None:
        raise ValueError(f"Unsupported Zarr v3 data_type for fingerprinting: {dtype_name}")

    prefix = "<" if endian == "little" else ">"
    if base.endswith("1"):
        prefix = "|"
    return np.dtype(prefix + base)


def _iter_zarr_chunk_paths(split_path: Path) -> list[Path]:
    chunk_dir = split_path / "c"
    if chunk_dir.exists():
        entries = [path for path in chunk_dir.iterdir() if path.is_file() and path.name.isdigit()]
        return sorted(entries, key=lambda path: int(path.name))

    entries = [path for path in split_path.iterdir() if path.is_file() and path.name.isdigit()]
    return sorted(entries, key=lambda path: int(path.name))


def _fingerprint_zarr_split(split_path: Path) -> tuple[int, str]:
    meta_info = _load_zarr_metadata(split_path)
    if meta_info is None:
        raise FileNotFoundError(f"No Zarr metadata found in {split_path}")

    metadata_name, metadata = meta_info
    shape_len = _infer_shape_len(metadata_name, metadata)
    chunk_len = _infer_chunk_len(metadata_name, metadata)
    dtype = _infer_dtype(metadata_name, metadata)
    chunk_paths = _iter_zarr_chunk_paths(split_path)
    expected_chunks = (shape_len + chunk_len - 1) // chunk_len
    if len(chunk_paths) != expected_chunks:
        raise ValueError(
            f"Expected {expected_chunks} chunk files in {split_path}, found {len(chunk_paths)}"
        )

    h = hashlib.sha256()
    header = {
        "format": "zarr",
        "metadata_name": metadata_name,
        "dtype": dtype.str,
        "shape": [shape_len],
        "chunk_len": chunk_len,
    }
    h.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    remaining = shape_len
    for chunk_index, chunk_path in enumerate(chunk_paths):
        expected_items = min(chunk_len, remaining)
        expected_bytes = expected_items * dtype.itemsize
        raw = chunk_path.read_bytes()
        if len(raw) < expected_bytes:
            raise ValueError(
                f"Chunk {chunk_index} in {split_path} is shorter than expected: "
                f"{len(raw)} < {expected_bytes}"
            )
        h.update(raw[:expected_bytes])
        remaining -= expected_items

    if remaining != 0:
        raise ValueError(f"Fingerprint accounting error for {split_path}: remaining={remaining}")

    return shape_len, h.hexdigest()


def _fingerprint_split(root: Path, split: str) -> tuple[int, str, str]:
    json_tokens = _load_tokens_from_json(root / f"{split}.json")
    if json_tokens is not None:
        return len(json_tokens), _tokens_sha256(json_tokens), "json_token_list"

    txt_tokens = _load_tokens_from_txt(root / f"{split}.txt")
    if txt_tokens is not None:
        return len(txt_tokens), _tokens_sha256(txt_tokens), "txt_token_list"

    split_path = root / split
    if split_path.exists():
        meta_info = _load_zarr_metadata(split_path)
        if meta_info is not None:
            num_tokens, sha = _fingerprint_zarr_split(split_path)
            return num_tokens, sha, "zarr_storage_bytes"

    raise FileNotFoundError(
        f"No supported token source found for split '{split}' in {root}. "
        "Expected one of: {split}.json, {split}.txt, or a Zarr split directory."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic split-level dataset fingerprint for token streams."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-id", default="unknown")
    parser.add_argument("--tokenizer-revision", default="unknown")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.path).expanduser().resolve()
    num_tokens, sha, fingerprint_kind = _fingerprint_split(root, split=args.split)

    ref = DatasetRef(
        dataset_id=args.dataset_id,
        path=str(root),
        split=args.split,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        num_tokens=num_tokens,
        sha256=sha,
    )

    payload = {
        "schema_version": ref.schema_version,
        "dataset": ref.to_dict(),
        "fingerprint_kind": fingerprint_kind,
    }

    out_path = args.out
    if out_path is None:
        out_path = root / f"{args.split}.fingerprint.json"
    else:
        out_path = out_path.expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote fingerprint: {out_path}")
    print(f"  num_tokens={num_tokens} sha256={sha} kind={fingerprint_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
