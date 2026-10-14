"""Deterministic split-level fingerprints for token datasets.

A fingerprint records how many tokens a split holds and a SHA-256 digest of
their values. The research orchestrator refuses to launch a stage unless each
dataset root it reads carries a ``<split>.fingerprint.json`` sidecar, so every
run manifest names the exact data it saw.

Supported layouts mirror the phase-1 loader in ``lm_dataset``: a JSON token
list (``<split>.json``), a whitespace-separated text file (``<split>.txt``), or
a 1-D Zarr array directory (``<split>/``). JSON and text lists are hashed by
token value; Zarr splits are hashed over their stored chunk bytes plus a small
header describing dtype, shape and chunking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ttt.research.types import DatasetRef

FINGERPRINT_SUFFIX = ".fingerprint.json"


def fingerprint_path(root: Path, split: str) -> Path:
    """Location of the sidecar the orchestrator looks for."""

    return Path(root) / f"{split}{FINGERPRINT_SUFFIX}"


def tokens_sha256(tokens: list[int]) -> str:
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


def fingerprint_split(root: Path, split: str) -> tuple[int, str, str]:
    """Return ``(num_tokens, sha256, fingerprint_kind)`` for one split."""

    root = Path(root)
    json_tokens = _load_tokens_from_json(root / f"{split}.json")
    if json_tokens is not None:
        return len(json_tokens), tokens_sha256(json_tokens), "json_token_list"

    txt_tokens = _load_tokens_from_txt(root / f"{split}.txt")
    if txt_tokens is not None:
        return len(txt_tokens), tokens_sha256(txt_tokens), "txt_token_list"

    split_path = root / split
    if split_path.exists():
        meta_info = _load_zarr_metadata(split_path)
        if meta_info is not None:
            num_tokens, sha = _fingerprint_zarr_split(split_path)
            return num_tokens, sha, "zarr_storage_bytes"

    raise FileNotFoundError(
        f"No supported token source found for split '{split}' in {root}. "
        f"Expected one of: {split}.json, {split}.txt, or a Zarr split directory."
    )


def build_fingerprint_payload(
    *,
    root: Path,
    split: str,
    dataset_id: str,
    tokenizer_id: str = "unknown",
    tokenizer_revision: str = "unknown",
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    num_tokens, sha, fingerprint_kind = fingerprint_split(root, split=split)
    ref = DatasetRef(
        dataset_id=dataset_id,
        path=str(root),
        split=split,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        num_tokens=num_tokens,
        sha256=sha,
    )
    return {
        "schema_version": ref.schema_version,
        "dataset": ref.to_dict(),
        "fingerprint_kind": fingerprint_kind,
    }


def write_split_fingerprint(
    *,
    root: Path,
    split: str,
    dataset_id: str,
    tokenizer_id: str = "unknown",
    tokenizer_revision: str = "unknown",
    out: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fingerprint ``root``/``split`` and write the JSON sidecar.

    The default destination is ``<root>/<split>.fingerprint.json``, which is
    where the orchestrator looks. Returns the written path and payload.
    """

    payload = build_fingerprint_payload(
        root=root,
        split=split,
        dataset_id=dataset_id,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    resolved_root = Path(payload["dataset"]["path"])
    out_path = fingerprint_path(resolved_root, split) if out is None else Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path, payload
