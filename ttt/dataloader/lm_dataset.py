"""Minimal sequence dataloader for phase-1 local experiments.

Design goals:
- Work without heavyweight runtime deps.
- Support deterministic dummy data for fast local validation.
- Allow simple token-stream sources from disk for early real-data checks.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SequenceSample:
    input_ids: list[int]
    target_tokens: list[int]


@dataclass(frozen=True)
class Batch:
    samples: list[SequenceSample]


def _sliding_windows(tokens: list[int], seq_len: int) -> list[list[int]]:
    n_windows = (len(tokens) - 1) // seq_len
    windows = []
    for i in range(n_windows):
        window = tokens[i * seq_len : (i + 1) * seq_len + 1]
        if len(window) == seq_len + 1:
            windows.append(window)
    return windows


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

    parts = path.read_text().split()
    return [int(p) for p in parts]


def _load_tokens_from_zarr(path: Path, split: str) -> list[int] | None:
    split_path = path / split
    if not split_path.exists():
        return None

    # Zarr support is optional in phase 1.
    if not ((split_path / ".zarray").exists() or (split_path / ".zgroup").exists()):
        return None

    try:
        import zarr
        import zarr.storage
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "Zarr dataset detected but zarr is not installed. "
            "Install zarr or provide train/val token files as JSON/TXT."
        ) from exc

    store = zarr.storage.LocalStore(str(path), read_only=True)
    arr = zarr.open_array(store, path=f"/{split}")
    return [int(x) for x in arr[:].tolist()]


def _load_token_stream(dataset_path: str, split: str) -> list[int]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    # Try simple local formats first for portability.
    json_tokens = _load_tokens_from_json(root / f"{split}.json")
    if json_tokens is not None:
        return json_tokens

    txt_tokens = _load_tokens_from_txt(root / f"{split}.txt")
    if txt_tokens is not None:
        return txt_tokens

    zarr_tokens = _load_tokens_from_zarr(root, split)
    if zarr_tokens is not None:
        return zarr_tokens

    raise FileNotFoundError(
        f"No supported token source found for split '{split}' in {root}. "
        "Expected one of: {split}.json, {split}.txt, or Zarr split directory."
    )


def _to_sample(window: list[int]) -> SequenceSample:
    return SequenceSample(input_ids=window[:-1], target_tokens=window[1:])


def _dummy_window_generator(
    *,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> Iterator[list[int]]:
    rng = random.Random(seed)
    while True:
        yield [rng.randrange(vocab_size) for _ in range(seq_len + 1)]


def _window_generator(
    *,
    tokens: list[int],
    seq_len: int,
    repeat: bool,
    shuffle: bool,
    seed: int,
) -> Iterator[list[int]]:
    windows = _sliding_windows(tokens, seq_len)
    if not windows:
        raise ValueError(
            f"Token stream too short for seq_len={seq_len}. Got {len(tokens)} tokens."
        )

    rng = random.Random(seed)
    while True:
        idxs = list(range(len(windows)))
        if shuffle:
            rng.shuffle(idxs)

        for idx in idxs:
            yield windows[idx]

        if not repeat:
            return


def build_batch_iterator(
    *,
    dataset_path: str,
    split: str,
    seq_len: int,
    global_batch_size: int,
    repeat: bool,
    shuffle: bool,
    seed: int,
    dummy_dataset: bool,
    vocab_size: int,
) -> Iterator[Batch]:
    if global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")

    if dummy_dataset:
        windows = _dummy_window_generator(seq_len=seq_len, vocab_size=vocab_size, seed=seed)
    else:
        tokens = _load_token_stream(dataset_path, split)
        windows = _window_generator(
            tokens=tokens,
            seq_len=seq_len,
            repeat=repeat,
            shuffle=shuffle,
            seed=seed,
        )

    while True:
        samples: list[SequenceSample] = []
        try:
            for _ in range(global_batch_size):
                window = next(windows)
                samples.append(_to_sample(window))
        except StopIteration:
            if samples:
                yield Batch(samples=samples)
            return

        yield Batch(samples=samples)
