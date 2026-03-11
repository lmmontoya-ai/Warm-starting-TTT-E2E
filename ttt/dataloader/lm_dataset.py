"""Dataset loaders for both phase-1 and parity JAX runtimes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import grain.python as grain
import jax
import numpy as np
import zarr.codecs
import zarr.storage

from ttt.jax_runtime.model.data import Batch as JaxBatch


@dataclass(frozen=True)
class SequenceSample:
    input_ids: list[int]
    target_tokens: list[int]


@dataclass(frozen=True)
class Batch:
    samples: list[SequenceSample]


def _sliding_windows(tokens: list[int], seq_len: int) -> list[list[int]]:
    n_windows = (len(tokens) - 1) // seq_len
    return [
        window
        for i in range(n_windows)
        if len(window := tokens[i * seq_len : (i + 1) * seq_len + 1]) == seq_len + 1
    ]


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


def _open_split_array(path: Path, split: str):
    split_path = path / split
    if not split_path.exists():
        return None
    if not (split_path / "zarr.json").exists() and not (split_path / ".zarray").exists():
        return None
    codec = zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=3,
        shuffle=zarr.codecs.BloscShuffle.shuffle,
    )
    store = zarr.storage.LocalStore(str(path), read_only=True)
    return zarr.open_array(store, path=f"/{split}", codec=codec)


def _load_tokens_from_zarr(path: Path, split: str) -> list[int] | None:
    arr = _open_split_array(path, split)
    if arr is None:
        return None
    return [int(x) for x in arr[:].tolist()]


def _load_token_stream(dataset_path: str, split: str) -> list[int]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    for loader in (
        lambda: _load_tokens_from_json(root / f"{split}.json"),
        lambda: _load_tokens_from_txt(root / f"{split}.txt"),
        lambda: _load_tokens_from_zarr(root, split),
    ):
        tokens = loader()
        if tokens is not None:
            return tokens

    raise FileNotFoundError(
        f"No supported token source found for split '{split}' in {root}. "
        f"Expected one of: {split}.json, {split}.txt, or Zarr split directory."
    )


def _to_sample(window: list[int]) -> SequenceSample:
    return SequenceSample(input_ids=window[:-1], target_tokens=window[1:])


def _dummy_window_generator(*, seq_len: int, vocab_size: int, seed: int) -> Iterator[list[int]]:
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
        raise ValueError(f"Token stream too short for seq_len={seq_len}. Got {len(tokens)} tokens.")
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
        windows = _window_generator(tokens=tokens, seq_len=seq_len, repeat=repeat, shuffle=shuffle, seed=seed)

    while True:
        samples: list[SequenceSample] = []
        try:
            for _ in range(global_batch_size):
                samples.append(_to_sample(next(windows)))
        except StopIteration:
            if samples:
                yield Batch(samples=samples)
            return
        yield Batch(samples=samples)


class Dataset(grain.RandomAccessDataSource):
    def __init__(self, *, path: str, split: str, seq_len: int):
        arr = _open_split_array(Path(path).expanduser().resolve(), split)
        if arr is None:
            raise FileNotFoundError(f"Expected Zarr split directory '{split}' under {path}")
        self.split = arr
        self.seq_len = int(seq_len)

    def __getitem__(self, idx):
        sample = self.split[idx * self.seq_len : (idx + 1) * self.seq_len + 1]
        if len(sample) != self.seq_len + 1:
            raise ValueError("Loader got a sequence with the wrong length.")
        return np.asarray(sample, dtype=np.int32)

    def __len__(self):
        return (int(self.split.shape[0]) - 1) // self.seq_len


class DummyDataset(grain.RandomAccessDataSource):
    def __init__(self, *, seq_len: int, num_tokens: int = 2**20):
        self.seq_len = int(seq_len)
        self.num_tokens = int(num_tokens)

    def __getitem__(self, idx):
        del idx
        return np.random.randint(0, 20, (self.seq_len + 1,), dtype=np.int32)

    def __len__(self):
        return (self.num_tokens - self.seq_len - 1) // self.seq_len


def _to_jax_batch(data: np.ndarray, *, bos_token_id: int) -> JaxBatch:
    tokens = np.asarray(data, dtype=np.int32)
    input_ids = tokens[..., :-1]
    target_tokens = tokens[..., 1:]
    loss_masks = (target_tokens != bos_token_id).astype(np.int32)
    position_ids = np.broadcast_to(np.arange(input_ids.shape[-1], dtype=np.int32), input_ids.shape)
    return JaxBatch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=loss_masks,
        attention_mask=None,
        position_ids=position_ids,
    )


def lm_dataset(
    *,
    path: str,
    split: str,
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    seed=None,
    repeat: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shuffle: bool = True,
) -> grain.MapDataset:
    del eos_token_id
    if shard_index is None:
        shard_index = jax.process_index()
    if shard_count is None:
        shard_count = jax.process_count()
    if global_batch_size % shard_count != 0:
        raise ValueError("global_batch_size must be divisible by shard_count")
    host_batch_size = global_batch_size // shard_count

    source = Dataset(path=path, split=split, seq_len=seq_len)
    ds = grain.MapDataset.source(source)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.map(lambda data: _to_jax_batch(data, bos_token_id=bos_token_id)).batch(
        batch_size=host_batch_size,
        drop_remainder=True,
    )

    if repeat:
        ds = ds.repeat()
    else:
        trimmed_length = (len(ds) // shard_count) * shard_count
        ds = ds[:trimmed_length]

    return ds[shard_index::shard_count]


def dummy_dataset(
    *,
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    repeat: bool = False,
    num_tokens: int = 2**20,
):
    del eos_token_id
    shard_index = jax.process_index()
    shard_count = jax.process_count()
    if global_batch_size % shard_count != 0:
        raise ValueError("global_batch_size must be divisible by shard_count")

    ds = grain.MapDataset.source(DummyDataset(seq_len=seq_len, num_tokens=num_tokens))
    ds = ds.map(lambda data: _to_jax_batch(data, bos_token_id=bos_token_id)).batch(
        batch_size=global_batch_size // shard_count,
        drop_remainder=True,
    )
    if repeat:
        ds = ds.repeat()
    return ds[shard_index::shard_count]
