#!/usr/bin/env python3
"""Generate a toy token dataset for the token_stats and simulate runtimes.

Writes ``train.json`` and ``val.json`` (uniform random token ids) plus the
``train.fingerprint.json`` / ``val.fingerprint.json`` sidecars that the
research orchestrator (``scripts/23_warmstart_registry.py``) requires before it
launches a stage. The sidecars are written by the same code as
``scripts/13_dataset_fingerprint.py``.

The default size (70,000 tokens per split) leaves at least two windows for the
32K-context extension stages in the registry.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ttt.infra.dataset_fingerprint import write_split_fingerprint

DEFAULT_SPLIT_TOKENS = 70_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simple token-stream JSON files (and their fingerprint sidecars) "
            "for the phase-1 token_stats runtime."
        )
    )
    parser.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--train-tokens", type=int, default=DEFAULT_SPLIT_TOKENS)
    parser.add_argument("--val-tokens", type=int, default=DEFAULT_SPLIT_TOKENS)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset-id",
        default="toy_uniform_tokens",
        help="dataset_id recorded in the fingerprint sidecars and run manifests.",
    )
    args = parser.parse_args()

    for flag, value in (
        ("--train-tokens", args.train_tokens),
        ("--val-tokens", args.val_tokens),
        ("--vocab-size", args.vocab_size),
    ):
        if value <= 0:
            parser.error(f"{flag} must be positive, got {value}")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    def sample(n: int) -> list[int]:
        return [rng.randrange(args.vocab_size) for _ in range(n)]

    (out / "train.json").write_text(json.dumps(sample(args.train_tokens)))
    (out / "val.json").write_text(json.dumps(sample(args.val_tokens)))
    print(f"Wrote train.json ({args.train_tokens} tokens) and val.json ({args.val_tokens} tokens) to {out}")

    for split in ("train", "val"):
        fp_path, _ = write_split_fingerprint(
            root=out,
            split=split,
            dataset_id=args.dataset_id,
            tokenizer_id="none",
            tokenizer_revision="synthetic_uniform",
        )
        print(f"Wrote fingerprint: {fp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
