#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate simple token-stream JSON files for phase-1 token_stats runtime."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--train-tokens", type=int, default=20000)
    parser.add_argument("--val-tokens", type=int, default=4000)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    def sample(n: int) -> list[int]:
        return [rng.randrange(args.vocab_size) for _ in range(n)]

    (out / "train.json").write_text(json.dumps(sample(args.train_tokens)))
    (out / "val.json").write_text(json.dumps(sample(args.val_tokens)))

    print(f"Wrote train.json ({args.train_tokens} tokens) and val.json ({args.val_tokens} tokens) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
