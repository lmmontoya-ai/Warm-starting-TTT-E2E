#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttt.dataloader.lm_dataset import _load_token_stream
from ttt.research.types import DatasetRef



def _tokens_sha256(tokens: list[int]) -> str:
    h = hashlib.sha256()
    for token in tokens:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()



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
    tokens = _load_token_stream(str(root), split=args.split)
    sha = _tokens_sha256(tokens)

    ref = DatasetRef(
        dataset_id=args.dataset_id,
        path=str(root),
        split=args.split,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        num_tokens=len(tokens),
        sha256=sha,
    )

    payload = {
        "schema_version": ref.schema_version,
        "dataset": ref.to_dict(),
    }

    out_path = args.out
    if out_path is None:
        out_path = root / f"{args.split}.fingerprint.json"
    else:
        out_path = out_path.expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote fingerprint: {out_path}")
    print(f"  num_tokens={len(tokens)} sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
