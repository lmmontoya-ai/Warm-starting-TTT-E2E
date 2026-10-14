#!/usr/bin/env python3
"""Write a deterministic split fingerprint (``<split>.fingerprint.json``).

The fingerprint logic lives in ``ttt.infra.dataset_fingerprint`` so that the toy
data generator (``scripts/04_make_token_data.py``) writes identical sidecars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ttt.infra.dataset_fingerprint import write_split_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic split-level dataset fingerprint for token streams."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-id", default="unknown")
    parser.add_argument("--tokenizer-revision", default="unknown")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file. Defaults to <path>/<split>.fingerprint.json, where the orchestrator looks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path, payload = write_split_fingerprint(
        root=Path(args.path),
        split=args.split,
        dataset_id=args.dataset_id,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        out=args.out,
    )
    dataset = payload["dataset"]
    print(f"Wrote fingerprint: {out_path}")
    print(
        f"  num_tokens={dataset['num_tokens']} sha256={dataset['sha256']} "
        f"kind={payload['fingerprint_kind']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
