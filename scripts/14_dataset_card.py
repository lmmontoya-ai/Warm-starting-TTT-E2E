#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path



def _load_fingerprint(path: Path) -> dict:
    payload = json.loads(path.read_text())
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"Invalid fingerprint payload: {path}")
    return dataset



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset card CSV/JSON from split fingerprint files."
    )
    parser.add_argument("--fingerprints", required=True, help="Comma-separated fingerprint JSON files")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    fp_paths = [Path(x.strip()).expanduser().resolve() for x in args.fingerprints.split(",") if x.strip()]
    if not fp_paths:
        raise ValueError("No fingerprint files provided")

    rows: list[dict] = []
    for path in fp_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing fingerprint file: {path}")
        rows.append(_load_fingerprint(path))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset_id",
        "path",
        "split",
        "tokenizer_id",
        "tokenizer_revision",
        "num_tokens",
        "sha256",
    ]
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"Wrote dataset card JSON: {args.json_out}")
    print(f"Wrote dataset card CSV:  {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
