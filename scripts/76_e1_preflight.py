#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ttt.research.e1_preflight import run_e1_preflight
from ttt.research.registry import load_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revision-v2 E1 artifact preflight.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("./configs/research/warmstart_registry.yaml"),
    )
    parser.add_argument("--select", "--stage-ids", dest="stage_ids", default="S2_MINUS_125M")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", default="protocol_r_125m_main_v1")
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--canonical-books-root", type=Path, default=None)
    parser.add_argument("--canonical-dclm-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./reports/revision_v2/e1_preflight"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = {item.strip() for item in str(args.stage_ids).split(",") if item.strip()}
    if "S2_MINUS_125M" not in requested:
        raise ValueError("E1 preflight requires --select/--stage-ids to include S2_MINUS_125M")

    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(args.registry.expanduser().resolve())
    result = run_e1_preflight(
        repo_root=repo_root,
        registry=registry,
        checkpoint_root=args.checkpoint_root,
        exp_folder=args.exp_folder,
        books_root=args.books_root,
        dclm_root=args.dclm_root,
        canonical_books_root=args.canonical_books_root,
        canonical_dclm_root=args.canonical_dclm_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote E1 preflight manifest: {result.output_json}")
    print(f"PREFLIGHT: {result.status} ({result.checks_executed_count} checks executed)")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
