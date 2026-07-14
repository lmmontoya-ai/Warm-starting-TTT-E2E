#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ttt.research.e2a_proxy import (
    DEFAULT_EXAMPLE_SEED,
    DEFAULT_POSITIONS,
    PRIMARY_COMPARISON_ID,
    apply_secondary_holm,
    compare_conditions,
    generate_example_manifest,
    load_manifest,
    validate_condition_rows,
    write_manifest,
    write_rows_csv,
)


def _parse_positions(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mock_correct(*, example_id: str, condition_id: str, mode: str) -> bool:
    digest = hashlib.sha256(f"{condition_id}:{example_id}".encode("utf-8")).digest()
    bucket = digest[0] / 255.0
    if mode == "perfect":
        return True
    if mode == "zero":
        return False
    if mode == "strong":
        return bucket < 0.75
    if mode == "medium":
        return bucket < 0.50
    if mode == "weak":
        return bucket < 0.25
    raise ValueError(f"Unknown mock score mode: {mode}")


def cmd_generate_manifest(args: argparse.Namespace) -> int:
    manifest = generate_example_manifest(
        num_examples=args.num_examples,
        seed=args.seed,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        candidates=args.candidates,
        positions=_parse_positions(args.positions),
        scale=args.scale,
    )
    write_manifest(args.out, manifest)
    print(json.dumps({"out": str(args.out), "manifest_hash": manifest["manifest_hash"]}, indent=2))
    return 0


def cmd_mock_score(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    for example in manifest["examples"]:
        correct = _mock_correct(
            example_id=str(example["example_id"]),
            condition_id=args.condition_id,
            mode=args.mode,
        )
        needle = int(example["needle"])
        wrong_prediction = next(int(candidate) for candidate in example["candidates"] if int(candidate) != needle)
        rows.append(
            {
                "schema_version": "1.0",
                "record_type": "e2a_proxy_example",
                "example_id": example["example_id"],
                "condition_id": args.condition_id,
                "stage_id": args.stage_id,
                "run_id": args.run_id,
                "checkpoint_id": args.checkpoint_id,
                "context_length": int(example["context_length"]),
                "position_fraction": float(example["position_fraction"]),
                "position_index": int(example["position_index"]),
                "needle": needle,
                "candidates": ",".join(str(x) for x in example["candidates"]),
                "prediction": needle if correct else wrong_prediction,
                "correct": correct,
                "manifest_hash": manifest["manifest_hash"],
                "binarization_rule": manifest["binarization_rule"],
                "raw_model_output": "mock_correct" if correct else "mock_incorrect",
                "seed": int(manifest["seed"]),
            }
        )
    write_rows_csv(args.out_csv, rows)
    print(json.dumps({"out_csv": str(args.out_csv), "rows": len(rows)}, indent=2))
    return 0


def _load_conditions(paths: list[Path], *, expected_manifest_hash: str) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for path in paths:
        rows = _read_csv(path)
        condition = validate_condition_rows(rows, expected_manifest_hash=expected_manifest_hash)
        if condition.stage_id in conditions:
            raise ValueError(f"Duplicate condition stage_id: {condition.stage_id}")
        conditions[condition.stage_id] = condition
    return conditions


def _parse_comparison(raw: str) -> tuple[str, str, str, str]:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) not in {3, 4}:
        raise ValueError(
            "Comparison must be comparison_id:model_a_stage:model_b_stage[:alternative], "
            f"got {raw!r}"
        )
    comparison_id, model_a, model_b = parts[:3]
    alternative = parts[3] if len(parts) == 4 else "greater"
    return comparison_id, model_a, model_b, alternative


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Revision V2 E2a Proxy Analysis",
        "",
        "| label | comparison | A | B | acc A | acc B | diff | discordant | p | adjusted p | verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {hierarchy} | {comparison_id} | {model_a} | {model_b} | "
            "{model_a_accuracy:.4f} | {model_b_accuracy:.4f} | "
            "{accuracy_difference_a_minus_b:.4f} | {discordant_pairs} | "
            "{mcnemar_p:.6f} | {mcnemar_p_adjusted:.6f} | {expansion_verdict} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Primary comparison is uncorrected by preregistration. Secondary comparisons are exploratory and Holm-corrected within the secondary family.",
            "Proxy-only evidence may support a complementarity finding or hypothesis, but not a headline claim without E2b corroboration.",
        ]
    )
    return "\n".join(lines) + "\n"


def cmd_aggregate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    conditions = _load_conditions(args.condition_csv, expected_manifest_hash=manifest["manifest_hash"])
    rows: list[dict[str, Any]] = []
    for raw in args.comparison:
        comparison_id, model_a, model_b, alternative = _parse_comparison(raw)
        if model_a not in conditions:
            raise ValueError(f"Missing condition for model_a={model_a}")
        if model_b not in conditions:
            raise ValueError(f"Missing condition for model_b={model_b}")
        rows.append(
            compare_conditions(
                comparison_id=comparison_id,
                model_a=conditions[model_a],
                model_b=conditions[model_b],
                requested_examples=int(manifest["num_examples"]),
                alternative=alternative,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    rows = apply_secondary_holm(rows)
    payload = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "manifest_hash": manifest["manifest_hash"],
        "num_examples": int(manifest["num_examples"]),
        "context_length": int(manifest["context_length"]),
        "binarization_rule": manifest["binarization_rule"],
        "primary_comparison_id": PRIMARY_COMPARISON_ID,
        "rows": rows,
    }
    _write_json(args.out_json, payload)
    write_rows_csv(args.out_csv, rows)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown_summary(rows), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_csv": str(args.out_csv), "out_md": str(args.out_md)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revision-v2 E2a paired retrieval-proxy harness.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate-manifest", help="Generate deterministic E2a example manifest.")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--num-examples", type=int, default=500)
    gen.add_argument("--seed", type=int, default=DEFAULT_EXAMPLE_SEED)
    gen.add_argument("--context-length", type=int, default=32768)
    gen.add_argument("--vocab-size", type=int, required=True)
    gen.add_argument("--candidates", type=int, default=16)
    gen.add_argument("--positions", default=",".join(str(x) for x in DEFAULT_POSITIONS))
    gen.add_argument("--scale", default="125M")
    gen.set_defaults(func=cmd_generate_manifest)

    mock = sub.add_parser("mock-score", help="Create deterministic mock per-example condition output.")
    mock.add_argument("--manifest", type=Path, required=True)
    mock.add_argument("--condition-id", required=True)
    mock.add_argument("--stage-id", required=True)
    mock.add_argument("--run-id", required=True)
    mock.add_argument("--checkpoint-id", required=True)
    mock.add_argument("--mode", choices=["perfect", "zero", "strong", "medium", "weak"], default="medium")
    mock.add_argument("--out-csv", type=Path, required=True)
    mock.set_defaults(func=cmd_mock_score)

    agg = sub.add_parser("aggregate", help="Aggregate paired E2a condition outputs.")
    agg.add_argument("--paper-run-id", default="revision_v2_e2a_proxy_v1")
    agg.add_argument("--manifest", type=Path, required=True)
    agg.add_argument("--condition-csv", type=Path, action="append", required=True)
    agg.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="comparison_id:model_a_stage:model_b_stage[:alternative]",
    )
    agg.add_argument("--bootstrap-resamples", type=int, default=10_000)
    agg.add_argument("--bootstrap-seed", type=int, default=0)
    agg.add_argument("--out-json", type=Path, required=True)
    agg.add_argument("--out-csv", type=Path, required=True)
    agg.add_argument("--out-md", type=Path, required=True)
    agg.set_defaults(func=cmd_aggregate)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
