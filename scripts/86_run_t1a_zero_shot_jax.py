#!/usr/bin/env python3
"""Score one Tier-1 T1a condition through the native JAX inference path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ttt.jax_runtime.research_eval import (
    load_session,
    make_continuation_score_masks,
    make_padded_batch,
    score_token_nll,
)
from ttt.research.tier1 import (
    condition_spec,
    expected_restore_paths,
    load_tier1_config,
    read_json,
    sha256_file,
    sha256_payload,
    verify_manifest,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native JAX T1a zero-shot scoring."
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--out-root", type=Path, default=Path("reports/revision_v2/tier1")
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_tier1_config(args.config)
    if args.condition not in {str(x) for x in config["conditions"]}:
        raise KeyError(f"Unknown condition {args.condition}")
    if args.condition not in {
        "FA_SEED_125M",
        "S2_125M_SEED001",
        "S2_MINUS_125M_SEED001",
        "S3_125M",
        "S2_BRIDGE_40PCT_125M",
    }:
        raise ValueError(f"Condition {args.condition} is not preregistered for T1a")
    spec = condition_spec(config, args.condition)
    manifest = read_json(args.manifest)
    manifest_hash = verify_manifest(manifest, expected_type="t1a_zero_shot")
    if manifest["paper_run_id"] != config["paper_run_id"]:
        raise ValueError("T1a manifest paper_run_id mismatch")
    expected_examples = int(config["t1a"]["examples_per_task"]) * len(
        config["t1a"]["tasks"]
    )
    if int(manifest["example_count"]) != expected_examples:
        raise ValueError(
            f"Production T1a requires {expected_examples} examples, got {manifest['example_count']}"
        )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    run_dir, checkpoint_dir = expected_restore_paths(
        config=config,
        condition_id=args.condition,
        exp_dir=args.exp_dir,
        checkpoint_root=args.checkpoint_root,
    )
    resolved_config = run_dir / "resolved_config.yaml"
    restore_manifest = run_dir / "hf_restore_manifest.json"
    if not resolved_config.exists() or not restore_manifest.exists():
        raise FileNotFoundError(f"Missing restored experiment metadata under {run_dir}")
    restored = read_json(restore_manifest)
    if (
        restored.get("source_stage_id") != spec.source_stage_id
        or restored.get("source_run_id") != spec.source_run_id
    ):
        raise ValueError("Checkpoint restore lineage does not match Tier-1 condition")

    context_length = int(config["t1a"]["context_length"])
    session = load_session(
        resolved_config=resolved_config,
        checkpoint_path=checkpoint_dir,
        context_length=context_length,
        chunk_length=context_length if spec.architecture == "fa" else 1024,
    )
    if session.checkpoint_step != spec.checkpoint_step:
        raise ValueError(
            f"Checkpoint step mismatch: expected {spec.checkpoint_step}, got {session.checkpoint_step}"
        )

    candidates: list[dict[str, Any]] = []
    for example in manifest["examples"]:
        for choice in example["choices"]:
            candidates.append(
                {
                    "example_id": str(example["example_id"]),
                    "task": str(example["task"]),
                    "gold": int(example["gold"]),
                    "choice_index": int(choice["choice_index"]),
                    "tokens": [int(x) for x in choice["tokens"]],
                    "choice_start": int(choice["choice_start"]),
                }
            )

    started = time.time()
    candidate_rows: list[dict[str, Any]] = []
    for start in range(0, len(candidates), args.batch_size):
        chunk = list(candidates[start : start + args.batch_size])
        real_count = len(chunk)
        while len(chunk) < args.batch_size:
            chunk.append(dict(chunk[-1]))
        token_rows = [row["tokens"] for row in chunk]
        choice_starts = [row["choice_start"] for row in chunk]
        # TTT adapts on every available causal-LM target. The independent
        # continuation mask below controls only the multiple-choice readout.
        batch = make_padded_batch(
            token_rows,
            choice_starts=None,
            context_length=context_length,
            pad_token_id=int(session.cfg.model.eos_token_id),
        )
        score_masks = make_continuation_score_masks(
            token_rows,
            choice_starts=choice_starts,
            context_length=context_length,
        )
        nll = score_token_nll(session, batch, global_batch_size=args.batch_size)
        masks = np.asarray(score_masks, dtype=np.float32)
        for idx in range(real_count):
            token_count = int(masks[idx].sum())
            if token_count <= 0:
                raise ValueError("Choice has no scored continuation tokens")
            total_nll = float(np.sum(nll[idx] * masks[idx], dtype=np.float64))
            if not math.isfinite(total_nll):
                raise FloatingPointError("Non-finite T1a score")
            source = chunk[idx]
            candidate_rows.append(
                {
                    "condition_id": args.condition,
                    "checkpoint_step": session.checkpoint_step,
                    "manifest_sha256": manifest_hash,
                    "example_id": source["example_id"],
                    "task": source["task"],
                    "gold": source["gold"],
                    "choice_index": source["choice_index"],
                    "choice_token_count": token_count,
                    "loglikelihood_sum": -total_nll,
                    "loglikelihood_mean": -total_nll / token_count,
                }
            )

    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_example.setdefault(str(row["example_id"]), []).append(row)
    example_rows: list[dict[str, Any]] = []
    for example_id, rows in sorted(by_example.items()):
        rows.sort(key=lambda row: int(row["choice_index"]))
        pred_sum = int(
            max(rows, key=lambda row: float(row["loglikelihood_sum"]))["choice_index"]
        )
        pred_norm = int(
            max(rows, key=lambda row: float(row["loglikelihood_mean"]))["choice_index"]
        )
        gold = int(rows[0]["gold"])
        example_rows.append(
            {
                "condition_id": args.condition,
                "checkpoint_step": session.checkpoint_step,
                "manifest_sha256": manifest_hash,
                "example_id": example_id,
                "task": rows[0]["task"],
                "gold": gold,
                "prediction_acc": pred_sum,
                "prediction_acc_norm": pred_norm,
                "acc": int(pred_sum == gold),
                "acc_norm": int(pred_norm == gold),
            }
        )

    scorer_files = [
        Path(__file__),
        Path(__file__).parents[1] / "ttt/jax_runtime/research_eval.py",
    ]
    scorer_sha = sha256_payload(
        {str(path.name): sha256_file(path) for path in scorer_files}
    )
    out_dir = args.out_root / str(config["paper_run_id"]) / "t1a" / args.condition
    write_jsonl(out_dir / "candidate_scores.jsonl", candidate_rows)
    write_jsonl(out_dir / "example_scores.jsonl", example_rows)
    _write_csv(out_dir / "example_scores.csv", example_rows)
    metrics: dict[str, Any] = {}
    for task in [*map(str, config["t1a"]["tasks"]), "macro"]:
        selected = (
            example_rows
            if task == "macro"
            else [row for row in example_rows if row["task"] == task]
        )
        metrics[task] = {
            "n": len(selected),
            "acc": float(np.mean([row["acc"] for row in selected])),
            "acc_norm": float(np.mean([row["acc_norm"] for row in selected])),
        }
    run_manifest = {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "experiment": "T1a",
        "harness_version": config["t1a"]["harness_version"],
        "condition": spec.to_dict(),
        "checkpoint_step": session.checkpoint_step,
        "manifest_sha256": manifest_hash,
        "tier1_config_sha256": config["_config_sha256"],
        "scorer_sha256": scorer_sha,
        "batch_size": args.batch_size,
        "context_length": context_length,
        "examples": len(example_rows),
        "elapsed_seconds": float(time.time() - started),
        "metrics": metrics,
        "status": "succeeded",
    }
    run_manifest["run_manifest_sha256"] = sha256_payload(run_manifest)
    write_json(out_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
