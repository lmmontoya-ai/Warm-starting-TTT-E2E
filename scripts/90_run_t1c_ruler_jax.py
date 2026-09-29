#!/usr/bin/env python3
"""Run one preregistered T1c condition through native TTT-E2E generation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ttt.jax_runtime.meta_inference import generate_batch
from ttt.jax_runtime.model.data import Batch
from ttt.jax_runtime.research_eval import load_session
from ttt.research.ruler import SCORER_VERSION, string_match_all
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
    parser = argparse.ArgumentParser(description="Run native JAX T1c RULER scoring.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/tier1_hf"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--out-root", type=Path, default=Path("reports/revision_v2/tier1")
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a non-paper tiny manifest gate; relaxes only the production count check.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_prompt_batch(
    token_rows: np.ndarray, *, pad_token_id: int, bos_token_id: int
) -> Batch:
    input_ids = np.asarray(token_rows, dtype=np.int32)
    target_tokens = np.concatenate(
        [input_ids[:, 1:], np.full((len(input_ids), 1), pad_token_id, dtype=np.int32)],
        axis=1,
    )
    loss_masks = (
        (target_tokens != pad_token_id) & (target_tokens != bos_token_id)
    ).astype(np.int32)
    loss_masks[:, -1] = 0  # Never adapt on a synthetic answer token.
    return Batch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=loss_masks,
        attention_mask=None,
        position_ids=np.broadcast_to(
            np.arange(input_ids.shape[1], dtype=np.int32), input_ids.shape
        ),
    )


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install Tier-1 dependencies with `uv sync --extra tier1`"
        ) from exc

    config = load_tier1_config(args.config)
    t1c = config["t1c"]
    allowed = {str(x) for x in [*t1c["conditions_125m"], *t1c["conditions_760m"]]}
    if args.condition not in allowed:
        raise ValueError(f"Condition {args.condition} is not preregistered for T1c")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    validation = read_json(args.validation_report)
    if (
        validation.get("status") != "passed"
        or validation.get("gate") != "harness_validation"
    ):
        raise ValueError("T1c production run requires a passed harness validation gate")
    if validation.get("tier1_config_sha256") != config["_config_sha256"]:
        raise ValueError("T1c validation used a different Tier-1 config")
    if validation.get("scorer_version") != SCORER_VERSION:
        raise ValueError("T1c validation scorer version mismatch")
    validation_core = {
        key: value for key, value in validation.items() if key != "report_sha256"
    }
    if sha256_payload(validation_core) != validation.get("report_sha256"):
        raise ValueError("T1c validation report hash mismatch")

    manifest = read_json(args.manifest)
    manifest_hash = verify_manifest(manifest, expected_type="t1c_ruler_v1")
    production_count = int(t1c["examples_per_task"]) * len(t1c["tasks"])
    required_count = int(manifest["example_count"]) if args.smoke else production_count
    if not args.smoke and int(manifest["example_count"]) != production_count:
        raise ValueError(
            f"Production T1c requires {production_count} examples, got {manifest['example_count']}"
        )
    if args.smoke and required_count > 10 * len(t1c["tasks"]):
        raise ValueError("T1c smoke manifests must remain tiny and non-paper")
    token_path = args.manifest.parent / manifest["provenance"]["token_file"]["path"]
    if sha256_file(token_path) != manifest["provenance"]["token_file"]["sha256"]:
        raise ValueError("T1c production token matrix fingerprint mismatch")
    tokens = np.load(token_path, mmap_mode="r")
    context_length = int(t1c["context_length"])
    if tokens.shape != (required_count, context_length):
        raise ValueError(f"T1c token matrix shape mismatch: {tokens.shape}")

    condition = condition_spec(config, args.condition)
    if condition.architecture != "ttt_e2e":
        raise ValueError("T1c is defined only for TTT-E2E conditions")
    run_dir, checkpoint_dir = expected_restore_paths(
        config=config,
        condition_id=args.condition,
        exp_dir=args.exp_dir,
        checkpoint_root=args.checkpoint_root,
    )
    restore = read_json(run_dir / "hf_restore_manifest.json")
    if (
        restore.get("source_stage_id") != condition.source_stage_id
        or restore.get("source_run_id") != condition.source_run_id
    ):
        raise ValueError("T1c checkpoint lineage mismatch")
    session = load_session(
        resolved_config=run_dir / "resolved_config.yaml",
        checkpoint_path=checkpoint_dir,
        context_length=context_length,
        chunk_length=int(config["t1b"]["chunk_length"]),
    )
    if session.checkpoint_step != condition.checkpoint_step:
        raise ValueError("T1c checkpoint step mismatch")

    tokenizer_pin = config["pins"]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_pin["repo_id"],
        revision=tokenizer_pin["revision"],
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    bos_token_id = int(tokenizer.bos_token_id)
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
    if pad_token_id != int(manifest["provenance"]["pad_token_id"]):
        raise ValueError("T1c tokenizer pad id differs from the prepared manifest")

    started = time.time()
    output_rows: list[dict[str, Any]] = []
    examples = list(manifest["examples"])
    for task in map(str, t1c["tasks"]):
        task_examples = [row for row in examples if str(row["task"]) == task]
        limit = int(t1c["task_protocol"][task]["tokens_to_generate"])
        for start in range(0, len(task_examples), args.batch_size):
            selected = list(task_examples[start : start + args.batch_size])
            real_count = len(selected)
            while len(selected) < args.batch_size:
                selected.append(dict(selected[-1]))
            matrix = np.stack(
                [
                    np.asarray(tokens[int(row["token_row"])], dtype=np.int32)
                    for row in selected
                ]
            )
            batch = _make_prompt_batch(
                matrix, pad_token_id=pad_token_id, bos_token_id=bos_token_id
            )
            generated = generate_batch(
                session,
                batch,
                global_batch_size=args.batch_size,
                max_new_tokens=limit,
            )
            for local_idx in range(real_count):
                source = selected[local_idx]
                generated_ids = [int(x) for x in generated[local_idx]]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
                score = string_match_all(prediction, source["outputs"])
                output_rows.append(
                    {
                        "condition_id": args.condition,
                        "checkpoint_step": session.checkpoint_step,
                        "manifest_sha256": manifest_hash,
                        "example_id": source["example_id"],
                        "task": task,
                        "prediction": prediction,
                        "generated_token_ids": generated_ids,
                        "references": source["outputs"],
                        "score_fraction": float(score),
                        "exact_all_references": int(score == 1.0),
                        "original_prompt_tokens": int(source["original_prompt_tokens"]),
                        "left_padding_tokens": int(source["left_padding_tokens"]),
                    }
                )

    scorer_files = [
        Path(__file__),
        Path(__file__).parents[1] / "ttt/jax_runtime/meta_inference.py",
        Path(__file__).parents[1] / "ttt/research/ruler.py",
    ]
    scorer_sha = sha256_payload({path.name: sha256_file(path) for path in scorer_files})
    experiment_name = "T1c_smoke" if args.smoke else "T1c"
    output_section = "t1c_smoke" if args.smoke else "t1c"
    out_dir = (
        args.out_root / str(config["paper_run_id"]) / output_section / args.condition
    )
    write_jsonl(out_dir / "example_scores.jsonl", output_rows)
    csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"generated_token_ids", "references"}
        }
        for row in output_rows
    ]
    _write_csv(out_dir / "example_scores.csv", csv_rows)
    metrics: dict[str, Any] = {}
    for task in [*map(str, t1c["tasks"]), "macro"]:
        selected = (
            output_rows
            if task == "macro"
            else [row for row in output_rows if row["task"] == task]
        )
        metrics[task] = {
            "n": len(selected),
            "official_string_match_all_percent": 100.0
            * float(np.mean([row["score_fraction"] for row in selected])),
            "exact_all_references_accuracy": float(
                np.mean([row["exact_all_references"] for row in selected])
            ),
        }
    run_manifest = {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "experiment": experiment_name,
        "harness_version": t1c["harness_version"],
        "condition": condition.to_dict(),
        "checkpoint_step": session.checkpoint_step,
        "manifest_sha256": manifest_hash,
        "validation_report_sha256": validation["report_sha256"],
        "tier1_config_sha256": config["_config_sha256"],
        "scorer_version": SCORER_VERSION,
        "scorer_sha256": scorer_sha,
        "batch_size": args.batch_size,
        "context_length": context_length,
        "examples": len(output_rows),
        "paper_evidence": not args.smoke,
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
