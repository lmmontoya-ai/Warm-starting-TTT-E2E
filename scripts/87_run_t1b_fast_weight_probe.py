#!/usr/bin/env python3
"""Prepare or execute the preregistered T1b fast-weight probe."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ttt.dataloader.lm_dataset import Dataset
from ttt.jax_runtime.fast_weight_probe import probe_batch
from ttt.jax_runtime.research_eval import load_session, make_padded_batch
from ttt.research.tier1 import (
    condition_spec,
    expected_restore_paths,
    load_tier1_config,
    make_manifest,
    read_json,
    read_restore_manifest,
    sha256_file,
    sha256_payload,
    verify_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier-1 T1b fast-weight instrumentation."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-manifest")
    prepare.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    prepare.add_argument("--books-root", type=Path, required=True)
    prepare.add_argument(
        "--out", type=Path, default=Path("artifacts/tier1/t1b_manifest.json")
    )
    prepare.add_argument("--examples", type=int, default=None)

    run = sub.add_parser("run")
    run.add_argument("--condition", required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    run.add_argument("--books-root", type=Path, required=True)
    run.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    run.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    run.add_argument(
        "--checkpoint-steps",
        default="all",
        help="all, final, or a comma-separated list of restored checkpoint steps",
    )
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--out-root", type=Path, default=Path("reports/revision_v2/tier1"))
    return parser.parse_args()


def _fingerprint_path(root: Path, split: str) -> Path:
    return root.expanduser().resolve() / f"{split}.fingerprint.json"


def _recompute_and_verify_fingerprint(
    *, root: Path, split: str, canonical: dict[str, Any]
) -> dict[str, Any]:
    script = Path(__file__).with_name("13_dataset_fingerprint.py")
    with tempfile.TemporaryDirectory(prefix="tier1-books-fingerprint-") as tmp:
        observed_path = Path(tmp) / "observed.json"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-id",
                "books3",
                "--path",
                str(root.expanduser().resolve()),
                "--split",
                split,
                "--tokenizer-id",
                str(canonical.get("dataset", {}).get("tokenizer_id", "unknown")),
                "--tokenizer-revision",
                str(canonical.get("dataset", {}).get("tokenizer_revision", "unknown")),
                "--out",
                str(observed_path),
            ],
            check=True,
        )
        observed = read_json(observed_path)
    for field in ("dataset_id", "split", "num_tokens", "sha256"):
        expected_value = canonical.get("dataset", {}).get(field)
        observed_value = observed.get("dataset", {}).get(field)
        if observed_value != expected_value:
            raise ValueError(
                f"Books3 fingerprint mismatch for {field}: "
                f"{observed_value!r} != {expected_value!r}"
            )
    if observed.get("fingerprint_kind") != canonical.get("fingerprint_kind"):
        raise ValueError("Books3 fingerprint kind differs from the canonical sidecar")
    return observed


def prepare_manifest(args: argparse.Namespace) -> int:
    config = load_tier1_config(args.config)
    spec = config["t1b"]
    examples = int(args.examples or spec["examples"])
    fingerprint_path = _fingerprint_path(args.books_root, str(spec["split"]))
    if not fingerprint_path.exists():
        raise FileNotFoundError(
            f"Missing mandatory Books3 fingerprint: {fingerprint_path}"
        )
    fingerprint = read_json(fingerprint_path)
    dataset_sha = str(fingerprint.get("dataset", {}).get("sha256", ""))
    if not dataset_sha:
        raise ValueError("Books3 fingerprint has no dataset.sha256")
    observed_fingerprint = _recompute_and_verify_fingerprint(
        root=args.books_root,
        split=str(spec["split"]),
        canonical=fingerprint,
    )
    source = Dataset(
        path=str(args.books_root.expanduser().resolve()),
        split=str(spec["split"]),
        seq_len=int(spec["context_length"]),
    )
    if len(source) < examples:
        raise ValueError(f"Books3 validation has only {len(source)} complete sequences")
    rows = [
        {
            "example_id": f"books3:{spec['split']}:{idx:06d}",
            "sequence_index": idx,
        }
        for idx in range(examples)
    ]
    manifest = make_manifest(
        manifest_type="t1b_fast_weight",
        paper_run_id=str(config["paper_run_id"]),
        provenance={
            "tier1_config_sha256": config["_config_sha256"],
            "dataset_id": "books3",
            "split": spec["split"],
            "dataset_fingerprint": fingerprint,
            "dataset_fingerprint_file_sha256": sha256_file(fingerprint_path),
            "recomputed_dataset_fingerprint": observed_fingerprint,
            "context_length": int(spec["context_length"]),
            "chunk_length": int(spec["chunk_length"]),
        },
        examples=rows,
    )
    write_json(args.out, manifest)
    print(
        json.dumps(
            {"path": str(args.out.resolve()), "sha256": manifest["manifest_sha256"]},
            indent=2,
        )
    )
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select_checkpoint_steps(
    checkpoint_dir: Path, requested: str, final_step: int
) -> list[int]:
    available = sorted(
        int(path.name)
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "model_weights").exists()
    )
    if not available:
        raise FileNotFoundError(
            f"No restored model_weights checkpoints under {checkpoint_dir}"
        )
    requested = requested.strip().lower()
    if requested == "all":
        selected = available
    elif requested == "final":
        selected = [final_step]
    else:
        selected = sorted(
            {int(part.strip()) for part in requested.split(",") if part.strip()}
        )
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(
            f"Requested T1b checkpoints are not restored: {missing}"
        )
    if final_step not in selected:
        raise ValueError("T1b checkpoint selection must include the final checkpoint")
    return selected


def _probe_checkpoint(
    *,
    session,
    source: Dataset,
    manifest_examples: list[dict[str, Any]],
    batch_size: int,
    context_length: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    chunks: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "weight_l2",
            "raw_grad_l2",
            "applied_update_l2",
            "relative_update_l2",
            "clip_crossed",
            "all_finite",
        )
    }
    example_ids: list[str] = []
    for start in range(0, len(manifest_examples), batch_size):
        selected = list(manifest_examples[start : start + batch_size])
        real_count = len(selected)
        while len(selected) < batch_size:
            selected.append(dict(selected[-1]))
        token_rows = [source[int(row["sequence_index"])].tolist() for row in selected]
        batch = make_padded_batch(
            token_rows,
            choice_starts=None,
            context_length=context_length,
            pad_token_id=int(session.cfg.model.eos_token_id),
        )
        diagnostics = probe_batch(session, batch, global_batch_size=batch_size)
        for key, value in diagnostics.items():
            chunks[key].append(np.asarray(value[:real_count]))
        example_ids.extend(str(row["example_id"]) for row in selected[:real_count])
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}
    if not bool(np.all(arrays["all_finite"])):
        raise FloatingPointError("T1b produced non-finite gradients or updates")
    return arrays, example_ids


def run_probe(args: argparse.Namespace) -> int:
    config = load_tier1_config(args.config)
    t1b = config["t1b"]
    allowed = {
        str(x) for x in [*t1b["conditions"], *t1b.get("optional_conditions", [])]
    }
    if args.condition not in allowed:
        raise ValueError(f"Condition {args.condition} is not preregistered for T1b")
    manifest = read_json(args.manifest)
    manifest_hash = verify_manifest(manifest, expected_type="t1b_fast_weight")
    if int(manifest["example_count"]) != int(t1b["examples"]):
        raise ValueError("Production T1b manifest example count mismatch")
    fingerprint_path = _fingerprint_path(args.books_root, str(t1b["split"]))
    if (
        sha256_file(fingerprint_path)
        != manifest["provenance"]["dataset_fingerprint_file_sha256"]
    ):
        raise ValueError("Books3 fingerprint changed after T1b manifest preparation")

    condition = condition_spec(config, args.condition)
    run_dir, checkpoint_dir = expected_restore_paths(
        config=config,
        condition_id=args.condition,
        exp_dir=args.exp_dir,
        checkpoint_root=args.checkpoint_root,
    )
    restore = read_restore_manifest(run_dir)
    if restore.get("source_run_id") != condition.source_run_id:
        raise ValueError("T1b checkpoint lineage mismatch")
    checkpoint_steps = _select_checkpoint_steps(
        checkpoint_dir, args.checkpoint_steps, condition.checkpoint_step
    )
    source = Dataset(
        path=str(args.books_root.expanduser().resolve()),
        split=str(t1b["split"]),
        seq_len=int(t1b["context_length"]),
    )

    started = time.time()
    manifest_examples = list(manifest["examples"])
    out_dir = args.out_root / str(config["paper_run_id"]) / "t1b" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    final_shape = (0, 0, 0)
    for checkpoint_step in checkpoint_steps:
        session = load_session(
            resolved_config=run_dir / "resolved_config.yaml",
            checkpoint_path=checkpoint_dir / str(checkpoint_step),
            context_length=int(t1b["context_length"]),
            chunk_length=int(t1b["chunk_length"]),
        )
        if session.checkpoint_step != checkpoint_step:
            raise ValueError(f"T1b restored step mismatch for {checkpoint_step}")
        arrays, example_ids = _probe_checkpoint(
            session=session,
            source=source,
            manifest_examples=manifest_examples,
            batch_size=args.batch_size,
            context_length=int(t1b["context_length"]),
        )
        np.savez_compressed(
            out_dir / f"fast_weight_trace_step_{checkpoint_step:08d}.npz",
            example_ids=np.asarray(example_ids),
            **arrays,
        )
        examples, chunk_count, layer_count = arrays["applied_update_l2"].shape
        final_shape = (examples, chunk_count, layer_count)
        checkpoint_summaries.append(
            {
                "checkpoint_step": checkpoint_step,
                "clip_crossing_rate": float(np.mean(arrays["clip_crossed"])),
                "mean_raw_grad_l2": float(np.mean(arrays["raw_grad_l2"])),
                "mean_applied_update_l2": float(np.mean(arrays["applied_update_l2"])),
                "mean_relative_update_l2": float(np.mean(arrays["relative_update_l2"])),
            }
        )
        for example_idx in range(examples):
            for chunk_idx in range(chunk_count):
                for layer_idx in range(layer_count):
                    tidy_rows.append(
                        {
                            "condition_id": args.condition,
                            "checkpoint_step": checkpoint_step,
                            "example_id": example_ids[example_idx],
                            "chunk_index": chunk_idx,
                            "suffix_layer_index": layer_idx,
                            "weight_l2": float(
                                arrays["weight_l2"][example_idx, chunk_idx, layer_idx]
                            ),
                            "raw_grad_l2": float(
                                arrays["raw_grad_l2"][example_idx, chunk_idx, layer_idx]
                            ),
                            "applied_update_l2": float(
                                arrays["applied_update_l2"][
                                    example_idx, chunk_idx, layer_idx
                                ]
                            ),
                            "relative_update_l2": float(
                                arrays["relative_update_l2"][
                                    example_idx, chunk_idx, layer_idx
                                ]
                            ),
                            "clip_crossed": bool(
                                arrays["clip_crossed"][example_idx, chunk_idx]
                            ),
                            "all_finite": bool(
                                arrays["all_finite"][example_idx, chunk_idx, layer_idx]
                            ),
                        }
                    )
    _write_csv(out_dir / "fast_weight_trace.csv", tidy_rows)
    scorer_sha = sha256_payload(
        {
            "runner": sha256_file(Path(__file__)),
            "probe": sha256_file(
                Path(__file__).parents[1] / "ttt/jax_runtime/fast_weight_probe.py"
            ),
        }
    )
    examples, chunk_count, layer_count = final_shape
    final_summary = checkpoint_summaries[-1]
    summary = {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "experiment": "T1b",
        "instrumentation_version": t1b["instrumentation_version"],
        "condition": condition.to_dict(),
        "checkpoint_step": condition.checkpoint_step,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_trajectory_scope": (
            "extension_checkpoint_trajectory"
            if len(checkpoint_steps) > 1
            else "final_checkpoint_only"
        ),
        "manifest_sha256": manifest_hash,
        "tier1_config_sha256": config["_config_sha256"],
        "scorer_sha256": scorer_sha,
        "examples": examples,
        "chunks": chunk_count,
        "suffix_layers": layer_count,
        "clip_crossing_rate": final_summary["clip_crossing_rate"],
        "non_finite_count": 0,
        "checkpoint_summaries": checkpoint_summaries,
        "elapsed_seconds": float(time.time() - started),
        "status": "succeeded",
    }
    summary["run_manifest_sha256"] = sha256_payload(summary)
    write_json(out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare-manifest":
        return prepare_manifest(args)
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
