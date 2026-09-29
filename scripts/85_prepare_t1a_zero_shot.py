#!/usr/bin/env python3
"""Prepare the deterministic, tokenizer-pinned T1a task manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

from ttt.research.tier1 import (
    load_tier1_config,
    make_manifest,
    sha256_file,
    sha256_payload,
    stable_rank,
    write_json,
)
from ttt.research.tier1_zero_shot import (
    PROMPT_RENDERER_VERSION,
    encode_prompt_choice,
    render_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare pinned T1a zero-shot token manifests."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/tier1/t1a_manifest.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/tier1_hf"))
    parser.add_argument("--examples-per-task", type=int, default=None)
    parser.add_argument("--tasks", default="")
    return parser.parse_args()


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("datasets", "transformers", "huggingface-hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install Tier-1 dependencies with `uv sync --extra tier1`"
        ) from exc

    config = load_tier1_config(args.config)
    t1a = config["t1a"]
    pins = config["pins"]
    task_filter = {x.strip() for x in args.tasks.split(",") if x.strip()}
    tasks = [str(x) for x in t1a["tasks"] if not task_filter or str(x) in task_filter]
    if not tasks:
        raise ValueError("No T1a tasks selected")
    per_task = int(args.examples_per_task or t1a["examples_per_task"])
    if per_task <= 0:
        raise ValueError("examples-per-task must be positive")

    tokenizer_pin = pins["tokenizer"]
    tokenizer_files: dict[str, str] = {}
    for filename, expected_sha in tokenizer_pin["expected_files_sha256"].items():
        downloaded = Path(
            hf_hub_download(
                repo_id=tokenizer_pin["repo_id"],
                filename=str(filename),
                revision=tokenizer_pin["revision"],
                cache_dir=args.cache_dir,
            )
        )
        observed_sha = sha256_file(downloaded)
        if observed_sha != str(expected_sha):
            raise ValueError(
                f"Tokenizer file hash mismatch for {filename}: "
                f"{observed_sha} != {expected_sha}"
            )
        tokenizer_files[str(filename)] = observed_sha
    datasets_version = importlib.metadata.version("datasets")
    dataset_cache = args.cache_dir / f"datasets-{datasets_version}"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_pin["repo_id"],
        revision=tokenizer_pin["revision"],
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    bos_token_id = int(tokenizer.bos_token_id)
    if bos_token_id != 128000:
        raise ValueError(f"Expected Llama-3 BOS 128000, got {bos_token_id}")
    if len(tokenizer) != 128256:
        raise ValueError(f"Expected Llama-3 vocabulary 128256, got {len(tokenizer)}")

    examples: list[dict[str, Any]] = []
    exclusions: dict[str, list[dict[str, str]]] = {}
    task_counts: dict[str, int] = {}
    max_tokens = int(t1a["context_length"]) + 1

    for task in tasks:
        dataset_pin = pins["datasets"][task]
        dataset = load_dataset(
            dataset_pin["repo_id"],
            dataset_pin.get("config") or None,
            split=dataset_pin["split"],
            revision=dataset_pin["revision"],
            cache_dir=str(dataset_cache),
            trust_remote_code=bool(dataset_pin.get("trust_remote_code", False)),
        )
        candidates: list[tuple[str, int, Any]] = []
        for row_index, row in enumerate(dataset):
            rendered = render_example(task, dict(row), row_index=row_index)
            candidates.append(
                (
                    stable_rank(
                        salt=str(t1a["selection_salt"]),
                        task=task,
                        example_id=rendered.example_id,
                    ),
                    row_index,
                    rendered,
                )
            )
        candidates.sort(key=lambda item: (item[0], item[2].example_id))
        selected = 0
        task_exclusions: list[dict[str, str]] = []
        for selection_hash, row_index, rendered in candidates:
            encoded_choices: list[dict[str, Any]] = []
            rejection = ""
            for choice_index, choice in enumerate(rendered.choices):
                try:
                    tokens, choice_start = encode_prompt_choice(
                        tokenizer, rendered.prompt, choice, bos_token_id=bos_token_id
                    )
                except Exception as exc:
                    rejection = f"tokenization_error:{type(exc).__name__}:{exc}"
                    break
                if len(tokens) > max_tokens:
                    rejection = f"over_context:{len(tokens)}>{max_tokens}"
                    break
                encoded_choices.append(
                    {
                        "choice_index": choice_index,
                        "tokens": tokens,
                        "choice_start": int(choice_start),
                        "token_count": len(tokens),
                        "token_sha256": sha256_payload(tokens),
                    }
                )
            if rejection:
                task_exclusions.append(
                    {"example_id": rendered.example_id, "reason": rejection}
                )
                continue
            examples.append(
                {
                    "example_id": f"{task}:{rendered.example_id}",
                    "source_example_id": rendered.example_id,
                    "source_row_index": row_index,
                    "selection_hash": selection_hash,
                    "task": task,
                    "gold": int(rendered.gold),
                    "choice_count": len(encoded_choices),
                    "choices": encoded_choices,
                }
            )
            selected += 1
            if selected >= per_task:
                break
        if selected != per_task:
            raise RuntimeError(
                f"Selected only {selected}/{per_task} examples for {task}"
            )
        exclusions[task] = task_exclusions
        task_counts[task] = selected

    provenance = {
        "tier1_config_sha256": config["_config_sha256"],
        "prompt_renderer_version": PROMPT_RENDERER_VERSION,
        "tokenizer": dict(tokenizer_pin),
        "tokenizer_files": tokenizer_files,
        "datasets": {task: dict(pins["datasets"][task]) for task in tasks},
        "package_versions": _versions(),
        "selection_salt": str(t1a["selection_salt"]),
        "context_length": int(t1a["context_length"]),
        "examples_per_task": per_task,
        "task_counts": task_counts,
        "exclusions": exclusions,
    }
    manifest = make_manifest(
        manifest_type="t1a_zero_shot",
        paper_run_id=str(config["paper_run_id"]),
        provenance=provenance,
        examples=examples,
    )
    write_json(args.out, manifest)
    print(
        f"Wrote {args.out.resolve()} with {len(examples)} examples; "
        f"sha256={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
