#!/usr/bin/env python3
"""Generate and tokenize the preregistered external RULER-v1 subset.

The official generator is executed from an exact detached commit. Its prompt
text and answer-prefix are preserved byte-for-byte; only left padding is added
to satisfy the native TTT runtime's fixed 1,024-token chunk contract.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.metadata
import os
import runpy
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ttt.research.tier1 import (
    load_tier1_config,
    make_manifest,
    read_jsonl,
    sha256_file,
    sha256_payload,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the pinned T1c RULER subset.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/tier1/t1c_ruler")
    )
    parser.add_argument("--tools-root", type=Path, default=Path(".cache/tier1_tools"))
    parser.add_argument("--hf-cache", type=Path, default=Path(".cache/tier1_hf"))
    parser.add_argument("--tasks", default="")
    parser.add_argument("--examples-per-task", type=int, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=cwd, check=True, env=env)


def _checkout_exact(repo_url: str, revision: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").is_dir():
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                repo_url,
                str(destination),
            ]
        )
    _run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision])
    _run(["git", "-C", str(destination), "checkout", "--detach", revision])
    head = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != revision:
        raise RuntimeError(f"RULER checkout mismatch: expected {revision}, got {head}")
    _run(
        [
            "git",
            "-C",
            str(destination),
            "lfs",
            "pull",
            "--include=scripts/data/synthetic/json/english_words.json",
        ]
    )


def _prepare_official_auxiliary_data(
    generator_dir: Path, tasks: list[str]
) -> dict[str, str]:
    json_dir = generator_dir / "scripts" / "data" / "synthetic" / "json"
    english_words = json_dir / "english_words.json"
    if "cwe" in tasks and (
        not english_words.is_file() or english_words.stat().st_size < 1_000_000
    ):
        raise FileNotFoundError("Pinned RULER english_words.json LFS payload is absent")
    essays = json_dir / "PaulGrahamEssays.json"
    if "niah_multikey_1" in tasks and not essays.is_file():
        _run([sys.executable, "download_paulgraham_essay.py"], cwd=json_dir)
    if "niah_multikey_1" in tasks and (
        not essays.is_file() or essays.stat().st_size < 100_000
    ):
        raise FileNotFoundError(
            "Official Paul Graham essay preparation produced no usable corpus"
        )
    return {
        path.name: sha256_file(path)
        for path in (english_words, essays)
        if path.is_file()
    }


def _prepare_nltk_data(root: Path, ruler_pin: dict[str, Any]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    tokenizers_root = root / "tokenizers"
    tokenizers_root.mkdir(parents=True, exist_ok=True)
    revision = str(ruler_pin["nltk_data_revision"])
    expected = dict(ruler_pin["nltk_zip_sha256"])
    for resource in ("punkt", "punkt_tab"):
        resource_root = tokenizers_root / resource
        if resource_root.is_dir() and any(resource_root.rglob("*")):
            continue
        url = (
            "https://raw.githubusercontent.com/nltk/nltk_data/"
            f"{revision}/packages/tokenizers/{resource}.zip"
        )
        with urllib.request.urlopen(url) as response:
            payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != str(expected[resource]):
            raise ValueError(f"Pinned NLTK {resource} zip hash mismatch: {actual}")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                target = (tokenizers_root / member.filename).resolve()
                if not str(target).startswith(str(tokenizers_root.resolve())):
                    raise ValueError(f"Unsafe NLTK zip member: {member.filename}")
            archive.extractall(tokenizers_root)
    os.environ["NLTK_DATA"] = str(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("NLTK resource preparation produced no files")
    return {
        "revision": revision,
        "aggregate_sha256": sha256_payload(
            {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
        ),
        "file_count": str(len(files)),
    }


def _run_official_task(
    *,
    generator_dir: Path,
    raw_dir: Path,
    task: str,
    tokenizer_snapshot: Path,
    tokenizer,
    context_length: int,
    examples_per_task: int,
    random_seed: int,
    protocol: dict[str, Any],
) -> None:
    data_dir = generator_dir / "scripts" / "data"
    custom = yaml.safe_load(
        (generator_dir / "scripts" / "synthetic.yaml").read_text(encoding="utf-8")
    )
    task_custom = dict(custom[task])
    if str(task_custom["task"]) != str(protocol["generator_task"]):
        raise ValueError(f"RULER generator family mismatch for {task}")
    base_tasks = runpy.run_path(str(data_dir / "synthetic" / "constants.py"))["TASKS"]
    base = dict(base_tasks[task_custom["task"]])
    if int(base["tokens_to_generate"]) != int(protocol["tokens_to_generate"]):
        raise ValueError(f"RULER generation-limit mismatch for {task}")
    if str(protocol["metric"]) != "string_match_all":
        raise ValueError(f"Unsupported preregistered RULER metric for {task}")
    template = str(base["template"]) + str(base["answer_prefix"])
    # Reproduce prepare.py --prepare_for_ns exactly for the base template.
    model_template_token = len(tokenizer.tokenize("{task_template}"))
    command = [
        sys.executable,
        str(data_dir / "synthetic" / f"{task_custom['task']}.py"),
        "--save_dir",
        str(raw_dir),
        "--save_name",
        task,
        "--subset",
        "test",
        "--tokenizer_path",
        str(tokenizer_snapshot),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(context_length),
        "--tokens_to_generate",
        str(int(base["tokens_to_generate"])),
        "--num_samples",
        str(examples_per_task),
        "--random_seed",
        str(random_seed),
        "--template",
        template,
        "--model_template_token",
        str(model_template_token),
    ]
    for key, value in dict(task_custom["args"]).items():
        command.extend([f"--{key}", str(value)])
    _run(command, cwd=data_dir)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("huggingface-hub", "nltk", "numpy", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install Tier-1 dependencies with `uv sync --extra tier1`"
        ) from exc

    config = load_tier1_config(args.config)
    pins = config["pins"]
    t1c = config["t1c"]
    ruler_pin = pins["ruler"]
    requested = {part.strip() for part in args.tasks.split(",") if part.strip()}
    tasks = [
        str(task) for task in t1c["tasks"] if not requested or str(task) in requested
    ]
    if not tasks:
        raise ValueError("No T1c tasks selected")
    examples_per_task = int(args.examples_per_task or t1c["examples_per_task"])
    if examples_per_task <= 0:
        raise ValueError("examples-per-task must be positive")

    out_dir = args.out_dir.expanduser().resolve()
    raw_dir = out_dir / "official_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    generator_dir = args.tools_root.expanduser().resolve() / "NVIDIA_RULER"
    generator_revision = str(ruler_pin["generator_revision"])
    _checkout_exact(
        str(ruler_pin["generator_repo_url"]), generator_revision, generator_dir
    )

    tokenizer_pin = pins["tokenizer"]
    tokenizer_snapshot = Path(
        snapshot_download(
            repo_id=str(tokenizer_pin["repo_id"]),
            revision=str(tokenizer_pin["revision"]),
            cache_dir=str(args.hf_cache.expanduser().resolve()),
            allow_patterns=[
                "generation_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    ).resolve()
    if " " in str(tokenizer_snapshot):
        raise RuntimeError(
            "Official RULER generator cannot safely consume a tokenizer path with spaces"
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, use_fast=True)
    if len(tokenizer) != 128256:
        raise ValueError(f"Expected Llama-3 vocabulary 128256, got {len(tokenizer)}")
    bos_token_id = int(tokenizer.bos_token_id)
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
    context_length = int(t1c["context_length"])
    if context_length % int(config["t1b"]["chunk_length"]) != 0:
        raise ValueError(
            "T1c context length must be divisible by the native chunk length"
        )

    nltk_files = _prepare_nltk_data(
        args.tools_root.expanduser().resolve() / "nltk_data", ruler_pin
    )
    auxiliary_files = _prepare_official_auxiliary_data(generator_dir, tasks)
    expected_auxiliary = dict(ruler_pin["auxiliary_files_sha256"])
    for filename, expected_hash in expected_auxiliary.items():
        if filename in auxiliary_files and auxiliary_files[filename] != str(
            expected_hash
        ):
            raise ValueError(f"Pinned RULER auxiliary file hash mismatch: {filename}")
    if nltk_files["aggregate_sha256"] != str(
        ruler_pin["nltk_extracted_aggregate_sha256"]
    ):
        raise ValueError("Pinned NLTK extracted-resource aggregate hash mismatch")
    for task in tasks:
        if task not in t1c["task_protocol"]:
            raise KeyError(f"Missing t1c.task_protocol for {task}")
        if not args.skip_generate:
            _run_official_task(
                generator_dir=generator_dir,
                raw_dir=raw_dir,
                task=task,
                tokenizer_snapshot=tokenizer_snapshot,
                tokenizer=tokenizer,
                context_length=context_length,
                examples_per_task=examples_per_task,
                random_seed=int(t1c["generation_seed"]),
                protocol=dict(t1c["task_protocol"][task]),
            )

    token_path = out_dir / "prompt_tokens.npy"
    total_examples = len(tasks) * examples_per_task
    token_matrix = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.int32, shape=(total_examples, context_length)
    )
    examples: list[dict[str, Any]] = []
    raw_files: dict[str, dict[str, Any]] = {}
    row_index = 0
    for task in tasks:
        raw_path = raw_dir / task / "test.jsonl"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Official RULER output missing: {raw_path}")
        rows = read_jsonl(raw_path)
        if len(rows) != examples_per_task:
            raise RuntimeError(
                f"Official RULER count mismatch for {task}: {len(rows)} != {examples_per_task}"
            )
        raw_files[task] = {"path": str(raw_path), "sha256": sha256_file(raw_path)}
        protocol = dict(t1c["task_protocol"][task])
        generation_limit = int(protocol["tokens_to_generate"])
        for source_index, row in enumerate(rows):
            required = {"input", "outputs", "answer_prefix"}
            if not required.issubset(row):
                raise ValueError(
                    f"RULER row {task}:{source_index} missing {sorted(required - set(row))}"
                )
            outputs = row["outputs"]
            if (
                not isinstance(outputs, list)
                or not outputs
                or not all(isinstance(x, str) for x in outputs)
            ):
                raise TypeError(f"Invalid RULER outputs for {task}:{source_index}")
            prompt = str(row["input"]) + str(row["answer_prefix"])
            prompt_tokens = [
                bos_token_id,
                *tokenizer.encode(prompt, add_special_tokens=False),
            ]
            if len(prompt_tokens) > context_length:
                raise RuntimeError(
                    f"RULER prompt exceeds 32K after BOS for {task}:{source_index}: "
                    f"{len(prompt_tokens)}>{context_length}"
                )
            minimum_prompt_tokens = int(
                np.floor(
                    context_length * float(t1c["minimum_original_prompt_fraction"])
                )
            )
            if len(prompt_tokens) < minimum_prompt_tokens:
                raise RuntimeError(
                    f"RULER prompt is not a 32K example for {task}:{source_index}: "
                    f"{len(prompt_tokens)}<{minimum_prompt_tokens}"
                )
            left_padding = context_length - len(prompt_tokens)
            padded = np.asarray(
                [pad_token_id] * left_padding + prompt_tokens, dtype=np.int32
            )
            token_matrix[row_index] = padded
            examples.append(
                {
                    "example_id": f"{task}:{source_index:04d}",
                    "task": task,
                    "source_index": source_index,
                    "token_row": row_index,
                    "original_prompt_tokens": len(prompt_tokens),
                    "left_padding_tokens": left_padding,
                    "padded_prompt_tokens": context_length,
                    "prompt_token_sha256": sha256_payload(padded.tolist()),
                    "outputs": outputs,
                    "answer_prefix": str(row["answer_prefix"]),
                    "tokens_to_generate": generation_limit,
                    "official_length": int(row.get("length", -1)),
                }
            )
            row_index += 1
    token_matrix.flush()
    del token_matrix

    provenance = {
        "tier1_config_sha256": config["_config_sha256"],
        "generator": {
            "repo_url": str(ruler_pin["generator_repo_url"]),
            "revision": generator_revision,
            "published_rulerv1_pointer_revision": str(ruler_pin["revision"]),
            "nemo_skills_revision": str(ruler_pin["nemo_skills_revision"]),
        },
        "official_auxiliary_files": auxiliary_files,
        "nltk_resources": nltk_files,
        "tokenizer": dict(tokenizer_pin),
        "tokenizer_files": {
            path.name: sha256_file(path)
            for path in sorted(tokenizer_snapshot.iterdir())
            if path.is_file()
            and path.name
            in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json",
            }
        },
        "package_versions": _package_versions(),
        "tasks": tasks,
        "task_protocol": {task: dict(t1c["task_protocol"][task]) for task in tasks},
        "examples_per_task": examples_per_task,
        "generation_seed": int(t1c["generation_seed"]),
        "context_length": context_length,
        "minimum_original_prompt_fraction": float(
            t1c["minimum_original_prompt_fraction"]
        ),
        "padding_policy": "left_pad_with_eos_or_pad_to_native_chunk_boundary_32768",
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "raw_files": raw_files,
        "token_file": {"path": token_path.name, "sha256": sha256_file(token_path)},
    }
    manifest = make_manifest(
        manifest_type="t1c_ruler_v1",
        paper_run_id=str(config["paper_run_id"]),
        provenance=provenance,
        examples=examples,
    )
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        f"Wrote {manifest_path} with {len(examples)} examples; "
        f"sha256={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
