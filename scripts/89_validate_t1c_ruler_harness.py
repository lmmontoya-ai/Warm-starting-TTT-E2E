#!/usr/bin/env python3
"""Fail-closed validation gate for the pinned T1c RULER harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from ttt.research.ruler import (
    SCORER_VERSION,
    aggregate_string_match_all,
    string_match_all,
)
from ttt.research.tier1 import (
    load_tier1_config,
    read_json,
    sha256_file,
    sha256_payload,
    verify_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the T1c tokenizer/prompt/scorer harness."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--started-at", type=dt.date.fromisoformat, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/tier1/t1c_harness_validation.json")
    )
    parser.add_argument(
        "--deferral-out",
        type=Path,
        default=Path("reports/revision_v2/tier1/decision_log/T1C_DEFERRAL.md"),
    )
    return parser.parse_args()


def _working_days(start: dt.date, end: dt.date) -> int:
    if end < start:
        raise ValueError("started-at is in the future")
    return sum(
        1
        for offset in range((end - start).days + 1)
        if (start + dt.timedelta(days=offset)).weekday() < 5
    )


def _tiny_model_smoke() -> list[int]:
    """Exercise adaptation plus frozen decode without loading any real checkpoint."""
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jrandom

    from ttt.config import Config
    from ttt.jax_runtime.meta_inference import generate_one
    from ttt.jax_runtime.model.data import Batch
    from ttt.jax_runtime.model.transformer import MetaModel

    cfg = Config()
    cfg.training.train_mode = cfg.training.TrainMode.meta
    cfg.training.seq_length = 8
    cfg.training.spec_inner = ["language_model.**.suffix_blocks.feed_forward_prime.**"]
    cfg.training.optimizer_inner.lr = 0.1
    cfg.training.optimizer_inner.clip_gradient = 1.0
    cfg.model.vocab_size = 32
    cfg.model.output_size = 32
    cfg.model.hidden_size = 8
    cfg.model.intermediate_size = 16
    cfg.model.num_hidden_layers = 2
    cfg.model.num_attention_heads = 2
    cfg.model.seq_modeling_block = "SWA"
    cfg.model.mini_batch_size = 4
    cfg.model.sliding_window_size = 4
    cfg.model.seq_len = 16
    cfg.model.suffix_len = 1
    cfg.model.prime = True
    cfg.model.force_flash = False
    cfg.model.compute_dtype = "fp32"
    cfg.model.param_dtype = "fp32"
    cfg.model.state_dtype = "fp32"
    model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=jrandom.PRNGKey(0))
    tokens = jnp.asarray([1, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.int32)
    seq = Batch(
        input_ids=tokens,
        target_tokens=jnp.roll(tokens, -1),
        loss_masks=jnp.asarray([1, 1, 1, 1, 1, 1, 1, 0], dtype=jnp.int32),
        attention_mask=None,
        position_ids=jnp.arange(8, dtype=jnp.int32),
    )
    generated = generate_one(model, seq, state, max_new_tokens=3, eos_token_id=2)
    result = [int(x) for x in generated.tolist()]
    if len(result) != 3 or any(token < 0 or token >= 32 for token in result):
        raise AssertionError(f"Tiny meta-generation smoke failed: {result}")
    return result


def main() -> int:
    args = parse_args()
    config = load_tier1_config(args.config)
    manifest = read_json(args.smoke_manifest)
    manifest_hash = verify_manifest(manifest, expected_type="t1c_ruler_v1")
    provenance = manifest["provenance"]
    if provenance.get("tier1_config_sha256") != config["_config_sha256"]:
        raise ValueError("T1c smoke manifest used a different Tier-1 config")
    if (
        provenance.get("generator", {}).get("revision")
        != config["pins"]["ruler"]["generator_revision"]
    ):
        raise ValueError("T1c smoke generator revision mismatch")
    if provenance.get("tokenizer") != config["pins"]["tokenizer"]:
        raise ValueError("T1c smoke tokenizer pin mismatch")
    if not provenance.get("tokenizer_files"):
        raise ValueError("T1c smoke manifest has no tokenizer file hashes")
    if (
        provenance["tokenizer_files"]
        != config["pins"]["tokenizer"]["expected_files_sha256"]
    ):
        raise ValueError(
            "T1c tokenizer file fingerprints differ from the preregistered values"
        )
    if int(provenance.get("generation_seed", -1)) != int(
        config["t1c"]["generation_seed"]
    ):
        raise ValueError("T1c smoke generation seed mismatch")
    expected_tasks = [str(x) for x in config["t1c"]["tasks"]]
    observed_tasks = sorted({str(row["task"]) for row in manifest["examples"]})
    if observed_tasks != sorted(expected_tasks):
        raise ValueError(
            f"Smoke task set mismatch: {observed_tasks} != {sorted(expected_tasks)}"
        )
    if not manifest["examples"]:
        raise ValueError("T1c smoke manifest is empty")
    token_path = (
        args.smoke_manifest.parent / manifest["provenance"]["token_file"]["path"]
    )
    if sha256_file(token_path) != manifest["provenance"]["token_file"]["sha256"]:
        raise ValueError("T1c smoke token file fingerprint mismatch")
    tokens = np.load(token_path, mmap_mode="r")
    if tokens.shape != (
        int(manifest["example_count"]),
        int(config["t1c"]["context_length"]),
    ):
        raise ValueError(f"Unexpected T1c smoke token matrix shape: {tokens.shape}")
    for row in manifest["examples"]:
        task = str(row["task"])
        protocol = config["t1c"]["task_protocol"][task]
        if int(row["tokens_to_generate"]) != int(protocol["tokens_to_generate"]):
            raise ValueError(f"Generation limit mismatch for {row['example_id']}")
        token_row = np.asarray(tokens[int(row["token_row"])], dtype=np.int32)
        if sha256_payload(token_row.tolist()) != row["prompt_token_sha256"]:
            raise ValueError(f"Token row hash mismatch for {row['example_id']}")
        if int(row["original_prompt_tokens"]) + int(row["left_padding_tokens"]) != len(
            token_row
        ):
            raise ValueError(f"Padding accounting mismatch for {row['example_id']}")
        minimum_prompt_tokens = int(
            np.floor(
                int(config["t1c"]["context_length"])
                * float(config["t1c"]["minimum_original_prompt_fraction"])
            )
        )
        if int(row["original_prompt_tokens"]) < minimum_prompt_tokens:
            raise ValueError(
                f"RULER row is below the preregistered 32K floor: {row['example_id']}"
            )

    golden = {
        "case_insensitive": string_match_all("ANSWER: Magic42", ["magic42"]) == 1.0,
        "fractional_multi_reference": string_match_all("alpha only", ["alpha", "beta"])
        == 0.5,
        "no_match": string_match_all("gamma", ["alpha", "beta"]) == 0.0,
        "aggregate_percent": aggregate_string_match_all(
            ["alpha beta", "alpha"], [["alpha", "beta"], ["alpha", "beta"]]
        )
        == 75.0,
    }
    if not all(golden.values()):
        raise AssertionError(f"RULER golden scorer cases failed: {golden}")
    tiny_generated_tokens = _tiny_model_smoke()
    elapsed = _working_days(args.started_at, dt.date.today())
    limit = int(config["t1c"]["engineering_timebox_working_days"])
    if elapsed > limit:
        args.deferral_out.parent.mkdir(parents=True, exist_ok=True)
        args.deferral_out.write_text(
            "# T1c RULER deferral\n\n"
            f"Date: {dt.date.today().isoformat()}\n\n"
            f"The preregistered engineering time-box expired after {elapsed} working days "
            f"(limit: {limit}). T1c is deferred without inspecting production checkpoint "
            "results. T1a/T1b and manuscript preparation continue.\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"T1c engineering time-box expired: {elapsed}>{limit} working days"
        )
    scorer_files = [Path(__file__), Path(__file__).parents[1] / "ttt/research/ruler.py"]
    scorer_sha = sha256_payload({path.name: sha256_file(path) for path in scorer_files})
    report = {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "experiment": "T1c",
        "gate": "harness_validation",
        "status": "passed",
        "tier1_config_sha256": config["_config_sha256"],
        "smoke_manifest_sha256": manifest_hash,
        "generator_revision": config["pins"]["ruler"]["generator_revision"],
        "tokenizer_revision": config["pins"]["tokenizer"]["revision"],
        "scorer_version": SCORER_VERSION,
        "scorer_sha256": scorer_sha,
        "golden_cases": golden,
        "tiny_model_generated_tokens": tiny_generated_tokens,
        "smoke_examples": int(manifest["example_count"]),
        "started_at": args.started_at.isoformat(),
        "validated_at": dt.date.today().isoformat(),
        "working_days_elapsed": elapsed,
        "working_days_limit": limit,
    }
    report["report_sha256"] = sha256_payload(report)
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
