"""Revision-v2 E2a paired retrieval-proxy helpers.

The functions in this module are intentionally model-agnostic. They define the
deterministic example manifest, per-example output schema, paired statistics,
hierarchy labels, and expansion-ladder decisions used by the E2a proxy run.
The GPU scorer should only produce per-example binary outcomes that conform to
this contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from ttt.research.preregistration import (
    ConfidenceInterval,
    exact_mcnemar,
    paired_binary_counts,
    wilson_interval,
)


PRIMARY_COMPARISON_ID = "125m_s2_vs_s3_32k"
PRIMARY_MODEL_A = "S2_125M"
PRIMARY_MODEL_B = "S3_125M"
PRIMARY_CONTEXT_LENGTH = 32768
DEFAULT_EXAMPLE_SEED = 20260610
DEFAULT_POSITIONS = (0.1, 0.5, 0.9)
DEFAULT_DISCORDANT_MIN = 50

ExpansionVerdict = Literal["adequate", "expand_1000", "expand_2000", "underpowered"]
HierarchyLabel = Literal["PRIMARY", "SECONDARY"]


@dataclass(frozen=True)
class E2aExample:
    example_id: str
    context_length: int
    position_fraction: float
    position_index: int
    needle: int
    candidates: tuple[int, ...]
    placeholder: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class E2aConditionOutput:
    condition_id: str
    stage_id: str
    run_id: str
    checkpoint_id: str
    context_length: int
    manifest_hash: str
    outcomes: dict[str, bool]


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Return a stable SHA256 over the canonical JSON manifest."""

    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _random_token_excluding(rng: random.Random, vocab_size: int, excluded: set[int]) -> int:
    while True:
        token = rng.randrange(vocab_size)
        if token not in excluded:
            return int(token)


def generate_example_manifest(
    *,
    num_examples: int,
    seed: int = DEFAULT_EXAMPLE_SEED,
    context_length: int = PRIMARY_CONTEXT_LENGTH,
    vocab_size: int,
    candidates: int = 16,
    positions: Sequence[float] = DEFAULT_POSITIONS,
    scale: str = "125M",
) -> dict[str, Any]:
    """Generate a deterministic NIAH-style example manifest.

    `num_examples` is the total number of examples, not examples per position.
    Positions are assigned by cycling through the committed position fractions.
    """

    if num_examples <= 0:
        raise ValueError(f"num_examples must be > 0, got {num_examples}")
    if context_length <= 1:
        raise ValueError(f"context_length must be > 1, got {context_length}")
    if vocab_size <= candidates:
        raise ValueError(f"vocab_size must exceed candidates, got vocab_size={vocab_size}")
    if candidates < 2:
        raise ValueError(f"candidates must be >= 2, got {candidates}")
    if not positions:
        raise ValueError("positions must be non-empty")
    for pos in positions:
        if not 0.0 <= float(pos) <= 1.0:
            raise ValueError(f"position fractions must be in [0, 1], got {pos}")

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for idx in range(num_examples):
        pos = float(positions[idx % len(positions)])
        pos_idx = int(round((context_length - 1) * pos))
        pos_idx = min(max(pos_idx, 0), context_length - 1)
        needle = int(rng.randrange(vocab_size))
        context = [
            _random_token_excluding(rng, vocab_size, {needle})
            for _ in range(context_length)
        ]
        context[pos_idx] = needle
        choice_set = [needle]
        seen = {needle}
        while len(choice_set) < candidates:
            cand = _random_token_excluding(rng, vocab_size, seen)
            choice_set.append(cand)
            seen.add(cand)
        rng.shuffle(choice_set)
        placeholder = _random_token_excluding(rng, vocab_size, {needle})
        examples.append(
            {
                "example_id": f"{scale.lower()}_ctx{context_length}_seed{seed}_ex{idx:06d}",
                "context_length": int(context_length),
                "position_fraction": pos,
                "position_index": int(pos_idx),
                "needle": int(needle),
                "candidates": [int(x) for x in choice_set],
                "placeholder": int(placeholder),
                "tokens": [*context, int(placeholder)],
            }
        )

    manifest = {
        "schema_version": "1.0",
        "task": "revision_v2_e2a_niah_proxy",
        "scale": scale,
        "seed": int(seed),
        "num_examples": int(num_examples),
        "context_length": int(context_length),
        "vocab_size": int(vocab_size),
        "candidates": int(candidates),
        "positions": [float(x) for x in positions],
        "binarization_rule": "candidate_argmax_equals_needle_token",
        "examples": examples,
    }
    manifest["manifest_hash"] = manifest_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected manifest object: {path}")
    expected = payload.get("manifest_hash")
    actual = manifest_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    if expected != actual:
        raise ValueError(f"Manifest hash mismatch for {path}: expected={expected!r} actual={actual!r}")
    return payload


def validate_condition_rows(rows: Sequence[dict[str, Any]], *, expected_manifest_hash: str) -> E2aConditionOutput:
    if not rows:
        raise ValueError("Condition output contains no rows")

    first = rows[0]
    required = {
        "example_id",
        "correct",
        "condition_id",
        "stage_id",
        "run_id",
        "checkpoint_id",
        "context_length",
        "manifest_hash",
    }
    missing = sorted(required - set(first))
    if missing:
        raise ValueError(f"Condition output missing required fields: {missing}")

    condition_id = str(first["condition_id"])
    stage_id = str(first["stage_id"])
    run_id = str(first["run_id"])
    checkpoint_id = str(first["checkpoint_id"])
    context_length = int(first["context_length"])
    manifest = str(first["manifest_hash"])
    if manifest != expected_manifest_hash:
        raise ValueError(
            f"Manifest hash mismatch for {condition_id}: expected={expected_manifest_hash} got={manifest}"
        )

    outcomes: dict[str, bool] = {}
    for row in rows:
        for key, value in {
            "condition_id": condition_id,
            "stage_id": stage_id,
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
            "manifest_hash": manifest,
        }.items():
            if str(row.get(key)) != value:
                raise ValueError(f"Inconsistent {key} in condition output for {condition_id}")
        if int(row.get("context_length", -1)) != context_length:
            raise ValueError(f"Inconsistent context_length in condition output for {condition_id}")
        example_id = str(row["example_id"])
        if example_id in outcomes:
            raise ValueError(f"Duplicate example_id in condition output: {example_id}")
        outcomes[example_id] = parse_binary_outcome(row["correct"])

    return E2aConditionOutput(
        condition_id=condition_id,
        stage_id=stage_id,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        context_length=context_length,
        manifest_hash=manifest,
        outcomes=outcomes,
    )


def parse_binary_outcome(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "correct"}:
        return True
    if text in {"0", "false", "no", "incorrect"}:
        return False
    raise ValueError(f"Cannot parse binary outcome: {raw!r}")


def expansion_verdict(
    *,
    effective_n: int,
    requested_examples: int,
    discordant_min: int = DEFAULT_DISCORDANT_MIN,
) -> ExpansionVerdict:
    if effective_n >= discordant_min:
        return "adequate"
    if requested_examples < 1000:
        return "expand_1000"
    if requested_examples < 2000:
        return "expand_2000"
    return "underpowered"


def comparison_hierarchy(comparison_id: str, *, model_a: str, model_b: str, context_length: int) -> HierarchyLabel:
    if (
        comparison_id == PRIMARY_COMPARISON_ID
        and model_a == PRIMARY_MODEL_A
        and model_b == PRIMARY_MODEL_B
        and int(context_length) == PRIMARY_CONTEXT_LENGTH
    ):
        return "PRIMARY"
    return "SECONDARY"


def paired_accuracy_difference_ci(
    model_a: dict[str, bool],
    model_b: dict[str, bool],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    if set(model_a) != set(model_b):
        paired_binary_counts(model_a, model_b)
    ids = sorted(model_a)
    if not ids:
        raise ValueError("Cannot bootstrap empty paired outcomes")
    diffs = [float(model_a[i]) - float(model_b[i]) for i in ids]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in ids:
            total += diffs[rng.randrange(len(diffs))]
        means.append(total / len(ids))
    means.sort()
    low = means[int(0.025 * n_resamples)]
    high = means[min(n_resamples - 1, int(0.975 * n_resamples))]
    return ConfidenceInterval(low=low, high=high)


def compare_conditions(
    *,
    comparison_id: str,
    model_a: E2aConditionOutput,
    model_b: E2aConditionOutput,
    requested_examples: int,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if model_a.manifest_hash != model_b.manifest_hash:
        raise ValueError(
            "Cannot compare outputs with different example manifests: "
            f"{model_a.condition_id}={model_a.manifest_hash} {model_b.condition_id}={model_b.manifest_hash}"
        )
    both_correct, a_only, b_only, both_wrong = paired_binary_counts(model_a.outcomes, model_b.outcomes)
    n = len(model_a.outcomes)
    a_success = sum(1 for value in model_a.outcomes.values() if value)
    b_success = sum(1 for value in model_b.outcomes.values() if value)
    mcnemar = exact_mcnemar(b=a_only, c=b_only, alternative=alternative)
    diff_ci = paired_accuracy_difference_ci(
        model_a.outcomes,
        model_b.outcomes,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    hierarchy = comparison_hierarchy(
        comparison_id,
        model_a=model_a.stage_id,
        model_b=model_b.stage_id,
        context_length=model_a.context_length,
    )
    return {
        "comparison_id": comparison_id,
        "hierarchy": hierarchy,
        "correction_family": "confirmatory_uncorrected" if hierarchy == "PRIMARY" else "secondary_holm",
        "model_a": model_a.stage_id,
        "model_b": model_b.stage_id,
        "run_a": model_a.run_id,
        "run_b": model_b.run_id,
        "context_length": model_a.context_length,
        "manifest_hash": model_a.manifest_hash,
        "n_examples": n,
        "requested_examples": int(requested_examples),
        "model_a_successes": a_success,
        "model_b_successes": b_success,
        "model_a_accuracy": a_success / float(n),
        "model_b_accuracy": b_success / float(n),
        "model_a_wilson_low": wilson_interval(a_success, n).low,
        "model_a_wilson_high": wilson_interval(a_success, n).high,
        "model_b_wilson_low": wilson_interval(b_success, n).low,
        "model_b_wilson_high": wilson_interval(b_success, n).high,
        "accuracy_difference_a_minus_b": (a_success - b_success) / float(n),
        "accuracy_difference_ci_low": diff_ci.low,
        "accuracy_difference_ci_high": diff_ci.high,
        "both_correct": both_correct,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
        "discordant_pairs": mcnemar.effective_n,
        "mcnemar_p": mcnemar.p_value,
        "mcnemar_alternative": mcnemar.alternative,
        "expansion_verdict": expansion_verdict(
            effective_n=mcnemar.effective_n,
            requested_examples=requested_examples,
        ),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values keyed by comparison id."""

    m = len(p_values)
    if m == 0:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted_raw: dict[str, float] = {}
    running = 0.0
    for rank, (key, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * float(p_value))
        running = max(running, adjusted)
        adjusted_raw[key] = running
    return adjusted_raw


def apply_secondary_holm(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    secondary = {
        str(row["comparison_id"]): float(row["mcnemar_p"])
        for row in out
        if row.get("hierarchy") == "SECONDARY"
    }
    adjusted = holm_adjust(secondary)
    for row in out:
        if row.get("hierarchy") == "PRIMARY":
            row["mcnemar_p_adjusted"] = row["mcnemar_p"]
            row["p_adjustment"] = "none_primary_uncorrected"
        else:
            row["mcnemar_p_adjusted"] = adjusted.get(str(row["comparison_id"]))
            row["p_adjustment"] = "holm_secondary_family"
    return out


def write_rows_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

