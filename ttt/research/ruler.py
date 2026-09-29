"""Pinned RULER-v1 scoring helpers used by the Tier-1 external slice."""

from __future__ import annotations

from typing import Iterable


SCORER_VERSION = "nvidia_ruler_string_match_all_v1"


def string_match_all(prediction: str, references: Iterable[str]) -> float:
    """Return NVIDIA/RULER's per-example fractional all-reference score.

    The upstream evaluator lowercases both sides and averages substring matches
    over all references. Task aggregation multiplies this fraction by 100.
    """

    refs = list(references)
    if not refs or not all(isinstance(ref, str) for ref in refs):
        raise ValueError("RULER references must be a non-empty string list")
    lowered = str(prediction).lower()
    return sum(float(ref.lower() in lowered) for ref in refs) / len(refs)


def aggregate_string_match_all(
    predictions: list[str], references: list[list[str]]
) -> float:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("RULER predictions/references must be non-empty and aligned")
    return (
        100.0
        * sum(
            string_match_all(prediction, refs)
            for prediction, refs in zip(predictions, references, strict=True)
        )
        / len(predictions)
    )
