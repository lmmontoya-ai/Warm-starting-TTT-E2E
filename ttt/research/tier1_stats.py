"""Deterministic paired statistics used by Tier-1 reports."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def paired_bootstrap_delta(
    left: Iterable[float],
    right: Iterable[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    left_arr = np.asarray(list(left), dtype=np.float64)
    right_arr = np.asarray(list(right), dtype=np.float64)
    if left_arr.shape != right_arr.shape or left_arr.ndim != 1 or left_arr.size == 0:
        raise ValueError(
            "Paired bootstrap inputs must be non-empty vectors with equal shape"
        )
    if not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
        raise ValueError("Paired bootstrap inputs must be finite")
    deltas = left_arr - right_arr
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(int(resamples), deltas.size))
    means = deltas[indices].mean(axis=1)
    return {
        "n": float(deltas.size),
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "resamples": float(resamples),
        "seed": float(seed),
    }


def exact_mcnemar_two_sided(
    left_correct: Iterable[bool], right_correct: Iterable[bool]
) -> dict[str, float]:
    left = list(bool(x) for x in left_correct)
    right = list(bool(x) for x in right_correct)
    if len(left) != len(right) or not left:
        raise ValueError("McNemar inputs must be non-empty and paired")
    b = sum(1 for x, y in zip(left, right) if x and not y)
    c = sum(1 for x, y in zip(left, right) if not x and y)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {
        "left_only": float(b),
        "right_only": float(c),
        "discordant": float(n),
        "p_two_sided": float(p),
    }


def bootstrap_mean_curve(
    values: np.ndarray, *, resamples: int = 10_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[0] == 0:
        raise ValueError("Curve values must have shape [examples, ...]")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Curve values must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.shape[0], size=(int(resamples), arr.shape[0]))
    sampled = arr[indices].mean(axis=1)
    return (
        arr.mean(axis=0),
        np.quantile(sampled, 0.025, axis=0),
        np.quantile(sampled, 0.975, axis=0),
    )
