"""Revision-v2 preregistration statistics and validity checks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence


BridgeDecision = Literal["helps", "harmful", "negligible", "inconclusive"]


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float


@dataclass(frozen=True)
class BridgeEffectResult:
    deltas: tuple[float, ...]
    mean_delta: float
    ci95: ConfidenceInterval
    decision: BridgeDecision


@dataclass(frozen=True)
class McNemarResult:
    b: int
    c: int
    effective_n: int
    p_value: float
    alternative: str


@dataclass(frozen=True)
class ConfigComparison:
    equal: bool
    differences: tuple[str, ...]


@dataclass(frozen=True)
class FingerprintComparison:
    equal: bool
    differences: tuple[str, ...]


DEFAULT_E1_ALLOWED_CONFIG_PATHS = frozenset(
    {
        "stage_id",
        "run_id",
        "exp_name",
        "name",
        "notes",
        "tags",
        "lineage",
        "checkpoint",
        "checkpoint_parents",
        "parent_checkpoints",
        "resume_checkpoint_path",
        "resume_checkpoint_dir",
        "training.stage_id",
        "training.run_id",
        "training.exp_name",
        "training.paper_run_id",
        "training.output_dir",
        "training.run_dir",
        "training.resume_exp_name",
        "training.resume_checkpoint_path",
        "training.resume_checkpoint_dir",
    }
)


def paired_deltas(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    """Return elementwise left - right deltas for paired observations."""

    if len(left) != len(right):
        raise ValueError(f"Expected paired sequences of equal length, got {len(left)} and {len(right)}")
    if not left:
        raise ValueError("Expected at least one paired observation")
    return tuple(float(a) - float(b) for a, b in zip(left, right, strict=True))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap a confidence interval for the mean of paired deltas."""

    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    vals = tuple(float(v) for v in values)
    rng = random.Random(seed)
    means: list[float] = []
    n = len(vals)
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    alpha = 1.0 - confidence
    low_idx = max(min(int(math.floor((alpha / 2.0) * n_resamples)), n_resamples - 1), 0)
    high_idx = max(
        min(int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1, n_resamples - 1),
        0,
    )
    return ConfidenceInterval(low=means[low_idx], high=means[high_idx])


def classify_bridge_effect(
    *,
    mean_delta: float,
    ci95: ConfidenceInterval,
    margin: float = 0.10,
) -> BridgeDecision:
    """Classify the preregistered E1 bridge effect."""

    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    if mean_delta >= margin and ci95.low > 0:
        return "helps"
    if mean_delta <= -margin and ci95.high < 0:
        return "harmful"
    if ci95.low >= -margin and ci95.high <= margin:
        return "negligible"
    return "inconclusive"


def bridge_effect_from_losses(
    *,
    s2_minus_losses: Sequence[float],
    s2_losses: Sequence[float],
    margin: float = 0.10,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BridgeEffectResult:
    """Compute the E1 bridge-effect result from paired seed losses."""

    deltas = paired_deltas(s2_minus_losses, s2_losses)
    mean_delta = sum(deltas) / len(deltas)
    ci95 = bootstrap_mean_ci(deltas, n_resamples=n_resamples, seed=seed)
    decision = classify_bridge_effect(mean_delta=mean_delta, ci95=ci95, margin=margin)
    return BridgeEffectResult(deltas=deltas, mean_delta=mean_delta, ci95=ci95, decision=decision)


def wilson_interval(successes: int, n: int, *, z: float = 1.959963984540054) -> ConfidenceInterval:
    """Wilson binomial interval for a proportion."""

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(f"successes must be in [0, n], got {successes} of {n}")
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half_width = (
        z
        * math.sqrt((phat * (1.0 - phat) / n) + ((z * z) / (4.0 * n * n)))
        / denom
    )
    return ConfidenceInterval(low=max(0.0, center - half_width), high=min(1.0, center + half_width))


def _binom_pmf(k: int, n: int) -> float:
    return math.comb(n, k) * (0.5**n)


def exact_mcnemar(
    *,
    b: int,
    c: int,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> McNemarResult:
    """Exact McNemar/binomial test over discordant pairs.

    `b` is count(model_a correct, model_b wrong). `c` is count(model_a wrong,
    model_b correct). For the preregistered S2>S3 test, use
    `alternative="greater"`.
    """

    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        p_value = 1.0
    elif alternative == "greater":
        p_value = sum(_binom_pmf(k, n) for k in range(b, n + 1))
    elif alternative == "less":
        p_value = sum(_binom_pmf(k, n) for k in range(0, b + 1))
    elif alternative == "two-sided":
        tail = min(
            sum(_binom_pmf(k, n) for k in range(b, n + 1)),
            sum(_binom_pmf(k, n) for k in range(0, b + 1)),
        )
        p_value = min(1.0, 2.0 * tail)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return McNemarResult(b=b, c=c, effective_n=n, p_value=p_value, alternative=alternative)


def paired_binary_counts(
    model_a: dict[str, bool],
    model_b: dict[str, bool],
) -> tuple[int, int, int, int]:
    """Return paired binary counts (both_correct, a_only, b_only, both_wrong)."""

    ids_a = set(model_a)
    ids_b = set(model_b)
    if ids_a != ids_b:
        missing_a = sorted(ids_b - ids_a)
        missing_b = sorted(ids_a - ids_b)
        raise ValueError(
            "Unpaired example ids: "
            f"missing_from_a={missing_a[:5]} missing_from_b={missing_b[:5]}"
        )

    both_correct = a_only = b_only = both_wrong = 0
    for example_id in sorted(ids_a):
        a = bool(model_a[example_id])
        b = bool(model_b[example_id])
        if a and b:
            both_correct += 1
        elif a and not b:
            a_only += 1
        elif not a and b:
            b_only += 1
        else:
            both_wrong += 1
    return both_correct, a_only, b_only, both_wrong


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, child))
        return out
    if isinstance(payload, list):
        return {prefix: tuple(payload)}
    return {prefix: payload}


def compare_resolved_configs(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    allowed_paths: Iterable[str] = DEFAULT_E1_ALLOWED_CONFIG_PATHS,
) -> ConfigComparison:
    """Compare resolved configs while ignoring preregistered lineage fields."""

    allowed = set(allowed_paths)
    flat_left = _flatten(left)
    flat_right = _flatten(right)
    keys = sorted(set(flat_left) | set(flat_right))
    differences: list[str] = []
    for key in keys:
        if key in allowed or any(key.startswith(f"{path}.") for path in allowed):
            continue
        if flat_left.get(key) != flat_right.get(key):
            differences.append(key)
    return ConfigComparison(equal=not differences, differences=tuple(differences))


def _fingerprint_signature(payload: dict[str, Any]) -> tuple[str, str, str, int, str, str]:
    dataset = payload.get("dataset", payload)
    if not isinstance(dataset, dict):
        raise ValueError("Fingerprint payload must be a mapping or contain a mapping at key 'dataset'")
    return (
        str(dataset.get("dataset_id", "")),
        str(dataset.get("split", "")),
        str(dataset.get("sha256", "")),
        int(dataset.get("num_tokens", 0) or 0),
        str(dataset.get("tokenizer_id", "")),
        str(dataset.get("tokenizer_revision", "")),
    )


def compare_fingerprints(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label_left: str = "left",
    label_right: str = "right",
) -> FingerprintComparison:
    """Compare dataset fingerprint payloads by stable dataset identity fields."""

    sig_left = _fingerprint_signature(left)
    sig_right = _fingerprint_signature(right)
    if sig_left == sig_right:
        return FingerprintComparison(equal=True, differences=())

    names = ("dataset_id", "split", "sha256", "num_tokens", "tokenizer_id", "tokenizer_revision")
    differences = tuple(
        f"{name}: {label_left}={a!r} {label_right}={b!r}"
        for name, a, b in zip(names, sig_left, sig_right, strict=True)
        if a != b
    )
    return FingerprintComparison(equal=False, differences=differences)


def assert_params_only_restore(config: dict[str, Any]) -> None:
    """Validate params-only restore policy for S2-minus style conversion."""

    training = config.get("training", config)
    load_part = str(training.get("load_part", "")).lower()
    if load_part != "params":
        raise ValueError(f"Expected training.load_part=params, got {load_part!r}")


def assert_resume_lineage_direction(
    *,
    s2_extension_config: dict[str, Any],
    s2_minus_config: dict[str, Any],
    expected_s2_parent: str,
    expected_s2_minus_parent: str,
) -> None:
    """Validate that S2 resumes from bridge and S2-minus resumes from FA seed."""

    s2_training = s2_extension_config.get("training", s2_extension_config)
    minus_training = s2_minus_config.get("training", s2_minus_config)
    s2_parent = str(s2_training.get("resume_exp_name", ""))
    minus_parent = str(minus_training.get("resume_exp_name", ""))
    if s2_parent != expected_s2_parent:
        raise ValueError(
            f"Expected S2 extension to resume from {expected_s2_parent!r}, got {s2_parent!r}"
        )
    if minus_parent != expected_s2_minus_parent:
        raise ValueError(
            "Expected S2-minus extension to resume from "
            f"{expected_s2_minus_parent!r}, got {minus_parent!r}"
        )
    if s2_parent == minus_parent:
        raise ValueError("S2 and S2-minus resume lineage must differ")


def assert_fresh_optimizer_state(config: dict[str, Any]) -> None:
    """Validate that a run does not request optimizer-state restore."""

    training = config.get("training", config)
    load_part = str(training.get("load_part", "")).lower()
    if load_part in {"all", "optimizer", "opt_state"}:
        raise ValueError(f"Expected fresh optimizer state, got training.load_part={load_part!r}")
    for key in ("restore_optimizer_state", "resume_optimizer_state", "load_optimizer_state"):
        if bool(training.get(key, False)):
            raise ValueError(f"Expected fresh optimizer state, but {key}=true")


def assert_matching_extension_seed_policy(
    *,
    s2_extension_config: dict[str, Any],
    s2_minus_config: dict[str, Any],
) -> None:
    """Validate explicit, matched extension-stage RNG seeds.

    The E1 paired-seed design requires extension-stage seeds to be explicit and
    matched between S2 and S2-minus. The seed values must not depend on how much
    RNG the upstream bridge consumed.
    """

    s2_training = s2_extension_config.get("training", s2_extension_config)
    minus_training = s2_minus_config.get("training", s2_minus_config)
    for key in ("model_seed", "data_seed"):
        if key not in s2_training:
            raise ValueError(f"Expected explicit S2 extension training.{key}")
        if key not in minus_training:
            raise ValueError(f"Expected explicit S2-minus extension training.{key}")
        if int(s2_training[key]) != int(minus_training[key]):
            raise ValueError(
                f"Expected matched extension training.{key}: "
                f"S2={s2_training[key]!r} S2_MINUS={minus_training[key]!r}"
            )
