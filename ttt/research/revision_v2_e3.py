"""Revision-v2 E3 frontier planning constants and checks."""

from __future__ import annotations

from dataclasses import dataclass

from .types import StageSpec


E3_SEED = 1
CANONICAL_BRIDGE_PERCENT = 10
CANONICAL_BRIDGE_STEPS = 480
CANONICAL_EXTENSION_STEPS = 480
BRIDGE_CONTEXT_LENGTH = 8192
BRIDGE_GLOBAL_BATCH_SIZE = 64
EXTENSION_CONTEXT_LENGTH = 32768
EXTENSION_GLOBAL_BATCH_SIZE = 8
S2_MINUS_PARENT_FINAL_STEP = 479
S2_MINUS_CONTINUATION_ADDITIONAL_STEPS = 1440


@dataclass(frozen=True)
class E3ArmSpec:
    percent: int
    adapt_stage_id: str
    final_stage_id: str
    bridge_steps: int
    adapt_run_id: str
    final_run_id: str

    @property
    def bridge_tokens(self) -> int:
        return self.bridge_steps * BRIDGE_GLOBAL_BATCH_SIZE * BRIDGE_CONTEXT_LENGTH

    @property
    def extension_tokens(self) -> int:
        return CANONICAL_EXTENSION_STEPS * EXTENSION_GLOBAL_BATCH_SIZE * EXTENSION_CONTEXT_LENGTH


E3_ARMS: tuple[E3ArmSpec, ...] = (
    E3ArmSpec(
        percent=5,
        adapt_stage_id="S2_BRIDGE_5PCT_ADAPT_125M",
        final_stage_id="S2_BRIDGE_5PCT_125M",
        bridge_steps=240,
        adapt_run_id="adapt-125m-e2e-8K-from-fa-bridge5pct-seed001",
        final_run_id="ext-125m-e2e-32K-from-fa-bridge5pct-seed001",
    ),
    E3ArmSpec(
        percent=20,
        adapt_stage_id="S2_BRIDGE_20PCT_ADAPT_125M",
        final_stage_id="S2_BRIDGE_20PCT_125M",
        bridge_steps=960,
        adapt_run_id="adapt-125m-e2e-8K-from-fa-bridge20pct-seed001",
        final_run_id="ext-125m-e2e-32K-from-fa-bridge20pct-seed001",
    ),
    E3ArmSpec(
        percent=40,
        adapt_stage_id="S2_BRIDGE_40PCT_ADAPT_125M",
        final_stage_id="S2_BRIDGE_40PCT_125M",
        bridge_steps=1920,
        adapt_run_id="adapt-125m-e2e-8K-from-fa-bridge40pct-seed001",
        final_run_id="ext-125m-e2e-32K-from-fa-bridge40pct-seed001",
    ),
)


def e3_arm_by_percent(percent: int) -> E3ArmSpec:
    for arm in E3_ARMS:
        if arm.percent == int(percent):
            return arm
    raise KeyError(f"Unknown E3 bridge percent: {percent}")


def bridge_steps_for_percent(percent: int) -> int:
    """Return preregistered bridge steps for a bridge budget percentage."""

    numerator = int(percent) * CANONICAL_BRIDGE_STEPS
    if numerator % CANONICAL_BRIDGE_PERCENT != 0:
        raise ValueError(f"Bridge percent {percent} does not map to an integer step count")
    return numerator // CANONICAL_BRIDGE_PERCENT


def bridge_tokens_for_steps(steps: int) -> int:
    return int(steps) * BRIDGE_GLOBAL_BATCH_SIZE * BRIDGE_CONTEXT_LENGTH


def continuation_target_total_steps(
    *,
    parent_final_step: int = S2_MINUS_PARENT_FINAL_STEP,
    additional_steps: int = S2_MINUS_CONTINUATION_ADDITIONAL_STEPS,
) -> int:
    """Convert a final checkpoint step plus extra steps to absolute total_steps."""

    if parent_final_step < 0:
        raise ValueError(f"parent_final_step must be non-negative, got {parent_final_step}")
    if additional_steps <= 0:
        raise ValueError(f"additional_steps must be positive, got {additional_steps}")
    return int(parent_final_step) + 1 + int(additional_steps)


def assert_e3_budget_arithmetic() -> None:
    """Assert preregistered bridge-budget percent-to-step/token arithmetic."""

    expected = {5: 240, 10: 480, 20: 960, 40: 1920}
    for percent, steps in expected.items():
        observed = bridge_steps_for_percent(percent)
        if observed != steps:
            raise ValueError(f"{percent}% bridge expected {steps} steps, got {observed}")

    if bridge_tokens_for_steps(1920) != 4 * bridge_tokens_for_steps(480):
        raise ValueError("40% bridge tokens must be 4x the canonical 10% bridge tokens")
    if bridge_tokens_for_steps(240) != bridge_tokens_for_steps(480) // 2:
        raise ValueError("5% bridge tokens must be 0.5x the canonical 10% bridge tokens")


def assert_e3_registry_stages(stage_map: dict[str, StageSpec]) -> None:
    """Validate the registry topology for the E3 frontier and continuation."""

    assert_e3_budget_arithmetic()
    canonical_adapt = stage_map["S2_ADAPT_125M"]
    canonical_ext = stage_map["S2_125M"]
    for arm in E3_ARMS:
        adapt = stage_map[arm.adapt_stage_id]
        final = stage_map[arm.final_stage_id]
        if adapt.kind != "adapt" or final.kind != "ext":
            raise ValueError(f"E3 arm {arm.percent}% must be adapt -> ext")
        if adapt.experiment != canonical_adapt.experiment:
            raise ValueError(f"E3 arm {arm.percent}% adapt config drifted from canonical S2 adapt")
        if final.experiment != canonical_ext.experiment:
            raise ValueError(f"E3 arm {arm.percent}% extension config drifted from canonical S2 extension")
        if adapt.required_parent_checkpoint_ids != ["S0_PRETRAIN_FA_125M"]:
            raise ValueError(f"E3 arm {arm.percent}% adapt must resume from the shared FA seed")
        if final.required_parent_checkpoint_ids != [arm.adapt_stage_id]:
            raise ValueError(f"E3 arm {arm.percent}% extension parent mismatch")
        expected_resume = f"training.resume_exp_name={arm.adapt_run_id}"
        if expected_resume not in final.extra_overrides:
            raise ValueError(f"E3 arm {arm.percent}% final stage missing {expected_resume}")

    cont = stage_map["S2_MINUS_CONT_125M"]
    if cont.kind != "ext":
        raise ValueError("S2_MINUS_CONT_125M must be an extension/continuation stage")
    if cont.required_parent_checkpoint_ids != ["S2_MINUS_125M"]:
        raise ValueError("S2_MINUS_CONT_125M must resume from S2_MINUS_125M")
    if "training.load_part=all" not in cont.extra_overrides:
        raise ValueError("S2_MINUS_CONT_125M must restore optimizer state via load_part=all")
