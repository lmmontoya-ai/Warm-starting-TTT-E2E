"""Learning-rate schedule sanity checks for the JAX trainer.

The revision-v2 runs launched 480-step extensions with a schedule written for
120 steps (warmup 12, cosine decay over 120), so the learning rate reached its
floor at step 120 and stayed there for three quarters of every run. The bridge
had the opposite problem: warmup equal to the whole run, so it ended at peak LR
and never annealed. Both came from overriding ``training.total_steps`` without
touching the schedule. These checks compare the schedule with the run length
before the first step.

Pure Python, so it can be unit tested without JAX.
"""

from __future__ import annotations

import math


class ScheduleValidationError(RuntimeError):
    """Raised when the outer learning-rate schedule does not fit the run length."""


SCHEDULE_CHECK_MODES = ("error", "warn", "off")


def warmup_cosine_lr(step: int, *, init_lr: float, peak_lr: float, end_lr: float, warmup_steps: int, decay_steps: int) -> float:
    """Value of ``optax.warmup_cosine_decay_schedule`` at ``step`` (decay_steps includes warmup)."""

    if warmup_steps > 0 and step < warmup_steps:
        return init_lr + (peak_lr - init_lr) * step / warmup_steps
    cosine_steps = max(decay_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0), cosine_steps) / cosine_steps
    return end_lr + 0.5 * (peak_lr - end_lr) * (1.0 + math.cos(math.pi * progress))


def lr_schedule_problems(
    *,
    total_steps: int,
    peak_lr: float,
    init_lr: float,
    end_lr: float,
    warmup_steps: int,
    decay_steps: int,
    ilr_warmup_steps: int = 0,
    ilr_init: float | None = None,
    ilr_target: float | None = None,
) -> list[str]:
    """Return human-readable problems with a warmup+cosine schedule for a run of ``total_steps``."""

    problems: list[str] = []
    if peak_lr == 0.0 or total_steps <= 0:
        return problems
    if warmup_steps >= total_steps:
        final = warmup_cosine_lr(total_steps - 1, init_lr=init_lr, peak_lr=peak_lr, end_lr=end_lr, warmup_steps=warmup_steps, decay_steps=decay_steps)
        problems.append(
            f"lr_warmup_steps={warmup_steps} >= total_steps={total_steps}: the run ends inside the warmup at LR {final:.3g} "
            f"and never anneals."
        )
    elif decay_steps < total_steps:
        tail = total_steps - decay_steps
        problems.append(
            f"lr_decay_steps={decay_steps} < total_steps={total_steps}: the LR reaches its floor {end_lr:.3g} at step {decay_steps} "
            f"and stays there for the last {tail} steps ({100.0 * tail / total_steps:.0f}% of the run)."
        )
    elif decay_steps > total_steps:
        final = warmup_cosine_lr(total_steps - 1, init_lr=init_lr, peak_lr=peak_lr, end_lr=end_lr, warmup_steps=warmup_steps, decay_steps=decay_steps)
        problems.append(
            f"lr_decay_steps={decay_steps} > total_steps={total_steps}: the run stops before the cosine finishes, "
            f"so the last checkpoint is taken at LR {final:.3g} (peak {peak_lr:.3g}, floor {end_lr:.3g})."
        )
    if (
        ilr_warmup_steps >= total_steps
        and ilr_init is not None
        and ilr_target is not None
        and ilr_init != ilr_target
    ):
        problems.append(
            f"ilr_warmup_steps={ilr_warmup_steps} >= total_steps={total_steps}: the inner learning rate is still warming up "
            f"({ilr_init:.3g} -> {ilr_target:.3g}) when the run ends."
        )
    return problems


def enforce_lr_schedule(problems: list[str], *, mode: str, label: str, logger=None) -> None:
    """Apply the configured policy (``error`` / ``warn`` / ``off``) to schedule problems."""

    mode = str(mode).strip().lower()
    if mode not in SCHEDULE_CHECK_MODES:
        raise ValueError(f"training.lr_schedule_check must be one of {SCHEDULE_CHECK_MODES}, got {mode!r}")
    if not problems or mode == "off":
        return
    message = f"{label} learning-rate schedule does not fit the run:\n- " + "\n- ".join(problems)
    if mode == "error":
        raise ScheduleValidationError(
            message
            + "\nScale training.optimizer_outer.lr_warmup_steps / lr_decay_steps (and training.ilr_warmup_steps) to the run "
            "length, or set training.lr_schedule_check=warn for a deliberate truncated schedule (e.g. a continuation)."
        )
    if logger is not None:
        logger.warning(message)
