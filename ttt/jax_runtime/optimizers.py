"""Optimizer builders for native JAX runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

import optax


@dataclass(frozen=True)
class LrScheduleSpec:
    init_lr: float
    peak_lr: float
    end_lr: float
    warmup_steps: int
    decay_steps: int


def make_lr_schedule(spec: LrScheduleSpec):
    warmup_steps = max(int(spec.warmup_steps), 0)
    decay_steps = max(int(spec.decay_steps), 1)

    def schedule(step: int):
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            frac = float(step + 1) / float(warmup_steps)
            return spec.init_lr + frac * (spec.peak_lr - spec.init_lr)

        t = min(max((step - warmup_steps) / float(max(decay_steps - warmup_steps, 1)), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return spec.end_lr + cosine * (spec.peak_lr - spec.end_lr)

    return schedule


def build_outer_optimizer(opt_cfg):
    schedule = make_lr_schedule(
        LrScheduleSpec(
            init_lr=float(getattr(opt_cfg, "init_lr", 0.0)),
            peak_lr=float(getattr(opt_cfg, "lr", 1e-3)),
            end_lr=float(getattr(opt_cfg, "end_lr", 1e-5)),
            warmup_steps=int(getattr(opt_cfg, "lr_warmup_steps", 0)),
            decay_steps=int(getattr(opt_cfg, "lr_decay_steps", 1)),
        )
    )

    clip = float(getattr(opt_cfg, "clip_gradient", 0.0) or 0.0)
    chain = []
    if clip > 0:
        chain.append(optax.clip_by_global_norm(clip))

    opt_type = str(getattr(opt_cfg, "optimizer_type", "adamw"))
    if opt_type == "sgd":
        chain.append(optax.sgd(learning_rate=schedule))
    else:
        chain.append(
            optax.adamw(
                learning_rate=schedule,
                b1=float(getattr(opt_cfg, "b1", 0.9)),
                b2=float(getattr(opt_cfg, "b2", 0.95)),
                weight_decay=float(getattr(opt_cfg, "weight_decay", 0.0)),
            )
        )

    return optax.chain(*chain), schedule
