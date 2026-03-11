"""Optimizer builders for the parity runtime."""

from __future__ import annotations

import re

import jax.numpy as jnp
import optax

from ttt.config import AdamWOptimizerConfig, OptimizerConfig, SGDOptimizerConfig
from ttt.utils.filter_utils import get_mask_fn


def make_adamw_optimizer(config: AdamWOptimizerConfig, weight_decay_mask=None):
    if config.lr == 0.0:
        learning_rate_schedule = optax.constant_schedule(0.0)
    else:
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.init_lr,
            peak_value=config.lr,
            warmup_steps=config.lr_warmup_steps,
            decay_steps=config.lr_decay_steps,
            end_value=config.end_lr,
        )
    if not config.emb_wd:
        exclude_emb = lambda name: False if re.search("wte", name) else True
        weight_decay_mask = lambda params: get_mask_fn(exclude_emb, params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_gradient),
        optax.adamw(
            learning_rate=learning_rate_schedule,
            weight_decay=config.weight_decay,
            b1=config.b1,
            b2=config.b2,
            mask=weight_decay_mask,
            mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
        ),
    )
    return optimizer, {"learning_rate_schedule": learning_rate_schedule}


def make_sgd_optimizer(config: SGDOptimizerConfig, ilr_multiplier: jnp.ndarray | None = None):
    multiplier = 1.0 if ilr_multiplier is None else ilr_multiplier
    learning_rate_schedule = optax.constant_schedule(config.lr * multiplier)
    if config.clip_gradient > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.sgd(learning_rate=learning_rate_schedule, momentum=None),
        )
    else:
        optimizer = optax.sgd(learning_rate=learning_rate_schedule, momentum=None)
    return optimizer, {"learning_rate_schedule": learning_rate_schedule}


def make_optimizer(optimizer_config: OptimizerConfig, ilr_multiplier: jnp.ndarray | None = None):
    if optimizer_config.optimizer_type == "adamw":
        return make_adamw_optimizer(optimizer_config)
    if optimizer_config.optimizer_type == "sgd":
        return make_sgd_optimizer(optimizer_config, ilr_multiplier)
    raise ValueError(f"Unknown optimizer type: {optimizer_config.optimizer_type}")
