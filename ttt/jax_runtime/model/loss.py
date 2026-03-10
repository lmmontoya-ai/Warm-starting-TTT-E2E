"""Loss utilities for native JAX runtime."""

from __future__ import annotations

import jax.numpy as jnp
from jax import nn as jnn


def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Token-level cross-entropy over [batch, seq, vocab] logits."""
    log_probs = jnn.log_softmax(logits, axis=-1)
    tgt = targets[..., None]
    selected = jnp.take_along_axis(log_probs, tgt, axis=-1).squeeze(axis=-1)
    return -jnp.mean(selected)


def token_accuracy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == targets).astype(jnp.float32))
