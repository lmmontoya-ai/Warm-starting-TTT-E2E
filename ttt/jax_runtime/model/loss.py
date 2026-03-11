"""Loss helpers for the parity runtime."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def token_log_probs(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return selected


def cross_entropy_loss_and_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss_masks: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    nll = -token_log_probs(logits, targets)
    mask = loss_masks.astype(logits.dtype)
    denom = jnp.maximum(mask.sum(), 1.0)
    loss = jnp.sum(nll * mask) / denom
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.sum((preds == targets).astype(logits.dtype) * mask) / denom
    return loss, acc
