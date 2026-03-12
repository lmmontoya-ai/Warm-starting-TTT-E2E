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
    valid_text_length = jnp.maximum(jnp.sum(mask, axis=-1), 1e-10)
    token_wise_loss = nll * mask
    loss = jnp.mean(jnp.sum(token_wise_loss, axis=-1) / valid_text_length)
    loss_pure_ce = loss
    return loss, loss_pure_ce


def masked_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss_masks: jnp.ndarray,
) -> jnp.ndarray:
    mask = loss_masks.astype(logits.dtype)
    preds = jnp.argmax(logits, axis=-1)
    valid_text_length = jnp.maximum(jnp.sum(mask, axis=-1), 1e-10)
    per_seq_acc = jnp.sum((preds == targets).astype(logits.dtype) * mask, axis=-1) / valid_text_length
    return jnp.mean(per_seq_acc)
