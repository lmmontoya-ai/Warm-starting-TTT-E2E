"""Lightweight causal context operators for the native JAX runtime.

This module intentionally implements simple causal aggregation kernels that are
stable for long sequences and small-memory environments.
"""

from __future__ import annotations

import jax.numpy as jnp


def _causal_mean_full(hidden: jnp.ndarray) -> jnp.ndarray:
    """Causal mean over the full prefix for each position.

    Args:
        hidden: [batch, seq, hidden]
    Returns:
        [batch, seq, hidden] prefix-mean context.
    """
    prefix_sum = jnp.cumsum(hidden, axis=1)
    counts = jnp.arange(1, hidden.shape[1] + 1, dtype=hidden.dtype)
    counts = counts[None, :, None]
    return prefix_sum / counts


def _causal_mean_sliding(hidden: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """Sliding-window causal mean.

    This implementation is O(T^2) in sequence length but is intentionally kept
    simple and robust for the current research runtime where max sequence is
    explicitly capped.
    """
    if window_size <= 0:
        return _causal_mean_full(hidden)

    seq_len = hidden.shape[1]
    contexts: list[jnp.ndarray] = []
    for t in range(seq_len):
        start = max(0, t - window_size + 1)
        chunk = hidden[:, start : t + 1, :]
        contexts.append(jnp.mean(chunk, axis=1))
    return jnp.stack(contexts, axis=1)


def causal_context(hidden: jnp.ndarray, sliding_window_size: int | None) -> jnp.ndarray:
    """Compute causal context with optional sliding window."""
    if sliding_window_size is None:
        return _causal_mean_full(hidden)
    return _causal_mean_sliding(hidden, int(sliding_window_size))
