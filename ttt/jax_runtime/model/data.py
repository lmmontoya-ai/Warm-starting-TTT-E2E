from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn


def tree_slice(tree, i: int):
    return jax.tree.map(lambda x: x[i], tree)


class BaseModelOutput(eqx.Module):
    state: nn.State | None = None
    last_hidden_state: jnp.ndarray | None = None
    logits: jnp.ndarray | None = None


class Batch(eqx.Module):
    input_ids: jnp.ndarray
    target_tokens: jnp.ndarray
    loss_masks: jnp.ndarray
    attention_mask: jnp.ndarray | None = None
    position_ids: jnp.ndarray | None = None
    index: int | slice | None = eqx.field(static=True, default=None)

    @property
    def shape(self):
        return self.input_ids.shape

    def slice_index(self, index: int | slice) -> "Batch":
        return replace(tree_slice(self, index), index=index)
