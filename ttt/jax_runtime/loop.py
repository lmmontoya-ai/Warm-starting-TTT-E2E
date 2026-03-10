"""Training/eval loop primitives for native JAX runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ttt.dataloader import Batch

from .model.loss import cross_entropy_loss, token_accuracy
from .model.transformer import RuntimeModelSpec, forward


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    accuracy: float
    gradient_norm: float
    inner_loss: float = 0.0


def batch_to_arrays(batch: Batch, *, max_seq_tokens: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert phase-1 batch object to dense JAX arrays."""
    if max_seq_tokens <= 1:
        raise ValueError("max_seq_tokens must be > 1")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for sample in batch.samples:
        in_tokens = np.asarray(sample.input_ids, dtype=np.int32)
        tgt_tokens = np.asarray(sample.target_tokens, dtype=np.int32)

        n = min(len(in_tokens), len(tgt_tokens), max_seq_tokens)
        if n <= 0:
            continue
        inputs.append(in_tokens[:n])
        targets.append(tgt_tokens[:n])

    if not inputs:
        raise ValueError("Batch has no usable tokens after truncation")

    # Ragged safety: truncate to common min length.
    seq = min(arr.shape[0] for arr in inputs)
    x = np.stack([arr[:seq] for arr in inputs], axis=0)
    y = np.stack([arr[:seq] for arr in targets], axis=0)
    return jnp.asarray(x), jnp.asarray(y)


def _replace_prime(params: dict[str, Any], prime_layers) -> dict[str, Any]:
    return {
        "embed": params["embed"],
        "out_w": params["out_w"],
        "out_b": params["out_b"],
        "layers": params["layers"],
        "prime_layers": prime_layers,
    }


def _grad_norm(grads: Any) -> float:
    leaves = jax.tree_util.tree_leaves(grads)
    if not leaves:
        return 0.0
    sq = 0.0
    for leaf in leaves:
        sq += float(jnp.sum(jnp.square(leaf)))
    return float(np.sqrt(max(sq, 0.0)))


def pretrain_step(
    *,
    params,
    opt_state,
    optimizer,
    spec: RuntimeModelSpec,
    input_ids: jnp.ndarray,
    targets: jnp.ndarray,
    use_prime: bool,
):
    def loss_fn(p):
        logits = forward(p, input_ids, spec, use_prime=use_prime)
        loss = cross_entropy_loss(logits, targets)
        acc = token_accuracy(logits, targets)
        return loss, acc

    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, StepMetrics(
        loss=float(loss),
        accuracy=float(acc),
        gradient_norm=_grad_norm(grads),
    )


def meta_step(
    *,
    params,
    opt_state,
    optimizer,
    spec: RuntimeModelSpec,
    input_ids: jnp.ndarray,
    targets: jnp.ndarray,
    inner_steps: int,
    inner_lr: float,
):
    if (not spec.use_prime) or (len(params["prime_layers"]) == 0):
        return pretrain_step(
            params=params,
            opt_state=opt_state,
            optimizer=optimizer,
            spec=spec,
            input_ids=input_ids,
            targets=targets,
            use_prime=False,
        )

    seq = int(input_ids.shape[1])
    split = max(1, seq // 2)
    if split >= seq:
        split = seq - 1

    inner_inputs = input_ids[:, :split]
    inner_targets = targets[:, :split]
    outer_inputs = input_ids[:, split:]
    outer_targets = targets[:, split:]

    prime = params["prime_layers"]

    def inner_loss_fn(prime_params):
        local_params = _replace_prime(params, prime_params)
        logits = forward(local_params, inner_inputs, spec, use_prime=True)
        return cross_entropy_loss(logits, inner_targets)

    inner_loss_value = 0.0
    for _ in range(max(1, int(inner_steps))):
        inner_loss, inner_grads = jax.value_and_grad(inner_loss_fn)(prime)
        prime = jax.tree_util.tree_map(
            lambda p, g: p - jnp.asarray(inner_lr, dtype=p.dtype) * g,
            prime,
            inner_grads,
        )
        inner_loss_value = float(inner_loss)

    adapted_params = _replace_prime(params, prime)

    def outer_loss_fn(p):
        logits = forward(p, outer_inputs, spec, use_prime=True)
        loss = cross_entropy_loss(logits, outer_targets)
        acc = token_accuracy(logits, outer_targets)
        return loss, acc

    (outer_loss, outer_acc), outer_grads = jax.value_and_grad(outer_loss_fn, has_aux=True)(
        adapted_params
    )
    updates, new_opt_state = optimizer.update(outer_grads, opt_state, adapted_params)
    new_params = optax.apply_updates(adapted_params, updates)

    return new_params, new_opt_state, StepMetrics(
        loss=float(outer_loss),
        accuracy=float(outer_acc),
        gradient_norm=_grad_norm(outer_grads),
        inner_loss=inner_loss_value,
    )


def eval_step(
    *,
    params,
    spec: RuntimeModelSpec,
    input_ids: jnp.ndarray,
    targets: jnp.ndarray,
    use_prime: bool,
) -> StepMetrics:
    logits = forward(params, input_ids, spec, use_prime=use_prime)
    loss = cross_entropy_loss(logits, targets)
    acc = token_accuracy(logits, targets)
    return StepMetrics(loss=float(loss), accuracy=float(acc), gradient_norm=0.0)
