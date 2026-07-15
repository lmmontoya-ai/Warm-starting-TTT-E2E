"""Inference-only fast-weight diagnostics for preregistered T1b."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ttt.utils.filter_utils import filter_apply_updates, get_filter_spec
from ttt.utils.jax_utils import clone_pytree, scan_remat_chunk, tree_rearrange

from .model.data import Batch
from .model.transformer import BlockCollectionSplit, MetaModel
from .research_eval import ResearchEvalSession, to_sharded_batch


def _layer_squared_norm(tree, *, suffix_len: int) -> jnp.ndarray:
    total = jnp.zeros((suffix_len,), dtype=jnp.float32)
    found = False
    for leaf in jax.tree.leaves(tree):
        if not eqx.is_array(leaf):
            continue
        if leaf.ndim < 1 or int(leaf.shape[0]) != suffix_len:
            raise ValueError(
                f"Inner parameter leaf must start with suffix_len={suffix_len}, got {leaf.shape}"
            )
        axes = tuple(range(1, leaf.ndim))
        total = total + jnp.sum(jnp.square(leaf.astype(jnp.float32)), axis=axes)
        found = True
    if not found:
        raise ValueError("No inner fast-weight leaves were found")
    return total


def _layer_l2(tree, *, suffix_len: int) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(_layer_squared_norm(tree, suffix_len=suffix_len), 0.0))


def _layer_finite(tree, *, suffix_len: int) -> jnp.ndarray:
    result = jnp.ones((suffix_len,), dtype=jnp.bool_)
    found = False
    for leaf in jax.tree.leaves(tree):
        if not eqx.is_array(leaf):
            continue
        axes = tuple(range(1, leaf.ndim))
        result = jnp.logical_and(result, jnp.all(jnp.isfinite(leaf), axis=axes))
        found = True
    if not found:
        raise ValueError("No inner fast-weight leaves were found")
    return result


def probe_sequence(meta_model: MetaModel, seq: Batch, state):
    cfg = meta_model.config
    if str(cfg.training.train_mode) != "meta" or not bool(cfg.model.prime):
        raise ValueError("Fast-weight probe requires a meta/prime TTT-E2E checkpoint")
    suffix_len = int(cfg.model.suffix_len)
    if suffix_len <= 0:
        raise ValueError("Fast-weight probe requires suffix_len > 0")
    chunk_length = int(cfg.model.mini_batch_size)
    if int(cfg.training.seq_length) % chunk_length != 0:
        raise ValueError("Probe sequence length must be divisible by mini_batch_size")

    blocks = meta_model.language_model.model.h.blocks
    prime_storage = meta_model.language_model.model.h.prime_storage
    model = eqx.tree_at(
        lambda m: m.language_model.model.h,
        meta_model,
        BlockCollectionSplit(
            cfg.model, blocks, prime_storage, key=jax.random.PRNGKey(0)
        ),
    )
    split_state = state.substate(blocks)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(
        split_state, suffix_len
    )
    state_all = clone_pytree(state)
    model = jax.tree.map(lambda p: p.astype(meta_model.state_dtype), model)
    inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())

    embeddings = model.language_model.wte_call(seq.input_ids)
    prefix_output = eqx.filter_checkpoint(model.language_model.prefix_call)(
        model.language_model.model.h.prefix_blocks,
        embeddings,
        state_prefix,
        seq,
    ).last_hidden_state
    seq_chunks = tree_rearrange(
        seq, "(chunk token) ... -> chunk token ...", token=chunk_length
    )
    prefix_chunks = tree_rearrange(
        prefix_output, "(chunk token) ... -> chunk token ...", token=chunk_length
    )
    base_outer_model = model
    value_and_grad = eqx.filter_value_and_grad(MetaModel.lm_loss, has_aux=True)

    def process_chunk(carry, inputs):
        model_inner, opt_state, state_tuple = carry
        seq_chunk, prefix_chunk = inputs
        spec_inner = get_filter_spec(
            model_inner, cfg.training.spec_inner, "inner parameters"
        )
        inner_params, _ = eqx.partition(model_inner, spec_inner)
        _, outer_params = eqx.partition(base_outer_model, spec_inner)
        model_inner = eqx.combine(inner_params, outer_params)

        (_loss, (_pure_ce, _token_nll, next_suffix_state)), grads = value_and_grad(
            model_inner,
            seq_chunk,
            state_tuple[1],
            prefix_outputs=prefix_chunk,
        )
        inner_grads = grads.inner_parameters()
        updates, next_opt_state = model_inner.inner_optimizer(state_tuple[0]).update(
            inner_grads,
            opt_state,
            model_inner.inner_parameters(),
        )
        weight_l2 = _layer_l2(model_inner.inner_parameters(), suffix_len=suffix_len)
        grad_l2 = _layer_l2(inner_grads, suffix_len=suffix_len)
        update_l2 = _layer_l2(updates, suffix_len=suffix_len)
        relative = update_l2 / jnp.maximum(weight_l2, 1e-12)
        finite = jnp.logical_and(
            _layer_finite(inner_grads, suffix_len=suffix_len),
            _layer_finite(updates, suffix_len=suffix_len),
        )
        global_grad_l2 = jnp.sqrt(jnp.sum(jnp.square(grad_l2)))
        clip_threshold = float(cfg.training.optimizer_inner.clip_gradient)
        clip_crossed = (
            global_grad_l2 > clip_threshold
            if clip_threshold > 0.0
            else jnp.asarray(False)
        )
        next_model = filter_apply_updates(model_inner, updates)
        diagnostics = {
            "weight_l2": weight_l2,
            "raw_grad_l2": grad_l2,
            "applied_update_l2": update_l2,
            "relative_update_l2": relative,
            "clip_crossed": clip_crossed,
            "all_finite": finite,
        }
        return (
            next_model,
            next_opt_state,
            (state_tuple[0], next_suffix_state),
        ), diagnostics

    _carry, diagnostics = scan_remat_chunk(
        eqx.filter_checkpoint(process_chunk, prevent_cse=False),
        (model, inner_opt_state, (state_all, state_suffix)),
        (seq_chunks, prefix_chunks),
        remat_n_loops=cfg.training.inner_remat_freq,
        unroll=cfg.model.unroll_inner_scan,
    )
    return diagnostics


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(None, 0, None), out_axes=0, axis_name="data_parallel")
def _probe_sharded(meta_model: MetaModel, batch: Batch, state):
    return jax.vmap(lambda seq: probe_sequence(meta_model, seq, state))(batch)


def probe_batch(
    session: ResearchEvalSession,
    batch: Batch,
    *,
    global_batch_size: int,
) -> dict[str, np.ndarray]:
    sharded = to_sharded_batch(session, batch, global_batch_size=global_batch_size)
    with session.mesh:
        diagnostics = _probe_sharded(session.model, sharded, session.state)
    host = jax.device_get(diagnostics)
    result: dict[str, np.ndarray] = {}
    for key, value in host.items():
        arr = np.asarray(value)
        result[key] = arr.reshape(global_batch_size, *arr.shape[2:])
    return result
