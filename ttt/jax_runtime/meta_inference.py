"""Streaming prompt adaptation and greedy decode for TTT-E2E research evals.

The training path remains untouched. This module supplies the missing
autoregressive inference semantics required by the external RULER slice:
fast weights update on complete 1,024-token prompt chunks and remain frozen
during answer generation.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import nn

from ttt.utils.filter_utils import get_filter_spec
from ttt.utils.jax_utils import clone_pytree, scan_remat_chunk, tree_rearrange

from .model.attention import SWA
from .model.data import Batch
from .model.transformer import BlockCollectionSplit, MetaModel
from .research_eval import ResearchEvalSession, to_sharded_batch


def _apply_stack_chunk(blocks, hidden_states, state, seq: Batch):
    def apply_block(x, block_and_state):
        block, block_state = block_and_state
        x, block_state = block(x, block_state, seq, is_prefix=False)
        return x, block_state

    return jax.lax.scan(apply_block, hidden_states, (blocks, state))


def _swa_decode_step(attention: SWA, hidden_states: jnp.ndarray, state):
    """Apply SWA to a short decode fragment against a populated KV cache."""
    decode_len = int(hidden_states.shape[0])
    if decode_len <= 0 or decode_len > int(attention.mini_batch_size):
        raise ValueError("Decode fragment must be between 1 and mini_batch_size tokens")
    # Decode is only valid after at least one fixed prompt chunk populated the
    # cache; otherwise the zero-initialized cache would be treated as context.
    chunk_index = state.get(attention.chunk_index)

    xq, xk, xv = attention.project_qkv(hidden_states)
    xq, xk, xv = attention._split_heads((xq, xk, xv))
    if attention.config.qk_norm:
        rms = nn.RMSNorm.__call__
        xq = jax.vmap(jax.vmap(lambda x: rms(attention.q_norm, x)))(xq)
        xk = jax.vmap(jax.vmap(lambda x: rms(attention.k_norm, x)))(xk)
    prev_k, prev_v = state.get(attention.kv_cache_index)
    prev_k, prev_v = attention._split_heads((prev_k, prev_v))
    xk_all = jnp.concatenate([prev_k, xk], axis=0)
    xv_all = jnp.concatenate([prev_v, xv], axis=0)
    new_cache = attention._merge_heads(
        (xk_all[-attention.window_size :], xv_all[-attention.window_size :])
    )
    relative_positions = jnp.arange(attention.window_size + decode_len, dtype=jnp.int32)
    xq = attention.apply_rope(xq, position_ids=relative_positions[-decode_len:])
    xk_all = attention.apply_rope(xk_all, position_ids=relative_positions)
    qi = (jnp.arange(decode_len, dtype=jnp.int32) + attention.window_size)[:, None]
    ki = jnp.arange(attention.window_size + decode_len, dtype=jnp.int32)[None, :]
    mask = (qi >= ki) & (qi < ki + attention.window_size)
    # Assert through a value dependency without introducing a host-side branch
    # under JIT. Production validation separately checks prompt chunk count.
    mask = mask & (chunk_index > 0)
    output = attention.core_attention_op(xq, xk_all, xv_all, attention_mask=mask)
    output = attention.get_attention_output(output)
    state = state.set(attention.kv_cache_index, new_cache)
    state = state.set(attention.chunk_index, chunk_index + 1)
    return output, state


def _decode_block(block, hidden_states, state, seq: Batch):
    if not isinstance(block.seq_modeling_block, SWA):
        raise TypeError("Tier-1 streaming decode currently supports SWA blocks only")
    rms = nn.RMSNorm.__call__
    x = (
        jax.vmap(lambda h: rms(block.seq_norm, h))(hidden_states)
        if block.config.pre_norm
        else hidden_states
    )
    seq_output, state = _swa_decode_step(block.seq_modeling_block, x, state)
    if block.config.post_norm:
        seq_output = jax.vmap(lambda h: rms(block.seq_post_norm, h))(seq_output)
    hidden_states = hidden_states + seq_output
    if block.feed_forward_prime is not None:
        prime_output = block.ffn_forward(
            block.feed_forward_prime.__class__.__call__,
            rms,
            block.ffn_prime_norm,
            block.feed_forward_prime,
            block.ffn_prime_post_norm,
            hidden_states,
        )
        hidden_states = hidden_states + prime_output
    ff_output = block.ffn_forward(
        block.feed_forward.__class__.__call__,
        rms,
        block.ffn_norm,
        block.feed_forward,
        block.ffn_post_norm,
        hidden_states,
    )
    return hidden_states + ff_output, state


def _decode_stack(blocks, hidden_states, state, seq: Batch):
    def apply_block(x, block_and_state):
        block, block_state = block_and_state
        x, block_state = _decode_block(block, x, block_state, seq)
        return x, block_state

    return jax.lax.scan(apply_block, hidden_states, (blocks, state))


def _combine_current_inner(model_inner: MetaModel, base_outer: MetaModel) -> MetaModel:
    spec = get_filter_spec(
        model_inner, model_inner.config.training.spec_inner, "inner parameters"
    )
    inner_params, _ = eqx.partition(model_inner, spec)
    _, outer_params = eqx.partition(base_outer, spec)
    return eqx.combine(inner_params, outer_params)


def _prefill_prompt(meta_model: MetaModel, seq: Batch, state):
    cfg = meta_model.config
    if str(cfg.training.train_mode) != "meta" or not bool(cfg.model.prime):
        raise ValueError("RULER streaming inference requires a meta/prime checkpoint")
    chunk_length = int(cfg.model.mini_batch_size)
    if int(cfg.training.seq_length) % chunk_length != 0:
        raise ValueError("Prompt length must be divisible by the TTT chunk length")
    block_collection = meta_model.language_model.model.h.blocks
    prime_storage = meta_model.language_model.model.h.prime_storage
    model = eqx.tree_at(
        lambda m: m.language_model.model.h,
        meta_model,
        BlockCollectionSplit(
            cfg.model, block_collection, prime_storage, key=jax.random.PRNGKey(0)
        ),
    )
    split_state = state.substate(block_collection)
    prefix_state, suffix_state = BlockCollectionSplit.split_state(
        split_state, int(cfg.model.suffix_len)
    )
    state_all = clone_pytree(state)
    model = jax.tree.map(lambda p: p.astype(meta_model.state_dtype), model)
    base_outer = model
    opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())
    seq_chunks = tree_rearrange(
        seq, "(chunk token) ... -> chunk token ...", token=chunk_length
    )

    def process_chunk(carry, seq_chunk):
        model_inner, inner_opt_state, prefix_substate, suffix_substate = carry
        model_inner = _combine_current_inner(model_inner, base_outer)
        embeddings = model_inner.language_model.wte_call(seq_chunk.input_ids)
        prefix_output, prefix_substate = _apply_stack_chunk(
            model_inner.language_model.model.h.prefix_blocks,
            embeddings,
            prefix_substate,
            seq_chunk,
        )
        step = MetaModel.inner_loop_step(
            model_inner,
            inner_opt_state,
            (state_all, suffix_substate),
            seq_chunk,
            prefix_output,
        )
        return (
            step.new_model,
            step.new_optimizer_state,
            prefix_substate,
            step.new_state[1],
        ), None

    carry = (model, opt_state, prefix_state, suffix_state)
    if int(seq_chunks.input_ids.shape[0]) > 1:
        carry, _ = scan_remat_chunk(
            eqx.filter_checkpoint(process_chunk, prevent_cse=False),
            carry,
            jax.tree.map(lambda x: x[:-1], seq_chunks),
            remat_n_loops=cfg.training.inner_remat_freq,
            unroll=cfg.model.unroll_inner_scan,
        )
    model_inner, inner_opt_state, prefix_state, suffix_state = carry
    model_inner = _combine_current_inner(model_inner, base_outer)
    final_seq = jax.tree.map(lambda x: x[-1], seq_chunks)
    final_embeddings = model_inner.language_model.wte_call(final_seq.input_ids)
    final_prefix, prefix_state = _apply_stack_chunk(
        model_inner.language_model.model.h.prefix_blocks,
        final_embeddings,
        prefix_state,
        final_seq,
    )
    # The first answer token is predicted from the last prompt position before
    # the final chunk update, matching the existing leakage-safe final-token
    # scorer. The update is then retained for subsequent answer tokens.
    final_outputs = model_inner.language_model.suffix_call(
        final_prefix, suffix_state, final_seq
    )
    first_logits = final_outputs.logits[-1]
    final_step = MetaModel.inner_loop_step(
        model_inner,
        inner_opt_state,
        (state_all, suffix_state),
        final_seq,
        final_prefix,
    )
    return (
        final_step.new_model,
        prefix_state,
        final_step.new_state[1],
        first_logits,
    )


def _decode_one(
    model: MetaModel,
    token: jnp.ndarray,
    prefix_state,
    suffix_state,
    position: jnp.ndarray,
):
    token = jnp.asarray(token, dtype=jnp.int32).reshape(1)
    seq = Batch(
        input_ids=token,
        target_tokens=token,
        loss_masks=jnp.zeros((1,), dtype=jnp.int32),
        attention_mask=None,
        position_ids=jnp.asarray(position, dtype=jnp.int32).reshape(1),
    )
    hidden = model.language_model.wte_call(seq.input_ids)
    hidden, prefix_state = _decode_stack(
        model.language_model.model.h.prefix_blocks, hidden, prefix_state, seq
    )
    hidden, suffix_state = _decode_stack(
        model.language_model.model.h.suffix_blocks, hidden, suffix_state, seq
    )
    rms = nn.RMSNorm.__call__
    hidden = jax.vmap(lambda h: rms(model.language_model.model.ln_f, h))(hidden)
    logits = model.language_model._disembed(hidden)[-1]
    return logits, prefix_state, suffix_state


def generate_one(
    meta_model: MetaModel,
    seq: Batch,
    state,
    *,
    max_new_tokens: int,
    eos_token_id: int,
) -> jnp.ndarray:
    model, prefix_state, suffix_state, first_logits = _prefill_prompt(
        meta_model, seq, state
    )
    first_token = jnp.argmax(first_logits).astype(jnp.int32)
    context_length = int(meta_model.config.training.seq_length)

    def decode_step(carry, step_index):
        previous, prefix_substate, suffix_substate, finished = carry
        logits, prefix_substate, suffix_substate = _decode_one(
            model,
            previous,
            prefix_substate,
            suffix_substate,
            jnp.asarray(context_length + step_index, dtype=jnp.int32),
        )
        next_token = jnp.argmax(logits).astype(jnp.int32)
        next_token = jnp.where(
            finished, jnp.asarray(eos_token_id, dtype=jnp.int32), next_token
        )
        finished = jnp.logical_or(finished, next_token == eos_token_id)
        return (next_token, prefix_substate, suffix_substate, finished), next_token

    if max_new_tokens <= 1:
        return first_token.reshape(1)
    initial_finished = first_token == eos_token_id
    _carry, rest = jax.lax.scan(
        decode_step,
        (first_token, prefix_state, suffix_state, initial_finished),
        jnp.arange(max_new_tokens - 1, dtype=jnp.int32),
    )
    return jnp.concatenate([first_token.reshape(1), rest], axis=0)


def make_generate_sharded(*, max_new_tokens: int, eos_token_id: int):
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, None), out_axes=0, axis_name="data_parallel")
    def generate_sharded(meta_model: MetaModel, batch: Batch, state):
        return jax.vmap(
            lambda seq: generate_one(
                meta_model,
                seq,
                state,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        )(batch)

    return generate_sharded


def generate_batch(
    session: ResearchEvalSession,
    batch: Batch,
    *,
    global_batch_size: int,
    max_new_tokens: int,
) -> np.ndarray:
    sharded = to_sharded_batch(session, batch, global_batch_size=global_batch_size)
    generate = make_generate_sharded(
        max_new_tokens=int(max_new_tokens),
        eos_token_id=int(session.cfg.model.eos_token_id),
    )
    with session.mesh:
        tokens = generate(session.model, sharded, session.state)
    result = np.asarray(jax.device_get(tokens), dtype=np.int32)
    return result.reshape(global_batch_size, int(max_new_tokens))
