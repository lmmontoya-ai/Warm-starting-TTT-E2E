"""Attention blocks used by the parity runtime."""

from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import jax.random as jrandom
from equinox import nn
from jax.sharding import PartitionSpec as P

from ttt.config import ModelConfig
from ttt.utils.jax_utils import get_float_dtype_by_name, maybe_double_remat, promote_dtype, tree_rearrange

from .data import Batch


def _attention_implementation(*, use_flash: bool):
    override = os.environ.get("TTT_ATTENTION_IMPLEMENTATION", "").strip().lower()
    if override in {"xla", "cudnn"}:
        return override
    return "cudnn" if use_flash else None


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2).astype(dtype) / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).astype(dtype)
    return jnp.asarray(jnp.cos(freqs) + 1j * jnp.sin(freqs), dtype=jnp.complex64)


def apply_rotary_emb(x: jnp.ndarray, freqs_cis: jnp.ndarray) -> jnp.ndarray:
    input_dtype = x.dtype
    freqs_cis = freqs_cis.reshape(*freqs_cis.shape[:-1], 1, freqs_cis.shape[-1])
    reshape_x = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    complex_x = jax.lax.complex(reshape_x[..., 0], reshape_x[..., 1])
    out = complex_x * freqs_cis
    out = jnp.stack((jnp.real(out), jnp.imag(out)), axis=-1).reshape(*out.shape[:-1], -1)
    return out.astype(input_dtype)


class NormalLinear(eqx.Module):
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    weight: jax.Array

    def __init__(self, config: ModelConfig, in_features: int, out_features: int, *, name: str, std: float, key):
        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.weight = jrandom.normal(key, (in_features, out_features), dtype=self.param_dtype) * std

    @jax.named_scope("ttt.transformer.NormalLinear")
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"pre_promote_{self.name}")
        x, weight = promote_dtype(x, self.weight, dtype=self.compute_dtype)
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"pre_{self.name}")
        x = x @ weight
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"post_{self.name}")
        return x


class AttentionBase(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    sliding_window_size: int = eqx.field(static=True)

    wq: NormalLinear
    wk: NormalLinear
    wv: NormalLinear
    wo: NormalLinear
    q_norm: nn.RMSNorm
    k_norm: nn.RMSNorm
    resid_dropout: nn.Dropout = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.sliding_window_size = int(config.sliding_window_size)
        keys = jrandom.split(key, 4)
        self.wq, self.wk, self.wv, self.wo = (
            NormalLinear(config, config.hidden_size, config.hidden_size, name=name, std=config.initializer_range, key=k)
            for k, name in zip(keys, ("wq", "wk", "wv", "wo"))
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

    @property
    def freqs_cis(self) -> jnp.ndarray:
        with jax.ensure_compile_time_eval():
            return precompute_freqs_cis(self.head_dim, 2 * self.config.seq_len, theta=self.config.rope_theta)

    def _split_heads(self, x):
        return tree_rearrange(x, "... (head dim) -> ... head dim", head=self.num_heads, dim=self.head_dim)

    def _merge_heads(self, x):
        return tree_rearrange(x, "... head dim -> ... (head dim)", head=self.num_heads, dim=self.head_dim)

    def apply_rope(self, xis, position_ids: jnp.ndarray):
        freqs = jnp.take(self.freqs_cis, position_ids, axis=0)
        rope_fn = maybe_double_remat(
            apply_rotary_emb,
            prevent_cse=True,
            policy_remat=self.config.remat_rms,
            policy_remat_bwd=self.config.remat_rms_bwd,
        )
        return jax.tree.map(lambda x: rope_fn(x, freqs), xis)

    def project_qkv(self, hidden_states):
        return self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

    def get_attention_input(self, hidden_states: jnp.ndarray, position_ids: jnp.ndarray):
        xq, xk, xv = self.project_qkv(hidden_states)
        xq, xk, xv = self._split_heads((xq, xk, xv))
        if self.config.qk_norm:
            rms_fn = maybe_double_remat(
                nn.RMSNorm.__call__,
                prevent_cse=True,
                policy_remat=self.config.remat_rms,
                policy_remat_bwd=self.config.remat_rms_bwd,
            )
            xq = jax.vmap(jax.vmap(lambda x: rms_fn(self.q_norm, x)))(xq)
            xk = jax.vmap(jax.vmap(lambda x: rms_fn(self.k_norm, x)))(xk)
        xq, xk = self.apply_rope((xq, xk), position_ids)
        return xq, xk, xv

    def get_attention_output(self, attn_output):
        return self.resid_dropout(self.wo(attn_output))

    def core_attention_op(self, xq, xk, xv, attention_mask=None):
        use_flash = bool(self.config.force_flash)
        if use_flash:
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))
        context = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            mask=attention_mask,
            implementation=_attention_implementation(use_flash=use_flash),
        )
        if use_flash:
            context = jax.lax.with_sharding_constraint(context, P(None, "state", None))
        return self._merge_heads(context)

    def __call__(self, *_args, **_kwargs):
        raise NotImplementedError


class Attention(AttentionBase):
    def __call__(self, hidden_states: jnp.ndarray, seq: Batch, state: nn.State, *, is_prefix: bool = False):
        position_ids = seq.position_ids if seq.position_ids is not None else jnp.arange(hidden_states.shape[0], dtype=jnp.int32)
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)
        use_flash = (self.config.force_flash or is_prefix) and jax.default_backend() == "gpu"
        if use_flash:
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))
        attn_output = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=True,
            implementation=_attention_implementation(use_flash=use_flash),
        )
        if use_flash:
            attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))
        attn_output = self._merge_heads(attn_output)
        return self.get_attention_output(attn_output), state


class SWAFull(Attention):
    def __call__(self, hidden_states: jnp.ndarray, seq: Batch, state: nn.State, *, is_prefix: bool = False):
        position_ids = seq.position_ids if seq.position_ids is not None else jnp.arange(hidden_states.shape[0], dtype=jnp.int32)
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)
        use_flash = (self.config.force_flash or is_prefix) and jax.default_backend() == "gpu"
        if use_flash:
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))
        attn_output = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            local_window_size=(self.sliding_window_size - 1, 0),
            is_causal=True,
            implementation=_attention_implementation(use_flash=use_flash),
        )
        if use_flash:
            attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))
        attn_output = self._merge_heads(attn_output)
        return self.get_attention_output(attn_output), state


class SWA(AttentionBase):
    kv_cache_index: nn.StateIndex
    chunk_index: nn.StateIndex
    mini_batch_size: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key):
        super().__init__(config, key=key)
        self.mini_batch_size = int(config.mini_batch_size)
        self.window_size = int(config.sliding_window_size)
        self.kv_cache_index = nn.StateIndex(self.init_kv_cache())
        self.chunk_index = nn.StateIndex(jnp.array(0, dtype=jnp.int32))

    def init_kv_cache(self):
        return (
            jnp.zeros((self.window_size, self.config.hidden_size), dtype=self.compute_dtype),
            jnp.zeros((self.window_size, self.config.hidden_size), dtype=self.compute_dtype),
        )

    def sw_causal_mask(self, chunk_id):
        nk = self.window_size + self.mini_batch_size
        nq = self.mini_batch_size
        starting_query_idx = chunk_id * nq
        ending_query_idx = starting_query_idx + nq
        ending_key_idx = ending_query_idx
        qi = (jnp.arange(0, nq, dtype=jnp.int32) + starting_query_idx)[:, None]
        ki = (jnp.arange(-nk, 0, dtype=jnp.int32) + ending_key_idx)[None, :]
        return (qi >= ki) & (qi < ki + self.window_size) & (ki >= 0)

    def full_sw_attention(self, hidden_states, seq: Batch, state: nn.State):
        position_ids = seq.position_ids if seq.position_ids is not None else jnp.arange(hidden_states.shape[0], dtype=jnp.int32)
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)
        xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
        xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
        xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))
        attn_output = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=True,
            local_window_size=(self.window_size - 1, 0),
            implementation=_attention_implementation(use_flash=jax.default_backend() == "gpu"),
        )
        attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))
        attn_output = self._merge_heads(attn_output)
        return self.get_attention_output(attn_output), state

    def __call__(self, hidden_states: jnp.ndarray, seq: Batch, state: nn.State, *, is_prefix: bool = False):
        if is_prefix:
            return self.full_sw_attention(hidden_states, seq, state)

        xq, xk, xv = self.project_qkv(hidden_states)
        xq, xk, xv = self._split_heads((xq, xk, xv))
        if self.config.qk_norm:
            rms_fn = maybe_double_remat(
                nn.RMSNorm.__call__,
                prevent_cse=True,
                policy_remat=self.config.remat_rms,
                policy_remat_bwd=self.config.remat_rms_bwd,
            )
            xq = jax.vmap(jax.vmap(lambda x: rms_fn(self.q_norm, x)))(xq)
            xk = jax.vmap(jax.vmap(lambda x: rms_fn(self.k_norm, x)))(xk)

        prev_kv_cache = state.get(self.kv_cache_index)
        prev_k, prev_v = prev_kv_cache
        prev_k, prev_v = self._split_heads((prev_k, prev_v))

        xk = jnp.concatenate([prev_k, xk], axis=0)
        xv = jnp.concatenate([prev_v, xv], axis=0)
        new_kv_cache = self._merge_heads((xk[-self.window_size :], xv[-self.window_size :]))

        xq = self.apply_rope(
            xq,
            position_ids=jnp.arange(self.window_size + self.mini_batch_size, dtype=jnp.int32)[-self.mini_batch_size :],
        )
        xk = self.apply_rope(
            xk,
            position_ids=jnp.arange(self.window_size + self.mini_batch_size, dtype=jnp.int32),
        )

        chunk_id = state.get(self.chunk_index)
        causal_mask = self.sw_causal_mask(chunk_id)
        attn_output = self.core_attention_op(
            xq,
            xk,
            xv,
            attention_mask=causal_mask,
        )
        attn_output = self.get_attention_output(attn_output)
        state = state.set(self.kv_cache_index, new_kv_cache)
        state = state.set(self.chunk_index, chunk_id + 1)
        return attn_output, state
