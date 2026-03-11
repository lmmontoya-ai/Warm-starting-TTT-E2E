"""Attention blocks used by the parity runtime."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox import nn

from ttt.config import ModelConfig
from ttt.utils.jax_utils import get_float_dtype_by_name, maybe_double_remat, promote_dtype, tree_rearrange

from .data import Batch


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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x, weight = promote_dtype(x, self.weight, dtype=self.compute_dtype)
        return x @ weight


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

    def _attention_mask(self, seq_len: int) -> jnp.ndarray:
        idx = jnp.arange(seq_len)
        causal = idx[:, None] >= idx[None, :]
        if self.config.seq_modeling_block == "SWA" or self.config.attention_pattern == "swa":
            local = (idx[:, None] - idx[None, :]) < self.sliding_window_size
            causal = causal & local
        return causal

    def apply_rope(self, xq, xk, position_ids: jnp.ndarray):
        freqs = jnp.take(self.freqs_cis, position_ids, axis=0)
        rope_fn = maybe_double_remat(
            apply_rotary_emb,
            prevent_cse=True,
            policy_remat=self.config.remat_rms,
            policy_remat_bwd=self.config.remat_rms_bwd,
        )
        return rope_fn(xq, freqs), rope_fn(xk, freqs)

    def _project(self, hidden_states: jnp.ndarray, position_ids: jnp.ndarray):
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
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
        xq, xk = self.apply_rope(xq, xk, position_ids)
        return xq, xk, xv

    def __call__(self, hidden_states: jnp.ndarray, seq: Batch, state=None, *, is_prefix: bool = False):
        del state, is_prefix
        position_ids = seq.position_ids if seq.position_ids is not None else jnp.arange(hidden_states.shape[0], dtype=jnp.int32)
        xq, xk, xv = self._project(hidden_states, position_ids)
        scale = jnp.asarray(self.head_dim**-0.5, dtype=self.compute_dtype)
        scores = jnp.einsum("thd,shd->hts", xq, xk) * scale
        mask = self._attention_mask(hidden_states.shape[0])
        scores = jnp.where(mask[None, :, :], scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("hts,shd->thd", probs, xv)
        out = self.wo(self._merge_heads(context))
        return self.resid_dropout(out), None


class Attention(AttentionBase):
    pass


class SWA(AttentionBase):
    pass


class SWAFull(AttentionBase):
    pass
