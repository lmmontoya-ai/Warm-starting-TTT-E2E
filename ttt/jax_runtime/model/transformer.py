"""Minimal transformer-like sequence model for the native JAX runtime.

The goal here is not exact architectural parity with the reference code. The
objective is a deterministic, gradient-based JAX backend that supports the
same config contract and experiment lineage semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .attention import causal_context


@dataclass(frozen=True)
class RuntimeModelSpec:
    original_vocab_size: int
    original_hidden_size: int
    original_num_layers: int
    original_num_heads: int
    original_intermediate_size: int

    effective_vocab_size: int
    effective_hidden_size: int
    effective_num_layers: int
    effective_num_heads: int
    effective_intermediate_size: int

    seq_modeling_block: str
    attention_pattern: str
    use_sliding_window: bool
    sliding_window_size: int
    use_prime: bool
    suffix_len: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_vocab_size": self.original_vocab_size,
            "original_hidden_size": self.original_hidden_size,
            "original_num_layers": self.original_num_layers,
            "original_num_heads": self.original_num_heads,
            "original_intermediate_size": self.original_intermediate_size,
            "effective_vocab_size": self.effective_vocab_size,
            "effective_hidden_size": self.effective_hidden_size,
            "effective_num_layers": self.effective_num_layers,
            "effective_num_heads": self.effective_num_heads,
            "effective_intermediate_size": self.effective_intermediate_size,
            "seq_modeling_block": self.seq_modeling_block,
            "attention_pattern": self.attention_pattern,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window_size": self.sliding_window_size,
            "use_prime": self.use_prime,
            "suffix_len": self.suffix_len,
        }


ModelParams = dict[str, Any]


def _cap_dimension(value: int, cap: int) -> int:
    if cap <= 0:
        return max(1, int(value))
    return max(1, min(int(value), int(cap)))


def derive_model_spec(cfg) -> RuntimeModelSpec:
    model = cfg.model
    training = cfg.training

    effective_vocab = _cap_dimension(model.vocab_size, int(training.jax_vocab_size_cap))
    effective_hidden = _cap_dimension(model.hidden_size, int(training.jax_hidden_size_cap))
    effective_layers = _cap_dimension(model.num_hidden_layers, int(training.jax_num_layers_cap))
    effective_heads = _cap_dimension(model.num_attention_heads, int(training.jax_num_heads_cap))
    effective_intermediate = _cap_dimension(
        model.intermediate_size, int(training.jax_intermediate_size_cap)
    )

    # Head count should divide hidden size for multi-head style projections.
    while effective_heads > 1 and (effective_hidden % effective_heads) != 0:
        effective_heads -= 1

    seq_block = str(model.seq_modeling_block)
    attention_pattern = str(getattr(model, "attention_pattern", "full"))
    use_swa = (seq_block.lower() == "swa") or ("swa" in attention_pattern.lower())

    return RuntimeModelSpec(
        original_vocab_size=int(model.vocab_size),
        original_hidden_size=int(model.hidden_size),
        original_num_layers=int(model.num_hidden_layers),
        original_num_heads=int(model.num_attention_heads),
        original_intermediate_size=int(model.intermediate_size),
        effective_vocab_size=effective_vocab,
        effective_hidden_size=effective_hidden,
        effective_num_layers=effective_layers,
        effective_num_heads=effective_heads,
        effective_intermediate_size=effective_intermediate,
        seq_modeling_block=seq_block,
        attention_pattern=attention_pattern,
        use_sliding_window=use_swa,
        sliding_window_size=int(getattr(model, "sliding_window_size", 0) or 0),
        use_prime=bool(getattr(model, "prime", False)),
        suffix_len=int(getattr(model, "suffix_len", 0) or 0),
    )


def _init_layer(key: jax.Array, hidden: int, intermediate: int) -> dict[str, jnp.ndarray]:
    k1, k2 = jax.random.split(key)
    scale = 1.0 / jnp.sqrt(float(hidden))
    return {
        "w1": jax.random.normal(k1, (hidden, intermediate)) * scale,
        "b1": jnp.zeros((intermediate,), dtype=jnp.float32),
        "w2": jax.random.normal(k2, (intermediate, hidden)) * scale,
        "b2": jnp.zeros((hidden,), dtype=jnp.float32),
    }


def init_params(key: jax.Array, spec: RuntimeModelSpec) -> ModelParams:
    keys = jax.random.split(key, 3 + 2 * spec.effective_num_layers)
    k_embed, k_out, k_layers = keys[0], keys[1], keys[2:]

    scale = 1.0 / jnp.sqrt(float(spec.effective_hidden_size))
    embed = jax.random.normal(
        k_embed, (spec.effective_vocab_size, spec.effective_hidden_size)
    ) * scale
    out_w = jax.random.normal(
        k_out, (spec.effective_hidden_size, spec.effective_vocab_size)
    ) * scale
    out_b = jnp.zeros((spec.effective_vocab_size,), dtype=jnp.float32)

    base_layers: list[dict[str, jnp.ndarray]] = []
    prime_layers: list[dict[str, jnp.ndarray]] = []
    for i in range(spec.effective_num_layers):
        base_layers.append(
            _init_layer(
                key=k_layers[2 * i],
                hidden=spec.effective_hidden_size,
                intermediate=spec.effective_intermediate_size,
            )
        )
        if spec.use_prime:
            prime_layers.append(
                _init_layer(
                    key=k_layers[2 * i + 1],
                    hidden=spec.effective_hidden_size,
                    intermediate=spec.effective_intermediate_size,
                )
            )

    return {
        "embed": embed,
        "out_w": out_w,
        "out_b": out_b,
        "layers": tuple(base_layers),
        "prime_layers": tuple(prime_layers),
    }


def token_id_projection(token_ids: jnp.ndarray, spec: RuntimeModelSpec) -> jnp.ndarray:
    token_ids = token_ids.astype(jnp.int32)
    return jnp.mod(token_ids, jnp.int32(spec.effective_vocab_size))


def forward(
    params: ModelParams,
    input_ids: jnp.ndarray,
    spec: RuntimeModelSpec,
    *,
    use_prime: bool,
) -> jnp.ndarray:
    token_ids = token_id_projection(input_ids, spec)
    hidden = params["embed"][token_ids]

    for i, layer in enumerate(params["layers"]):
        window_size = spec.sliding_window_size if spec.use_sliding_window else None
        context = causal_context(hidden, sliding_window_size=window_size)

        h = jnp.tanh(jnp.einsum("bth,hi->bti", context, layer["w1"]) + layer["b1"])
        hidden = hidden + jnp.einsum("bti,ih->bth", h, layer["w2"]) + layer["b2"]

        if use_prime and spec.use_prime and i < len(params["prime_layers"]):
            prime = params["prime_layers"][i]
            p = jnp.tanh(jnp.einsum("bth,hi->bti", context, prime["w1"]) + prime["b1"])
            hidden = hidden + jnp.einsum("bti,ih->bth", p, prime["w2"]) + prime["b2"]

    logits = jnp.einsum("bth,hv->btv", hidden, params["out_w"]) + params["out_b"]
    return logits


def param_count(params: ModelParams) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    total = 0
    for leaf in leaves:
        total += int(leaf.size)
    return total
