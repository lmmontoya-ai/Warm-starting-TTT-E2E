"""Author-aligned transformer stack for the parity runtime."""

from __future__ import annotations

from enum import StrEnum, auto
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox import nn
from optax import OptState

from ttt.config import Config, ModelConfig
from ttt.utils.filter_utils import filter_apply_updates, filter_parameters, get_filter_spec
from ttt.utils.jax_utils import (
    clone_pytree,
    get_float_dtype_by_name,
    maybe_double_remat,
    promote_dtype,
    scan_or_loop,
    scan_remat_chunk,
    tree_rearrange,
)

from ..optimizers import make_optimizer
from .attention import Attention, AttentionBase, NormalLinear, SWA, SWAFull
from .data import BaseModelOutput, Batch
from .loss import cross_entropy_loss_and_accuracy, token_log_probs


class SwiGLUMLP(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    w1: NormalLinear
    w2: NormalLinear
    w3: NormalLinear
    dropout: nn.Dropout = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        k1, k2, k3 = jrandom.split(key, 3)
        self.w1 = NormalLinear(config, config.hidden_size, config.intermediate_size, name="w1", std=config.initializer_range, key=k1)
        self.w2 = NormalLinear(config, config.intermediate_size, config.hidden_size, name="w2", std=config.initializer_range, key=k2)
        self.w3 = NormalLinear(config, config.hidden_size, config.intermediate_size, name="w3", std=config.initializer_range, key=k3)
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.dropout(self.w2(jax.nn.silu(self.w1(x)) * self.w3(x)))


class PrimeStorage(eqx.Module):
    ffn_prime_norm: nn.RMSNorm
    ffn_prime_post_norm: nn.RMSNorm
    feed_forward_prime: SwiGLUMLP

    def __init__(self, config: ModelConfig, *, key):
        dtype = get_float_dtype_by_name(config.param_dtype)
        keys = jrandom.split(key, max(config.suffix_len, 1))
        suffix_keys = keys[: config.suffix_len]
        self.feed_forward_prime = jax.vmap(lambda k: SwiGLUMLP(config, key=k))(suffix_keys)
        self.ffn_prime_norm = jax.vmap(
            lambda _: nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=dtype)
        )(suffix_keys)
        self.ffn_prime_post_norm = jax.vmap(
            lambda _: nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=dtype)
        )(suffix_keys)


class Block(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    seq_modeling_block: AttentionBase
    feed_forward: SwiGLUMLP
    seq_norm: nn.RMSNorm
    ffn_norm: nn.RMSNorm
    seq_post_norm: nn.RMSNorm
    ffn_post_norm: nn.RMSNorm
    ffn_prime_norm: nn.RMSNorm | None
    ffn_prime_post_norm: nn.RMSNorm | None
    feed_forward_prime: SwiGLUMLP | None

    def __init__(self, config: ModelConfig, *, key, feed_forward_prime=None, ffn_prime_norm=None, ffn_prime_post_norm=None):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        block_cls = {"self_attention": Attention, "SWA": SWA, "SWAFull": SWAFull}[config.seq_modeling_block]
        key_attn, key_ff = jrandom.split(key, 2)
        self.seq_modeling_block = block_cls(config, key=key_attn)
        self.feed_forward = SwiGLUMLP(config, key=key_ff)
        norm = lambda: nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.seq_norm = norm()
        self.ffn_norm = norm()
        self.seq_post_norm = norm()
        self.ffn_post_norm = norm()
        self.ffn_prime_norm = ffn_prime_norm
        self.ffn_prime_post_norm = ffn_prime_post_norm
        self.feed_forward_prime = feed_forward_prime

    def seq_modeling_forward(self, hidden_states: jnp.ndarray, seq: Batch):
        rms_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd)
        block_fn = maybe_double_remat(
            partial(self.seq_modeling_block.__class__.__call__, is_prefix=False),
            prevent_cse=True,
            policy_remat=self.config.remat_attention,
            policy_remat_bwd=self.config.remat_attention_bwd,
        )
        x = jax.vmap(lambda h: rms_fn(self.seq_norm, h))(hidden_states) if self.config.pre_norm else hidden_states
        out, _ = block_fn(self.seq_modeling_block, x, seq, None)
        return jax.vmap(lambda h: rms_fn(self.seq_post_norm, h))(out) if self.config.post_norm else out

    def ffn_forward(self, hidden_states: jnp.ndarray, *, prime: bool = False):
        rms_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd)
        if prime:
            assert self.feed_forward_prime is not None
            norm = self.ffn_prime_norm
            post_norm = self.ffn_prime_post_norm
            block = self.feed_forward_prime
        else:
            norm = self.ffn_norm
            post_norm = self.ffn_post_norm
            block = self.feed_forward
        x = jax.vmap(lambda h: rms_fn(norm, h))(hidden_states) if self.config.pre_norm else hidden_states
        out = block(x)
        return jax.vmap(lambda h: rms_fn(post_norm, h))(out) if self.config.post_norm else out

    def __call__(self, hidden_states: jnp.ndarray, state, seq: Batch, *, is_prefix: bool = False):
        del state, is_prefix
        hidden_states = hidden_states + self.seq_modeling_forward(hidden_states, seq)
        if self.feed_forward_prime is not None:
            hidden_states = hidden_states + self.ffn_forward(hidden_states, prime=True)
        hidden_states = hidden_states + self.ffn_forward(hidden_states, prime=False)
        return hidden_states, None


class BlockCollectionSplit(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    prefix_blocks: Block
    suffix_blocks: Block | None

    def __init__(self, config: ModelConfig, block_collection: Block, prime_storage: PrimeStorage | None, *, key):
        self.config = config
        suffix_len = int(config.suffix_len)
        self.prefix_blocks = jax.tree.map(lambda m: m[:-suffix_len], block_collection) if suffix_len > 0 else block_collection
        self.suffix_blocks = None
        if suffix_len <= 0:
            return
        self.suffix_blocks = jax.tree.map(lambda m: m[-suffix_len:], block_collection)
        if prime_storage is None:
            return
        suffix_keys = jrandom.split(key, suffix_len)
        template = jax.vmap(
            lambda block_key, prime_norm, prime_post_norm, prime_ff: Block(
                config=config,
                key=block_key,
                feed_forward_prime=prime_ff,
                ffn_prime_norm=prime_norm,
                ffn_prime_post_norm=prime_post_norm,
            )
        )(suffix_keys, prime_storage.ffn_prime_norm, prime_storage.ffn_prime_post_norm, prime_storage.feed_forward_prime)
        self.suffix_blocks = eqx.tree_at(
            lambda m: (
                m.seq_norm,
                m.seq_modeling_block,
                m.seq_post_norm,
                m.ffn_norm,
                m.feed_forward,
                m.ffn_post_norm,
            ),
            template,
            (
                self.suffix_blocks.seq_norm,
                self.suffix_blocks.seq_modeling_block,
                self.suffix_blocks.seq_post_norm,
                self.suffix_blocks.ffn_norm,
                self.suffix_blocks.feed_forward,
                self.suffix_blocks.ffn_post_norm,
            ),
        )

    @staticmethod
    def split_state(state, suffix_len: int):
        del suffix_len
        return state, state

    def prefix_call(self, hidden_states: jnp.ndarray, seq: Batch):
        if self.prefix_blocks is None:
            return BaseModelOutput(last_hidden_state=hidden_states, state=None)

        def apply_block(x, block):
            x, _ = block(x, None, seq, is_prefix=True)
            return x, None

        hidden_states, _ = jax.lax.scan(apply_block, hidden_states, self.prefix_blocks, unroll=self.config.unroll_block_scan)
        return BaseModelOutput(last_hidden_state=hidden_states, state=None)

    def suffix_call(self, hidden_states: jnp.ndarray, seq: Batch):
        if self.suffix_blocks is None:
            return BaseModelOutput(last_hidden_state=hidden_states, state=None)

        def apply_block(x, block):
            x, _ = block(x, None, seq, is_prefix=False)
            return x, None

        hidden_states, _ = jax.lax.scan(apply_block, hidden_states, self.suffix_blocks, unroll=self.config.unroll_block_scan)
        return BaseModelOutput(last_hidden_state=hidden_states, state=None)


class BlockCollection(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    blocks: Block
    prime_storage: PrimeStorage | None

    def __init__(self, config: ModelConfig, *, key):
        self.config = config
        key_blocks, key_prime = jrandom.split(key, 2)
        block_keys = jrandom.split(key_blocks, config.num_hidden_layers)
        self.blocks = jax.vmap(lambda k: Block(config, key=k))(block_keys)
        self.prime_storage = PrimeStorage(config, key=key_prime) if config.prime else None

    def __call__(self, hidden_states: jnp.ndarray, seq: Batch):
        def apply_block(x, block):
            x, _ = block(x, None, seq, is_prefix=False)
            return x, None

        hidden_states, _ = scan_or_loop(
            apply_block,
            hidden_states,
            self.blocks,
            use_loop=bool(self.config.unroll_block_scan),
        )
        return BaseModelOutput(last_hidden_state=hidden_states, state=None)


class TransformerModel(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    wte: nn.Embedding
    dropout: nn.Dropout = eqx.field(static=True)
    ln_f: nn.RMSNorm
    h: BlockCollection | BlockCollectionSplit

    def __init__(self, config: ModelConfig, *, key):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        key_embed, key_block = jrandom.split(key, 2)
        self.wte = nn.Embedding(
            weight=jax.nn.initializers.normal(stddev=config.initializer_range, dtype=self.param_dtype)(
                key_embed,
                (config.vocab_size, config.hidden_size),
            )
        )
        self.dropout = nn.Dropout(p=config.embd_pdrop)
        self.h = BlockCollection(config, key=key_block)
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

    def wte_call(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        hidden_states = jax.vmap(self.wte)(input_ids.astype(jnp.int32)).astype(self.compute_dtype)
        return self.dropout(hidden_states)

    def prefix_call(self, hidden_states: jnp.ndarray, seq: Batch):
        assert isinstance(self.h, BlockCollectionSplit)
        return self.h.prefix_call(hidden_states, seq)

    def suffix_call(self, prefix_outputs: jnp.ndarray, seq: Batch):
        assert isinstance(self.h, BlockCollectionSplit)
        outputs = self.h.suffix_call(prefix_outputs, seq)
        rms_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd)
        hidden = jax.vmap(lambda x: rms_fn(self.ln_f, x))(outputs.last_hidden_state)
        return BaseModelOutput(last_hidden_state=hidden, state=None)

    def __call__(self, seq: Batch):
        hidden = self.wte_call(seq.input_ids)
        outputs = self.h(hidden, seq)
        rms_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd)
        hidden = jax.vmap(lambda x: rms_fn(self.ln_f, x))(outputs.last_hidden_state)
        return BaseModelOutput(last_hidden_state=hidden, state=None)


class CausalLM(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    model: TransformerModel
    lm_head: NormalLinear | None

    class Output(eqx.Module):
        last_hidden_states: jnp.ndarray
        logits: jnp.ndarray
        new_state: nn.State | None

    def __init__(self, config: ModelConfig, *, key):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        key_model, key_head = jrandom.split(key, 2)
        self.model = TransformerModel(config, key=key_model)
        self.lm_head = None if config.tie_word_embeddings else NormalLinear(
            config, config.hidden_size, config.output_size, name="lm_head", std=config.initializer_range, key=key_head
        )

    def _disembed(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            return hidden_states @ shared_kernel
        return self.lm_head(hidden_states)

    def wte_call(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        return self.model.wte_call(input_ids)

    def prefix_call(self, hidden_states: jnp.ndarray, seq: Batch):
        return self.model.prefix_call(hidden_states, seq)

    def suffix_call(self, prefix_outputs: jnp.ndarray, seq: Batch):
        outputs = self.model.suffix_call(prefix_outputs, seq)
        return CausalLM.Output(
            last_hidden_states=outputs.last_hidden_state,
            logits=self._disembed(outputs.last_hidden_state),
            new_state=None,
        )

    def __call__(self, seq: Batch) -> "CausalLM.Output":
        outputs = self.model(seq)
        return CausalLM.Output(
            last_hidden_states=outputs.last_hidden_state,
            logits=self._disembed(outputs.last_hidden_state),
            new_state=None,
        )


class MetaModel(eqx.Module):
    class MetricType(StrEnum):
        loss = auto()
        token_nll_loss = auto()
        outer_grad_norm = auto()

    config: Config = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    state_dtype: jnp.dtype = eqx.field(static=True)
    step_index: nn.StateIndex
    language_model: CausalLM

    def __init__(self, config: Config, *, key):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(config.model.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.model.param_dtype)
        self.state_dtype = get_float_dtype_by_name(config.model.state_dtype)
        self.step_index = nn.StateIndex(jnp.array(0, dtype=jnp.int32))
        self.language_model = CausalLM(config.model, key=key)

    def get_ilr_multiplier(self, state: nn.State) -> jnp.ndarray:
        step = state.get(self.step_index)
        warmup_steps = int(self.config.training.ilr_warmup_steps)
        if warmup_steps <= 0:
            return jnp.asarray(1.0, dtype=self.state_dtype)
        progress = jnp.minimum(1.0, (step + 1) / warmup_steps)
        ilr = self.config.training.ilr_init + (self.config.training.optimizer_inner.lr - self.config.training.ilr_init) * progress
        return (ilr / self.config.training.optimizer_inner.lr).astype(self.state_dtype)

    def inner_optimizer(self, state: nn.State):
        return make_optimizer(self.config.training.optimizer_inner, self.get_ilr_multiplier(state))[0]

    def lm_loss(self, seq: Batch, *, prefix_outputs: jnp.ndarray | None = None):
        outputs = self.language_model(seq) if prefix_outputs is None else self.language_model.suffix_call(prefix_outputs, seq)
        loss, pure_ce = cross_entropy_loss_and_accuracy(outputs.logits, seq.target_tokens, seq.loss_masks)
        token_nll = -token_log_probs(outputs.logits, seq.target_tokens)
        return loss, (pure_ce, token_nll)

    def loss_for_sequence(self, seq: Batch, state: nn.State):
        cfg = self.config
        seq_len = int(seq.input_ids.shape[0])
        base_chunk = max(1, int(cfg.model.mini_batch_size))
        tokens_per_chunk = min(seq_len, base_chunk)
        while seq_len % tokens_per_chunk != 0 and tokens_per_chunk > 1:
            tokens_per_chunk -= 1
        metrics: dict[MetaModel.MetricType, jnp.ndarray] = {}

        if str(cfg.training.train_mode) == "pretrain":
            seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

            def process_one(_carry, seq_chunk):
                loss, (loss_ce, token_nll) = self.lm_loss(seq_chunk)
                return None, (loss, loss_ce, token_nll)

            _, (losses, metrics[MetaModel.MetricType.loss], metrics[MetaModel.MetricType.token_nll_loss]) = scan_remat_chunk(
                process_one,
                None,
                seq_chunks,
                remat_n_loops=cfg.training.inner_remat_freq,
                unroll=bool(cfg.model.unroll_inner_scan),
            )
            return losses.mean(), metrics

        block_collection = self.language_model.model.h.blocks
        prime_storage = self.language_model.model.h.prime_storage if cfg.model.prime else None
        split_collection = BlockCollectionSplit(cfg.model, block_collection, prime_storage, key=jrandom.PRNGKey(0))
        model = eqx.tree_at(lambda m: m.language_model.model.h, self, split_collection)

        seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
        embedded = model.language_model.wte_call(seq.input_ids)
        prefix_outputs = model.language_model.prefix_call(embedded, seq).last_hidden_state
        prefix_chunks = tree_rearrange(prefix_outputs, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

        inner_spec = get_filter_spec(model, cfg.training.spec_inner, "inner parameters")
        inner_model, frozen_model = eqx.partition(model, inner_spec)
        inner_opt = model.inner_optimizer(state)
        inner_opt_state = inner_opt.init(inner_model)

        def inner_loss_fn(trainable_inner, frozen_static, seq_chunk, prefix_chunk):
            inner_full = eqx.combine(trainable_inner, frozen_static)
            loss, (loss_ce, token_nll) = inner_full.lm_loss(seq_chunk, prefix_outputs=prefix_chunk)
            return loss, (loss_ce, token_nll)

        collected_loss = []
        collected_token_nll = []
        for idx in range(seq_chunks.input_ids.shape[0]):
            seq_chunk = seq_chunks.slice_index(idx)
            prefix_chunk = prefix_chunks[idx]
            (loss_value, (loss_ce, token_nll)), inner_grads = eqx.filter_value_and_grad(inner_loss_fn, has_aux=True)(
                inner_model, frozen_model, seq_chunk, prefix_chunk
            )
            updates, inner_opt_state = inner_opt.update(inner_grads, inner_opt_state, inner_model)
            inner_model = filter_apply_updates(inner_model, updates)
            collected_loss.append(loss_ce)
            collected_token_nll.append(token_nll)

        metrics[MetaModel.MetricType.loss] = jnp.stack(collected_loss)
        metrics[MetaModel.MetricType.token_nll_loss] = jnp.stack(collected_token_nll)
        return jnp.mean(metrics[MetaModel.MetricType.loss]), metrics

    def weights(self):
        return eqx.filter(self, eqx.is_inexact_array)

    def trainable_parameters(self):
        return filter_parameters(self.weights(), self.config.training.spec_outer, "outer parameters")

    def inner_parameters(self):
        return filter_parameters(self.weights(), self.config.training.spec_inner, "inner parameters")
