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
        z1 = self.w1(x)
        z1_act = jax.nn.silu(z1)
        z3 = self.w3(x)
        x2 = z1_act * z3
        z2 = self.w2(x2)
        return self.dropout(z2)


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

    def seq_modeling_forward(self, hidden_states: jnp.ndarray, state: nn.State | None, seq: Batch, *, is_prefix: bool):
        rms_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd)
        block_fn = maybe_double_remat(
            partial(self.seq_modeling_block.__class__.__call__, is_prefix=is_prefix),
            prevent_cse=True,
            policy_remat=self.config.remat_attention,
            policy_remat_bwd=self.config.remat_attention_bwd,
        )
        x = jax.vmap(lambda h: rms_fn(self.seq_norm, h))(hidden_states) if self.config.pre_norm else hidden_states
        out, state = block_fn(self.seq_modeling_block, x, seq, state)
        out = jax.vmap(lambda h: rms_fn(self.seq_post_norm, h))(out) if self.config.post_norm else out
        return out, state

    def ffn_forward(
        self,
        feed_forward_fn,
        rms_fn,
        norm: nn.RMSNorm,
        feed_forward: SwiGLUMLP,
        post_norm: nn.RMSNorm,
        hidden_states: jnp.ndarray,
    ):
        x = jax.vmap(lambda h: rms_fn(norm, h))(hidden_states) if self.config.pre_norm else hidden_states
        out = feed_forward_fn(feed_forward, x)
        return jax.vmap(lambda h: rms_fn(post_norm, h))(out) if self.config.post_norm else out

    def __call__(self, hidden_states: jnp.ndarray, state, seq: Batch, *, is_prefix: bool = False):
        config = self.config
        rms_fn = maybe_double_remat(
            nn.RMSNorm.__call__,
            prevent_cse=True,
            policy_remat=config.remat_rms,
            policy_remat_bwd=config.remat_rms_bwd,
        )
        feed_forward_fn = maybe_double_remat(
            self.feed_forward.__class__.__call__,
            prevent_cse=True,
            policy_remat=config.remat_mlp,
            policy_remat_bwd=config.remat_mlp_bwd,
        )
        seq_hidden_states, state = self.seq_modeling_forward(hidden_states, state, seq, is_prefix=is_prefix)
        hidden_states = hidden_states + seq_hidden_states
        if self.feed_forward_prime is not None:
            feed_forward_prime_fn = maybe_double_remat(
                self.feed_forward_prime.__class__.__call__,
                prevent_cse=True,
                policy_remat=config.remat_mlp,
                policy_remat_bwd=config.remat_mlp_bwd,
            )
            prime_hidden_states = self.ffn_forward(
                feed_forward_prime_fn,
                rms_fn,
                self.ffn_prime_norm,
                self.feed_forward_prime,
                self.ffn_prime_post_norm,
                hidden_states,
            )
            hidden_states = hidden_states + prime_hidden_states
        hidden_states = hidden_states + self.ffn_forward(
            feed_forward_fn,
            rms_fn,
            self.ffn_norm,
            self.feed_forward,
            self.ffn_post_norm,
            hidden_states,
        )
        return hidden_states, state


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
        if suffix_len > 0:
            return (
                jax.tree.map(lambda s: s[:-suffix_len], state),
                jax.tree.map(
                    lambda s: s[-suffix_len:] if len(s) >= suffix_len else jnp.zeros((suffix_len, *s.shape[1:]), dtype=s.dtype),
                    state,
                ),
            )
        return (state, None)

    def prefix_call(self, prefix_blocks, hidden_states: jnp.ndarray, state: nn.State, seq: Batch):
        if prefix_blocks is not None:
            prefix_fn = partial(prefix_blocks.__class__.__call__, is_prefix=True)
            block_fn = maybe_double_remat(
                prefix_fn,
                prevent_cse=True,
                policy_remat=self.config.remat_prefix_block,
                policy_remat_bwd="",
            )

            def apply_block_prefix(x, block):
                x, _ = block_fn(block, x, None, seq)
                return x, None

            hidden_states, _ = jax.lax.scan(
                apply_block_prefix,
                hidden_states,
                prefix_blocks,
                unroll=self.config.unroll_block_scan,
            )

        return BaseModelOutput(last_hidden_state=hidden_states, state=state)

    def suffix_call(self, hidden_states: jnp.ndarray, state: nn.State | None, seq: Batch):
        if self.suffix_blocks is not None:
            suffix_fn = partial(self.suffix_blocks.__class__.__call__, is_prefix=False)
            block_fn = maybe_double_remat(
                suffix_fn,
                prevent_cse=True,
                policy_remat=self.config.remat_block,
                policy_remat_bwd=self.config.remat_block_bwd,
            )

            def apply_block_suffix(x, block__substate):
                block, substate = block__substate
                x, substate = block_fn(block, x, substate, seq)
                return x, substate

            hidden_states, state = jax.lax.scan(
                apply_block_suffix,
                hidden_states,
                (self.suffix_blocks, state),
                unroll=self.config.unroll_block_scan,
            )

        return BaseModelOutput(last_hidden_state=hidden_states, state=state)

    def __call__(self, hidden_states, state: tuple[nn.State, nn.State | None], seq: Batch):
        block_fn = maybe_double_remat(
            self.prefix_blocks.__class__.__call__,
            prevent_cse=True,
            policy_remat=self.config.remat_block,
            policy_remat_bwd=self.config.remat_block_bwd,
        )

        def apply_block_prefix(x, block__substate):
            block, substate = block__substate
            x, substate = block_fn(block, x, substate, seq)
            return x, substate

        substate_prefix, substate_suffix = state
        hidden_states, substate_prefix = jax.lax.scan(
            apply_block_prefix,
            hidden_states,
            (self.prefix_blocks, substate_prefix),
            unroll=self.config.unroll_block_scan,
        )

        if self.suffix_blocks is not None:
            def apply_block_suffix(x, block__substate):
                block, substate = block__substate
                x, substate = block_fn(block, x, substate, seq)
                return x, substate

            hidden_states, substate_suffix = jax.lax.scan(
                apply_block_suffix,
                hidden_states,
                (self.suffix_blocks, substate_suffix),
                unroll=self.config.unroll_block_scan,
            )

        return BaseModelOutput(last_hidden_state=hidden_states, state=(substate_prefix, substate_suffix))


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

    def __call__(self, hidden_states: jnp.ndarray, state: nn.State | None, seq: Batch):
        if state is None:
            raise ValueError("BlockCollection requires a model state.")
        substate = state.substate(self.blocks)
        block_fn = maybe_double_remat(
            self.blocks.__class__.__call__,
            prevent_cse=True,
            policy_remat=self.config.remat_block,
            policy_remat_bwd=self.config.remat_block_bwd,
        )

        def apply_block(x, block__substate):
            block, block_state = block__substate
            x, block_state = block_fn(block, x, block_state, seq)
            return x, block_state

        hidden_states, substate = scan_or_loop(
            apply_block,
            hidden_states,
            (self.blocks, substate),
            use_loop=bool(self.config.unroll_block_scan),
        )
        state = state.update(substate)
        return BaseModelOutput(last_hidden_state=hidden_states, state=state)


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

    def prefix_call(self, prefix, hidden_states: jnp.ndarray, state: nn.State | None, seq: Batch):
        assert isinstance(self.h, BlockCollectionSplit)
        return self.h.prefix_call(prefix, hidden_states, state, seq)

    def suffix_call(self, prefix_outputs: jnp.ndarray, state: nn.State | None, seq: Batch):
        assert isinstance(self.h, BlockCollectionSplit)
        outputs = self.h.suffix_call(prefix_outputs, state, seq)
        rms_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd)
        hidden = jax.vmap(lambda x: rms_fn(self.ln_f, x))(outputs.last_hidden_state)
        return BaseModelOutput(last_hidden_state=hidden, state=outputs.state)

    def __call__(self, seq: Batch, state: nn.State):
        hidden = self.wte_call(seq.input_ids)
        outputs = self.h(hidden, state, seq)
        rms_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd)
        hidden = jax.vmap(lambda x: rms_fn(self.ln_f, x))(outputs.last_hidden_state)
        return BaseModelOutput(last_hidden_state=hidden, state=outputs.state)


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

    def prefix_call(self, prefix, hidden_states: jnp.ndarray, state: nn.State, seq: Batch):
        return self.model.prefix_call(prefix, hidden_states, state, seq)

    def suffix_call(self, prefix_outputs: jnp.ndarray, state: nn.State | None, seq: Batch):
        outputs = self.model.suffix_call(prefix_outputs, state, seq)
        return CausalLM.Output(
            last_hidden_states=outputs.last_hidden_state,
            logits=self._disembed(outputs.last_hidden_state),
            new_state=outputs.state,
        )

    def __call__(self, seq: Batch, state: nn.State) -> "CausalLM.Output":
        outputs = self.model(seq, state)
        return CausalLM.Output(
            last_hidden_states=outputs.last_hidden_state,
            logits=self._disembed(outputs.last_hidden_state),
            new_state=outputs.state,
        )


class MetaModel(eqx.Module):
    class Output(eqx.Module):
        lm_output: CausalLM.Output
        state: nn.State

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

    def __call__(self, seq: Batch, state: nn.State) -> "MetaModel.Output":
        raise NotImplementedError

    class InnerLoopStepResult(eqx.Module):
        new_model: "MetaModel"
        new_optimizer_state: OptState
        new_state: nn.State
        metrics: dict["MetaModel.MetricType", jnp.ndarray]

        def __iter__(self):
            return iter((self.new_model, self.new_optimizer_state, self.new_state, self.metrics))

    def inner_loop_step(
        self,
        opt_state: OptState,
        state_tuple: tuple[nn.State, nn.State | None],
        seq: Batch,
        prefix_outputs: jnp.ndarray,
    ) -> InnerLoopStepResult:
        M = MetaModel.MetricType
        metrics: dict[MetaModel.MetricType, jnp.ndarray] = {}
        state_all, suffix_state = state_tuple
        value_and_grad_fn = eqx.filter_value_and_grad(MetaModel.lm_loss, has_aux=True)
        (_loss_with_aux, (metrics[M.loss], metrics[M.token_nll_loss], new_suffix_state)), grads = value_and_grad_fn(
            self,
            seq,
            suffix_state,
            prefix_outputs=prefix_outputs,
        )
        inner_grads = grads.inner_parameters()
        updates, new_optimizer_state = self.inner_optimizer(state_all).update(
            inner_grads,
            opt_state,
            self.inner_parameters(),
        )
        new_model = filter_apply_updates(self, updates)
        return MetaModel.InnerLoopStepResult(
            new_model=new_model,
            new_optimizer_state=new_optimizer_state,
            new_state=(state_all, new_suffix_state),
            metrics=metrics,
        )

    def lm_loss(self, seq: Batch, state: nn.State, *, prefix_outputs: jnp.ndarray | None = None):
        outputs = (
            self.language_model(seq, state)
            if prefix_outputs is None
            else self.language_model.suffix_call(prefix_outputs, state, seq)
        )
        loss, pure_ce = cross_entropy_loss_and_accuracy(outputs.logits, seq.target_tokens, seq.loss_masks)
        token_nll = -token_log_probs(outputs.logits, seq.target_tokens)
        return loss, (pure_ce, token_nll, outputs.new_state)

    def loss_for_sequence(self, seq: Batch, state: nn.State):
        cfg = self.config
        block_collection = self.language_model.model.h.blocks
        prime_storage = self.language_model.model.h.prime_storage if cfg.model.prime else None
        self = eqx.tree_at(
            lambda m: m.language_model.model.h,
            self,
            BlockCollectionSplit(cfg.model, block_collection, prime_storage, key=jrandom.PRNGKey(0)),
        )

        state_prefix_suffix = state.substate(block_collection)
        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        state_all = clone_pytree(state)

        seq_len = cfg.training.seq_length
        tokens_per_chunk = cfg.model.mini_batch_size
        if seq_len % tokens_per_chunk != 0:
            raise ValueError(f"For now, seqlen {seq_len} must be divisible by chunk {tokens_per_chunk}")

        M = MetaModel.MetricType

        if str(cfg.training.train_mode) == "meta":
            model = jax.tree.map(lambda p: p.astype(self.state_dtype), self)
            inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())

            xt_embed = self.language_model.wte_call(seq.input_ids)
            prefix_output = eqx.filter_checkpoint(self.language_model.prefix_call)(
                self.language_model.model.h.prefix_blocks,
                xt_embed,
                state_prefix,
                seq,
            ).last_hidden_state

            def process_suffix_chunk(model__opt_state__state, inputs):
                model_inner, inner_opt_state, state_tuple = model__opt_state__state
                suffix_chunk, prefix_chunk = inputs
                spec_inner = get_filter_spec(model_inner, self.config.training.spec_inner, "inner parameters")
                inner_params, _ = eqx.partition(model_inner, spec_inner)
                _, outer_params = eqx.partition(model, spec_inner)
                model_inner = eqx.combine(inner_params, outer_params)
                new_model, inner_opt_state, state_tuple, metrics = MetaModel.inner_loop_step(
                    model_inner,
                    inner_opt_state,
                    state_tuple,
                    suffix_chunk,
                    prefix_chunk,
                )
                return (new_model, inner_opt_state, state_tuple), metrics

            seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
            prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
            _carry, metrics = scan_remat_chunk(
                eqx.filter_checkpoint(process_suffix_chunk, prevent_cse=False),
                (model, inner_opt_state, (state_all, state_suffix)),
                (seq, prefix_output),
                remat_n_loops=cfg.training.inner_remat_freq,
                unroll=cfg.model.unroll_inner_scan,
            )
            loss = metrics[M.loss].mean()

        elif str(cfg.training.train_mode) == "pretrain":
            metrics: dict[MetaModel.MetricType, jnp.ndarray] = {}
            seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

            def process_one_window(state_tuple, seq_chunk):
                loss, (loss_pure_ce, token_nll_loss, next_state) = self.lm_loss(seq_chunk, state_tuple)
                return next_state, (loss, loss_pure_ce, token_nll_loss)

            _state, (loss, metrics[M.loss], metrics[M.token_nll_loss]) = scan_remat_chunk(
                process_one_window,
                (state_prefix, state_suffix),
                seq,
                remat_n_loops=cfg.training.inner_remat_freq,
                unroll=cfg.model.unroll_inner_scan,
            )
            loss = loss.mean()
        else:
            raise NotImplementedError(f"Training mode {cfg.training.train_mode} not implemented")

        metrics = jax.tree.map(
            lambda x: x if x.ndim == 1 else tree_rearrange(x, "window data ... -> (window data) ..."),
            metrics,
        )
        return loss, metrics

    def weights(self):
        return eqx.filter(self, eqx.is_inexact_array)

    def trainable_parameters(self):
        return filter_parameters(self.weights(), self.config.training.spec_outer, "outer parameters")

    def inner_parameters(self):
        return filter_parameters(self.weights(), self.config.training.spec_inner, "inner parameters")
