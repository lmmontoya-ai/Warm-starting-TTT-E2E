"""Shared, inference-only JAX checkpoint and scoring utilities.

This module deliberately stays separate from the training runtime. Tier-1
experiments can add diagnostics and decode support without changing production
training semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from ttt.config import Config
from ttt.utils.jax_utils import initialize_distibuted, set_random_seed, tree_rearrange

from .checkpoint import apply_restored_weights, resolve_restore_payload, verify_restored_values
from .model.data import Batch
from .model.transformer import MetaModel
from .sharding import ModelSharding, put_replicated
from .warmstart_guard import WarmStartValidationError, check_restored_values


def _sanitize_optimizer_cfg(raw_cfg: Any) -> None:
    training = raw_cfg.get("training") if hasattr(raw_cfg, "get") else None
    if training is None:
        return
    for key in ("optimizer", "optimizer_inner"):
        value = training.get(key) if hasattr(training, "get") else None
        if value is None:
            continue
        optimizer_type = str(value.get("optimizer_type", ""))
        if optimizer_type == "sgd":
            for extra in ("emb_wd",):
                if extra in value:
                    del value[extra]


def load_resolved_config(path: Path) -> Config:
    raw_cfg = OmegaConf.load(path.expanduser().resolve())
    _sanitize_optimizer_cfg(raw_cfg)
    merged = OmegaConf.merge(OmegaConf.structured(Config), raw_cfg)
    cfg = OmegaConf.to_object(merged)
    if not isinstance(cfg, Config):
        raise TypeError(f"Expected Config object from {path}")
    return cfg


def restore_model(model: MetaModel, restored_weights):
    return apply_restored_weights(model, restored_weights)


def verify_complete_restore(model: MetaModel, payload, *, label: str) -> None:
    """Fail unless every tensor of ``model`` holds its checkpoint value.

    Same contract as jax_eval: research evals score exactly the checkpoint.
    """

    check_restored_values(verify_restored_values(model.weights(), payload.model_weights), report=payload.report, label=label)
    report = payload.report
    if report is not None and report.fresh_params > 0:
        raise WarmStartValidationError(
            f"{label} left {report.fresh_params/1e6:.2f}M parameters at fresh initialization "
            f"(e.g. {', '.join((report.mismatched_paths or report.missed_paths)[:4])}); the eval config does not "
            "match the checkpoint architecture."
        )


@dataclass
class ResearchEvalSession:
    cfg: Config
    model: MetaModel
    state: Any
    checkpoint_step: int
    model_sharding: ModelSharding
    data_sharding: Any
    replicated_sharding: Any
    n_data_parallel: int

    @property
    def mesh(self):
        return self.model_sharding.mesh


def load_session(
    *,
    resolved_config: Path,
    checkpoint_path: Path,
    context_length: int,
    chunk_length: int | None = None,
) -> ResearchEvalSession:
    cfg = load_resolved_config(resolved_config)
    cfg.training.runtime_mode = cfg.training.RuntimeMode.jax_eval
    cfg.training.load_part = cfg.training.LoadPart.params
    cfg.training.resume_checkpoint_format = "orbax"
    cfg.training.resume_checkpoint_path = str(checkpoint_path.expanduser().resolve())
    cfg.training.seq_length = int(context_length)
    cfg.model.seq_len = int(context_length)
    if chunk_length is not None:
        cfg.model.mini_batch_size = int(chunk_length)

    initialize_distibuted(cfg.backend)
    device_count = int(jax.device_count())
    cfg.backend.num_devices = device_count
    cfg.training.n_state_parallel = 1
    cfg.training.n_data_parallel = device_count

    key = set_random_seed(int(cfg.training.model_seed))
    sharding = ModelSharding(cfg)
    data_sharding = sharding.data_sharding()
    replicated = sharding.replicated_sharding()

    @eqx.filter_jit
    def create_sharded():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        return sharding.shard_params(model), put_replicated(state, replicated)

    @eqx.filter_jit
    def shard_model(model: MetaModel):
        return sharding.shard_params(model)

    checkpoint_path = checkpoint_path.expanduser().resolve()
    with sharding.mesh:
        model, state = create_sharded()
        payload = resolve_restore_payload(
            cfg=cfg,
            current_checkpoint_dir=checkpoint_path
            if checkpoint_path.name.isdigit()
            else checkpoint_path.parent,
            targets={"model_weights": model.weights()},
        )
        if payload is None:
            raise FileNotFoundError(
                f"No checkpoint restore payload at {checkpoint_path}"
            )
        model = shard_model(restore_model(model, payload.model_weights))
        verify_complete_restore(model, payload, label=f"research eval restore from {checkpoint_path}")
        state = put_replicated(
            state.set(model.step_index, jnp.asarray(payload.step, dtype=jnp.int32)),
            replicated,
        )
    return ResearchEvalSession(
        cfg=cfg,
        model=model,
        state=state,
        checkpoint_step=int(payload.step),
        model_sharding=sharding,
        data_sharding=data_sharding,
        replicated_sharding=replicated,
        n_data_parallel=device_count,
    )


def make_padded_batch(
    token_rows: list[list[int]],
    *,
    choice_starts: list[int] | None,
    context_length: int,
    pad_token_id: int,
) -> Batch:
    if not token_rows:
        raise ValueError("Cannot make an empty batch")
    if choice_starts is not None and len(choice_starts) != len(token_rows):
        raise ValueError("choice_starts length mismatch")
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    required = int(context_length) + 1
    for idx, raw_tokens in enumerate(token_rows):
        tokens = [int(x) for x in raw_tokens]
        if len(tokens) > required:
            raise ValueError(f"Token row length {len(tokens)} exceeds {required}")
        actual = len(tokens)
        tokens.extend([int(pad_token_id)] * (required - len(tokens)))
        padded.append(tokens)
        row_mask = [0] * int(context_length)
        if choice_starts is None:
            for pos in range(max(0, actual - 1)):
                row_mask[pos] = 1
        else:
            start = int(choice_starts[idx])
            if start <= 0 or start >= actual:
                raise ValueError(
                    f"Invalid continuation start {start} for {actual} tokens"
                )
            for pos in range(int(context_length)):
                target_index = pos + 1
                row_mask[pos] = int(start <= target_index < actual)
        masks.append(row_mask)
    arr = np.asarray(padded, dtype=np.int32)
    input_ids = arr[:, :-1]
    target_tokens = arr[:, 1:]
    return Batch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=np.asarray(masks, dtype=np.int32),
        attention_mask=None,
        position_ids=np.broadcast_to(
            np.arange(context_length, dtype=np.int32), input_ids.shape
        ),
    )


def make_continuation_score_masks(
    token_rows: list[list[int]],
    *,
    choice_starts: list[int],
    context_length: int,
) -> np.ndarray:
    """Build choice-only readout masks without changing inner-loop loss masks.

    TTT-E2E adapts on the available causal-LM tokens, while a multiple-choice
    harness scores only the candidate continuation. Keeping those two masks
    separate prevents the harness readout from silently changing adaptation.
    """
    if len(token_rows) != len(choice_starts):
        raise ValueError("choice_starts length mismatch")
    masks = np.zeros((len(token_rows), int(context_length)), dtype=np.int32)
    for row_index, (raw_tokens, raw_start) in enumerate(
        zip(token_rows, choice_starts, strict=True)
    ):
        actual = len(raw_tokens)
        start = int(raw_start)
        if actual > int(context_length) + 1:
            raise ValueError(
                f"Token row length {actual} exceeds {int(context_length) + 1}"
            )
        if start <= 0 or start >= actual:
            raise ValueError(f"Invalid continuation start {start} for {actual} tokens")
        masks[row_index, start - 1 : actual - 1] = 1
    return masks


def to_sharded_batch(
    session: ResearchEvalSession, batch: Batch, *, global_batch_size: int
) -> Batch:
    if global_batch_size % session.n_data_parallel != 0:
        raise ValueError(
            f"Batch size {global_batch_size} is not divisible by {session.n_data_parallel} devices"
        )

    def place(arr):
        return jax.make_array_from_process_local_data(
            sharding=session.data_sharding,
            local_data=arr,
            global_shape=(global_batch_size, *arr.shape[1:]),
        )

    placed = jax.tree.map(place, batch)
    return tree_rearrange(
        placed,
        "(data_parallel batch) ... -> data_parallel batch ...",
        data_parallel=session.n_data_parallel,
    )


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(None, 0, None), out_axes=0, axis_name="data_parallel")
def _score_token_nll(meta_model: MetaModel, batch: Batch, state):
    def score_one(seq: Batch):
        _loss, metrics = meta_model.loss_for_sequence(seq, state)
        return metrics[MetaModel.MetricType.token_nll_loss]

    return jax.vmap(score_one)(batch)


def score_token_nll(
    session: ResearchEvalSession,
    batch: Batch,
    *,
    global_batch_size: int,
) -> np.ndarray:
    sharded = to_sharded_batch(session, batch, global_batch_size=global_batch_size)
    with session.mesh:
        values = _score_token_nll(session.model, sharded, session.state)
    result = np.asarray(jax.device_get(values), dtype=np.float32)
    return result.reshape(global_batch_size, result.shape[-1])
