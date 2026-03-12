"""Parity-oriented JAX training runtime."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import grain.python as grain
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from ttt.config import Config
from ttt.dataloader import dummy_dataset, lm_dataset
from ttt.runtime import RunArtifacts
from ttt.utils.filter_utils import filter_apply_updates, get_filter_spec
from ttt.utils.jax_utils import initialize_distibuted, set_random_seed, welfords_online_mean

from .checkpoint import OrbaxCheckpointer, resolve_restore_payload, unify_dict_with_eqx_module
from .model.transformer import MetaModel
from .optimizers import make_optimizer
from .sharding import local_device_summary, replicate_pytree, reshape_batch_for_local_devices, unreplicate_pytree
from .wandb_utils import finish_wandb_run, log_wandb_metrics, start_wandb_run


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _mean_metric_tree(tree):
    return jax.tree.map(lambda x: x.mean(axis=0) if hasattr(x, "ndim") and x.ndim > 0 else x, tree)


def _grad_norm(tree) -> float:
    leaves = [leaf for leaf in jax.tree.leaves(tree) if leaf is not None]
    if not leaves:
        return 0.0
    sq = sum(float(jnp.sum(jnp.square(leaf))) for leaf in leaves)
    return float(sq**0.5)


def _pmean_tree(tree):
    return jax.tree.map(
        lambda x: jax.lax.pmean(x, axis_name="data") if x is not None else None,
        tree,
        is_leaf=lambda x: x is None,
    )


def _serialize_model_config(model_cfg):
    if OmegaConf.is_config(model_cfg):
        return OmegaConf.to_container(
            model_cfg,
            resolve=True,
            throw_on_missing=False,
        )
    if is_dataclass(model_cfg):
        return asdict(model_cfg)
    return model_cfg


def _reshape_for_accumulation(batch, accum_steps: int):
    batch_size = int(batch.input_ids.shape[0])
    accum_steps = max(1, accum_steps)
    if batch_size % accum_steps != 0:
        raise ValueError(
            f"Per-device batch size {batch_size} must be divisible by accum_steps={accum_steps}"
        )
    micro_batch_size = batch_size // accum_steps
    return jax.tree.map(
        lambda x: x.reshape((accum_steps, micro_batch_size) + x.shape[1:]),
        batch,
    )


def _make_train_dataset(cfg: Config):
    if cfg.training.dummy_dataset:
        return dummy_dataset(
            seq_len=cfg.training.seq_length,
            global_batch_size=cfg.training.global_batch_size,
            bos_token_id=cfg.model.bos_token_id,
            eos_token_id=cfg.model.eos_token_id,
            repeat=True,
        )
    return lm_dataset(
        path=cfg.training.dataset_path,
        split=cfg.training.data_split,
        seq_len=cfg.training.seq_length,
        global_batch_size=cfg.training.global_batch_size,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        seed=cfg.training.data_seed,
        repeat=True,
    )


def _restore_model(model: MetaModel, restored_weights):
    dynamic = model.weights()
    restored_dynamic, _ = unify_dict_with_eqx_module(restored_weights, dynamic)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(restored_dynamic, static)


def _build_targets(model: MetaModel, optimizer) -> dict[str, Any]:
    trainable = model.trainable_parameters()
    return {
        "model_weights": model.weights(),
        "opt_state": optimizer.init(trainable),
    }


def run(cfg: Config, artifacts: RunArtifacts, logger: logging.Logger) -> None:
    initialize_distibuted(cfg.backend)
    device_info = local_device_summary()
    logger.info("Parity JAX device summary: %s", device_info)

    cfg.model.seq_len = int(cfg.training.seq_length)
    key = set_random_seed(int(cfg.training.model_seed))
    model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
    optimizer, optimizer_info = make_optimizer(cfg.training.optimizer_outer)
    opt_state = optimizer.init(model.trainable_parameters())

    restore_targets = _build_targets(model, optimizer)
    restore_payload = resolve_restore_payload(
        cfg=cfg,
        current_checkpoint_dir=Path(cfg.checkpoint.checkpoint_dir),
        targets=restore_targets,
    )
    restore_meta: dict[str, Any] = {
        "load_part": str(cfg.training.load_part),
        "resume_exp_name": str(cfg.training.resume_exp_name),
        "resume_checkpoint_path": str(cfg.training.resume_checkpoint_path),
        "resume_checkpoint_format": str(cfg.training.resume_checkpoint_format),
    }
    start_step = 0
    if restore_payload is not None:
        model = _restore_model(model, restore_payload.model_weights)
        restore_meta["restore_step"] = int(restore_payload.step)
        restore_meta["restore_status"] = "ok"
        if str(cfg.training.load_part) == "all" and restore_payload.opt_state is not None:
            opt_state = restore_payload.opt_state
            start_step = int(restore_payload.step) + 1
    else:
        restore_meta["restore_status"] = "scratch"

    replicated_model = replicate_pytree(model)
    replicated_opt_state = replicate_pytree(opt_state)

    train_iter = iter(
        _make_train_dataset(cfg).to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, int(cfg.training.loader_workers)),
                prefetch_buffer_size=32,
            )
        )
    )

    outer_spec = get_filter_spec(model, cfg.training.spec_outer, "outer parameters")

    @eqx.filter_pmap(axis_name="data")
    def train_step(model, opt_state, state, batch):
        trainable, frozen = eqx.partition(model, outer_spec)
        accum_steps = max(1, int(cfg.training.accum_steps))

        def loss_fn(trainable_model, micro_batch):
            full_model = eqx.combine(trainable_model, frozen)
            mean_loss, metrics = welfords_online_mean(
                lambda one_seq: full_model.loss_for_sequence(one_seq, state),
                micro_batch,
            )
            return mean_loss, metrics

        micro_batches = _reshape_for_accumulation(batch, accum_steps)

        def microbatch_grad(micro_batch):
            (loss_value, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                trainable,
                micro_batch,
            )
            return loss_value, metrics, grads

        loss_value, metrics, grads = welfords_online_mean(microbatch_grad, micro_batches)
        grads = _pmean_tree(grads)
        loss_value = jax.lax.pmean(loss_value, axis_name="data")
        metrics = _pmean_tree(metrics)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
        new_trainable = filter_apply_updates(trainable, updates)
        new_model = eqx.combine(new_trainable, frozen)
        return new_model, new_opt_state, loss_value, metrics, grads

    checkpointer = OrbaxCheckpointer(cfg.checkpoint.checkpoint_dir)
    wandb_run = start_wandb_run(cfg=cfg, artifacts=artifacts, logger=logger, runtime_mode="jax_train")
    total_params = sum(x.size for x in jax.tree.leaves(model.weights()))

    _append_jsonl(
        artifacts.events_path,
        {
            "event": "run_started",
            "runtime_mode": "jax_train",
            "model_params": int(total_params),
            "restore": restore_meta,
            "model_config": _serialize_model_config(cfg.model),
            "device_info": device_info,
        },
    )

    total_steps = int(cfg.training.total_steps)
    save_freq = max(1, int(cfg.training.save_milestone_freq))
    run_start = time.time()
    tokens_seen = 0

    try:
        for step in range(start_step, total_steps):
            state = state.set(model.step_index, jnp.array(step, dtype=jnp.int32))
            batch = reshape_batch_for_local_devices(next(train_iter))
            replicated_state = replicate_pytree(state)
            replicated_model, replicated_opt_state, loss_value, metrics, grads = train_step(
                replicated_model,
                replicated_opt_state,
                replicated_state,
                batch,
            )
            seq_tokens = int(batch.input_ids.size)
            tokens_seen += seq_tokens

            loss_curve = jax.device_get(metrics[MetaModel.MetricType.loss][0])
            inner_curve = jax.device_get(metrics[MetaModel.MetricType.token_nll_loss][0])
            record = {
                "step": int(step),
                "loss": float(jax.device_get(loss_value[0])),
                "loss_ce": float(jax.device_get(jnp.mean(loss_curve))),
                "gradient_norm": _grad_norm(unreplicate_pytree(grads)),
                "outer_learning_rate": float(jax.device_get(optimizer_info["learning_rate_schedule"](step))),
                "tokens_seen": int(tokens_seen),
                "tokens_in_batch": seq_tokens,
                "runtime_mode": "jax_train",
                "train_mode": str(cfg.training.train_mode),
                "inner_loss_proxy": float(jax.device_get(jnp.mean(inner_curve))),
            }
            _append_jsonl(artifacts.metrics_path, record)
            log_wandb_metrics(
                wandb_run,
                step=step,
                metrics={
                    "train/loss": record["loss"],
                    "train/loss_ce": record["loss_ce"],
                    "train/gradient_norm": record["gradient_norm"],
                    "train/outer_learning_rate": record["outer_learning_rate"],
                    "train/tokens_seen": record["tokens_seen"],
                },
                logger=logger,
            )

            if (step > 0 and step % save_freq == 0) or step == total_steps - 1:
                save_model = unreplicate_pytree(replicated_model)
                save_opt_state = unreplicate_pytree(replicated_opt_state)
                sidecar = checkpointer.save(
                    step=step,
                    model_weights=save_model.weights(),
                    opt_state=save_opt_state,
                    metrics={
                        "loss": record["loss"],
                        "loss_ce": record["loss_ce"],
                        "gradient_norm": record["gradient_norm"],
                        "tokens_seen": record["tokens_seen"],
                        "elapsed_seconds": float(time.time() - run_start),
                    },
                    metadata={
                        "exp_name": str(cfg.training.exp_name),
                        "exp_folder": str(cfg.training.exp_folder),
                        "runtime_mode": "jax_train",
                        "train_mode": str(cfg.training.train_mode),
                        "restore": restore_meta,
                    },
                )
                logger.info("Saved Orbax checkpoint metadata: %s", sidecar)

        elapsed = float(time.time() - run_start)
        _append_jsonl(
            artifacts.events_path,
            {
                "event": "run_finished",
                "runtime_mode": "jax_train",
                "elapsed_seconds": elapsed,
                "total_steps": int(total_steps),
                "tokens_seen": int(tokens_seen),
            },
        )
    finally:
        checkpointer.close()
        finish_wandb_run(wandb_run, logger=logger)
