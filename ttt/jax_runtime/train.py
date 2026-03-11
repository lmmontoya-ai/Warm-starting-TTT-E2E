"""Parity-oriented JAX training runtime."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import grain.python as grain
import jax
import jax.numpy as jnp

from ttt.config import Config
from ttt.dataloader import dummy_dataset, lm_dataset
from ttt.runtime import RunArtifacts
from ttt.utils.filter_utils import filter_apply_updates, get_filter_spec
from ttt.utils.jax_utils import initialize_distibuted, set_random_seed

from .checkpoint import OrbaxCheckpointer, resolve_restore_payload, unify_dict_with_eqx_module
from .model.transformer import MetaModel
from .optimizers import make_optimizer
from .sharding import local_device_summary
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

    train_iter = iter(
        _make_train_dataset(cfg).to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, int(cfg.training.loader_workers)),
                prefetch_buffer_size=32,
            )
        )
    )

    outer_spec = get_filter_spec(model, cfg.training.spec_outer, "outer parameters")

    @eqx.filter_jit
    def train_step(model, opt_state, state, batch):
        trainable, frozen = eqx.partition(model, outer_spec)

        def loss_fn(trainable_model):
            full_model = eqx.combine(trainable_model, frozen)
            losses, metrics = jax.vmap(lambda one_seq: full_model.loss_for_sequence(one_seq, state))(batch)
            mean_loss = losses.mean()
            return mean_loss, _mean_metric_tree(metrics)

        (loss_value, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(trainable)
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
            "model_config": asdict(cfg.model),
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
            batch = next(train_iter)
            model, opt_state, loss_value, metrics, grads = train_step(model, opt_state, state, batch)
            seq_tokens = int(batch.input_ids.size)
            tokens_seen += seq_tokens

            loss_curve = jax.device_get(metrics[MetaModel.MetricType.loss])
            inner_curve = jax.device_get(metrics[MetaModel.MetricType.token_nll_loss])
            record = {
                "step": int(step),
                "loss": float(jax.device_get(loss_value)),
                "loss_ce": float(jax.device_get(jnp.mean(loss_curve))),
                "gradient_norm": _grad_norm(grads),
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
                sidecar = checkpointer.save(
                    step=step,
                    model_weights=model.weights(),
                    opt_state=opt_state,
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
