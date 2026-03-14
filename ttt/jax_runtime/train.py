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
from ttt.utils.jax_utils import (
    initialize_distibuted,
    set_random_seed,
)

from .checkpoint import OrbaxCheckpointer, resolve_restore_payload, unify_dict_with_eqx_module
from .loop import make_train_step
from .model.transformer import MetaModel
from .optimizers import make_optimizer
from .sharding import ModelSharding, local_device_summary, put_replicated, to_data_parallel_batch
from .wandb_utils import finish_wandb_run, log_wandb_metrics, start_wandb_run


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


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
    restored_dynamic, _, _ = unify_dict_with_eqx_module(restored_weights, dynamic)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(restored_dynamic, static)


def _block_tree(tree):
    return jax.tree.map(
        lambda x: jax.block_until_ready(x) if eqx.is_array(x) else x,
        tree,
    )


def run(cfg: Config, artifacts: RunArtifacts, logger: logging.Logger) -> None:
    initialize_distibuted(cfg.backend)
    device_info = local_device_summary()
    logger.info("Parity JAX device summary: %s", device_info)

    cfg.model.seq_len = int(cfg.training.seq_length)
    key = set_random_seed(int(cfg.training.model_seed))
    optimizer, optimizer_info = make_optimizer(cfg.training.optimizer_outer)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh
    data_sharding = model_sharding.data_sharding()
    replicated_sharding = model_sharding.replicated_sharding()
    n_data_parallel = int(cfg.training.n_data_parallel)

    train_iter = iter(
        _make_train_dataset(cfg).to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, int(cfg.training.loader_workers)),
                prefetch_buffer_size=32,
            )
        )
    )

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = put_replicated(state, replicated_sharding)
        model = model_sharding.shard_params(model)
        return model, state

    @eqx.filter_jit
    def create_stepped_opt_state(model: MetaModel):
        trainable_params = model.trainable_parameters()
        opt_state = optimizer.init(trainable_params)
        _, opt_state = optimizer.update(
            trainable_params,
            opt_state,
            model.trainable_parameters(),
        )
        return opt_state

    @eqx.filter_jit
    def restore_model_weights(model: MetaModel, restored_weights):
        model = _restore_model(model, restored_weights)
        return model_sharding.shard_params(model)

    @eqx.filter_jit
    def restore_opt_state(model: MetaModel, restored_opt_state):
        opt_state = create_stepped_opt_state(model)
        opt_state, _, _ = unify_dict_with_eqx_module(restored_opt_state, opt_state)
        return opt_state

    with mesh:
        model, state = create_sharded_model_and_state()
        opt_state = optimizer.init(model.trainable_parameters())

        restore_payload = resolve_restore_payload(
            cfg=cfg,
            current_checkpoint_dir=Path(cfg.checkpoint.checkpoint_dir),
            targets={
                "model_weights": model.weights(),
                "opt_state": create_stepped_opt_state(model),
            },
        )
        restore_meta: dict[str, Any] = {
            "load_part": str(cfg.training.load_part),
            "resume_exp_name": str(cfg.training.resume_exp_name),
            "resume_checkpoint_path": str(cfg.training.resume_checkpoint_path),
            "resume_checkpoint_format": str(cfg.training.resume_checkpoint_format),
        }
        start_step = 0
        if restore_payload is not None:
            model = restore_model_weights(model, restore_payload.model_weights)
            restore_meta["restore_step"] = int(restore_payload.step)
            restore_meta["restore_status"] = "ok"
            if str(cfg.training.load_part) == "all" and restore_payload.opt_state is not None:
                opt_state = restore_opt_state(model, restore_payload.opt_state)
                start_step = int(restore_payload.step) + 1
            else:
                opt_state = optimizer.init(model.trainable_parameters())
        else:
            restore_meta["restore_status"] = "scratch"

        checkpointer = OrbaxCheckpointer(cfg.checkpoint.checkpoint_dir)
        wandb_run = start_wandb_run(cfg=cfg, artifacts=artifacts, logger=logger, runtime_mode="jax_train")
        total_params = sum(x.size for x in jax.tree.leaves(model.weights()))
        train_step = make_train_step(cfg, optimizer)

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
                state = put_replicated(
                    state.set(model.step_index, jnp.array(step, dtype=jnp.int32)),
                    replicated_sharding,
                )
                data_wait_started = time.perf_counter()
                raw_batch = next(train_iter)
                data_wait_seconds = float(time.perf_counter() - data_wait_started)

                batch_sharding_started = time.perf_counter()
                batch = to_data_parallel_batch(
                    raw_batch,
                    data_sharding=data_sharding,
                    global_batch_size=int(cfg.training.global_batch_size),
                    n_data_parallel=n_data_parallel,
                )
                batch = _block_tree(batch)
                batch_sharding_seconds = float(time.perf_counter() - batch_sharding_started)

                train_step_started = time.perf_counter()
                model, opt_state, loss_value, metrics = train_step(
                    state,
                    model,
                    opt_state,
                    batch,
                )
                loss_value = jax.block_until_ready(loss_value)
                metrics = _block_tree(metrics)
                train_step_seconds = float(time.perf_counter() - train_step_started)

                seq_tokens = int(cfg.training.global_batch_size) * int(cfg.training.seq_length)
                tokens_seen += seq_tokens

                loss_curve = jax.device_get(metrics[MetaModel.MetricType.loss])
                inner_curve = jax.device_get(metrics[MetaModel.MetricType.token_nll_loss])
                record = {
                    "step": int(step),
                    "loss": float(jax.device_get(loss_value)),
                    "loss_ce": float(jax.device_get(jnp.mean(loss_curve))),
                    "gradient_norm": float(jax.device_get(metrics[MetaModel.MetricType.outer_grad_norm])),
                    "outer_learning_rate": float(jax.device_get(optimizer_info["learning_rate_schedule"](step))),
                    "tokens_seen": int(tokens_seen),
                    "tokens_in_batch": seq_tokens,
                    "runtime_mode": "jax_train",
                    "train_mode": str(cfg.training.train_mode),
                    "inner_loss_proxy": float(jax.device_get(jnp.mean(inner_curve))),
                    "data_wait_seconds": data_wait_seconds,
                    "batch_sharding_seconds": batch_sharding_seconds,
                    "train_step_seconds": train_step_seconds,
                    "checkpoint_save_seconds": 0.0,
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
                        "train/data_wait_seconds": record["data_wait_seconds"],
                        "train/batch_sharding_seconds": record["batch_sharding_seconds"],
                        "train/train_step_seconds": record["train_step_seconds"],
                    },
                    logger=logger,
                )

                if (step > 0 and step % save_freq == 0) or step == total_steps - 1:
                    checkpoint_save_started = time.perf_counter()
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
                    record["checkpoint_save_seconds"] = float(time.perf_counter() - checkpoint_save_started)
                    _append_jsonl(
                        artifacts.metrics_path,
                        {
                            "step": int(step),
                            "runtime_mode": "jax_train",
                            "checkpoint_save_seconds": record["checkpoint_save_seconds"],
                            "checkpoint_sidecar": str(sidecar),
                            "event": "checkpoint_saved",
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
