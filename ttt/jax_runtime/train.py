"""Native JAX training runtime.

This runtime provides a real gradient-based training path while keeping the
current phase-1 experiment/orchestration contracts intact.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp

from ttt.config import Config
from ttt.dataloader import build_batch_iterator
from ttt.runtime import RunArtifacts

from .checkpoint import JaxCheckpointer
from .loop import batch_to_arrays, meta_step, pretrain_step
from .model.transformer import derive_model_spec, init_params, param_count
from .optimizers import build_outer_optimizer
from .sharding import local_device_summary
from .wandb_utils import finish_wandb_run, log_wandb_metrics, start_wandb_run


def _append_jsonl(path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _merge_params_by_shape(target, source):
    """Shape-safe warm-start merge for param-only loading."""

    def _merge_leaf(t, s):
        try:
            same_shape = tuple(getattr(t, "shape", ())) == tuple(getattr(s, "shape", ()))
        except Exception:
            same_shape = False
        if same_shape:
            return jnp.asarray(s, dtype=t.dtype)
        return t

    return jax.tree_util.tree_map(_merge_leaf, target, source)


def _resolve_initial_state(
    *,
    cfg: Config,
    logger: logging.Logger,
    checkpointer: JaxCheckpointer,
    optimizer,
    params,
):
    load_part = str(cfg.training.load_part)
    resume_dir = checkpointer.checkpoint_dir.parent / str(cfg.training.resume_exp_name)
    resume_ckpt = JaxCheckpointer(resume_dir)

    payload: dict[str, Any] = {
        "load_part": load_part,
        "resume_checkpoint_dir": str(resume_dir),
        "resume_exp_name": str(cfg.training.resume_exp_name),
    }

    if load_part == "none":
        opt_state = optimizer.init(params)
        return 0, params, opt_state, payload

    restored = resume_ckpt.load(step=cfg.training.resume_step)
    if restored is None:
        raise FileNotFoundError(
            "Warm-start requested but checkpoint is missing. "
            f"load_part={load_part} resume_checkpoint_dir={resume_dir}"
        )

    payload["restore_status"] = "ok"
    payload["restore_step"] = restored.step

    if load_part == "all":
        logger.info("Restoring full state from %s", resume_dir)
        opt_state = restored.opt_state
        if opt_state is None:
            opt_state = optimizer.init(restored.params)
        return int(restored.step) + 1, restored.params, opt_state, payload

    # params-only restore
    logger.info("Restoring params-only from %s", resume_dir)
    merged_params = _merge_params_by_shape(params, restored.params)
    opt_state = optimizer.init(merged_params)
    return 0, merged_params, opt_state, payload


def run(cfg: Config, artifacts: RunArtifacts, logger: logging.Logger) -> None:
    model_spec = derive_model_spec(cfg)

    device_info = local_device_summary()
    logger.info("JAX runtime device summary: %s", device_info)
    logger.info("Model spec: %s", model_spec.to_dict())

    key = jax.random.PRNGKey(int(cfg.training.model_seed))
    params = init_params(key, model_spec)
    total_params = param_count(params)

    optimizer, lr_schedule = build_outer_optimizer(cfg.training.optimizer_outer)
    checkpointer = JaxCheckpointer(checkpoint_dir=cfg.checkpoint.checkpoint_dir)

    start_step, params, opt_state, restore_payload = _resolve_initial_state(
        cfg=cfg,
        logger=logger,
        checkpointer=checkpointer,
        optimizer=optimizer,
        params=params,
    )

    total_steps = int(cfg.training.total_steps)
    if start_step >= total_steps:
        logger.info(
            "JAX runtime: start_step=%d >= total_steps=%d; nothing to run",
            start_step,
            total_steps,
        )
        return

    iterator = build_batch_iterator(
        dataset_path=str(cfg.training.dataset_path),
        split=str(cfg.training.data_split),
        seq_len=int(cfg.training.seq_length),
        global_batch_size=int(cfg.training.global_batch_size),
        repeat=True,
        shuffle=True,
        seed=int(cfg.training.data_seed),
        dummy_dataset=bool(cfg.training.dummy_dataset),
        vocab_size=int(cfg.model.vocab_size),
    )

    wandb_run = start_wandb_run(
        cfg=cfg,
        artifacts=artifacts,
        logger=logger,
        runtime_mode="jax_train",
    )

    _append_jsonl(
        artifacts.events_path,
        {
            "event": "run_started",
            "runtime_mode": "jax_train",
            "model_params": int(total_params),
            "model_spec": model_spec.to_dict(),
            "restore": restore_payload,
            "wandb_enabled": bool(wandb_run is not None),
        },
    )

    save_freq = int(cfg.training.save_milestone_freq)
    inner_steps = int(cfg.training.jax_inner_steps)
    inner_lr = float(cfg.training.ilr_init)
    max_seq_tokens = int(cfg.training.jax_max_seq_tokens)

    run_start = time.time()
    total_tokens = 0

    try:
        for step in range(start_step, total_steps):
            batch = next(iterator)
            input_ids, targets = batch_to_arrays(batch, max_seq_tokens=max_seq_tokens)

            if str(cfg.training.train_mode) == "meta":
                params, opt_state, metrics = meta_step(
                    params=params,
                    opt_state=opt_state,
                    optimizer=optimizer,
                    spec=model_spec,
                    input_ids=input_ids,
                    targets=targets,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                )
            else:
                params, opt_state, metrics = pretrain_step(
                    params=params,
                    opt_state=opt_state,
                    optimizer=optimizer,
                    spec=model_spec,
                    input_ids=input_ids,
                    targets=targets,
                    use_prime=bool(model_spec.use_prime),
                )

            batch_tokens = int(input_ids.size)
            total_tokens += batch_tokens

            record = {
                "step": int(step),
                "loss": float(metrics.loss),
                "accuracy": float(metrics.accuracy),
                "inner_loss": float(metrics.inner_loss),
                "gradient_norm": float(metrics.gradient_norm),
                "outer_learning_rate": float(lr_schedule(step)),
                "train_mode": str(cfg.training.train_mode),
                "runtime_mode": "jax_train",
                "seq_length": int(cfg.training.seq_length),
                "global_batch_size": int(cfg.training.global_batch_size),
                "tokens_in_batch": batch_tokens,
                "tokens_seen": total_tokens,
                "restore": restore_payload,
                "jax_backend": device_info.get("backend", "unknown"),
            }
            _append_jsonl(artifacts.metrics_path, record)
            log_wandb_metrics(
                wandb_run,
                step=step,
                metrics={
                    "train/loss": record["loss"],
                    "train/accuracy": record["accuracy"],
                    "train/inner_loss": record["inner_loss"],
                    "train/gradient_norm": record["gradient_norm"],
                    "train/outer_learning_rate": record["outer_learning_rate"],
                    "train/tokens_seen": record["tokens_seen"],
                    "train/tokens_in_batch": record["tokens_in_batch"],
                },
                logger=logger,
            )

            should_save = (
                (save_freq > 0 and step > 0 and step % save_freq == 0)
                or step == total_steps - 1
            )
            if should_save:
                ckpt_metrics = {
                    "loss": float(metrics.loss),
                    "accuracy": float(metrics.accuracy),
                    "gradient_norm": float(metrics.gradient_norm),
                    "tokens_seen": int(total_tokens),
                    "elapsed_seconds": float(time.time() - run_start),
                }
                metadata = {
                    "exp_name": str(cfg.training.exp_name),
                    "exp_folder": str(cfg.training.exp_folder),
                    "runtime_mode": "jax_train",
                    "train_mode": str(cfg.training.train_mode),
                    "model_spec": model_spec.to_dict(),
                    "restore": restore_payload,
                }
                sidecar = checkpointer.save(
                    step=step,
                    params=params,
                    opt_state=opt_state,
                    metrics=ckpt_metrics,
                    metadata=metadata,
                )
                logger.info("Saved JAX checkpoint sidecar: %s", sidecar)

        elapsed = float(time.time() - run_start)
        _append_jsonl(
            artifacts.events_path,
            {
                "event": "run_finished",
                "runtime_mode": "jax_train",
                "elapsed_seconds": elapsed,
                "tokens_seen": total_tokens,
            },
        )

        logger.info(
            "JAX runtime finished in %.2fs (steps=%d tokens=%d)",
            elapsed,
            total_steps - start_step,
            total_tokens,
        )
    finally:
        finish_wandb_run(wandb_run, logger=logger)
