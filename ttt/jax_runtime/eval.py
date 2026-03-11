"""Parity-oriented JAX evaluation runtime."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from ttt.config import Config
from ttt.dataloader import dummy_dataset, lm_dataset
from ttt.runtime import RunArtifacts
from ttt.utils.jax_utils import initialize_distibuted, set_random_seed

from .checkpoint import resolve_restore_payload, unify_dict_with_eqx_module
from .model.transformer import MetaModel
from .wandb_utils import finish_wandb_run, log_wandb_metrics, start_wandb_run


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _make_eval_dataset(cfg: Config):
    if cfg.training.dummy_dataset:
        return dummy_dataset(
            seq_len=cfg.training.seq_length,
            global_batch_size=cfg.training.eval_batch_size,
            bos_token_id=cfg.model.bos_token_id,
            eos_token_id=cfg.model.eos_token_id,
            repeat=False,
        )
    return lm_dataset(
        path=cfg.training.dataset_path,
        split=cfg.training.eval_split,
        seq_len=cfg.training.seq_length,
        global_batch_size=cfg.training.eval_batch_size,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        seed=cfg.training.data_seed,
        repeat=False,
        shuffle=False,
    )


def _restore_model(model: MetaModel, restored_weights):
    dynamic = model.weights()
    restored_dynamic, _ = unify_dict_with_eqx_module(restored_weights, dynamic)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(restored_dynamic, static)


def run(cfg: Config, artifacts: RunArtifacts, logger: logging.Logger) -> None:
    initialize_distibuted(cfg.backend)
    cfg.model.seq_len = int(cfg.training.seq_length)
    key = set_random_seed(int(cfg.training.model_seed))
    model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
    if not str(cfg.training.resume_checkpoint_path).strip() and not str(cfg.training.resume_exp_name).strip():
        cfg.training.resume_checkpoint_path = str(cfg.checkpoint.checkpoint_dir)
    restore_payload = resolve_restore_payload(
        cfg=cfg,
        current_checkpoint_dir=Path(cfg.checkpoint.checkpoint_dir),
        targets={"model_weights": model.weights()},
    )
    if restore_payload is None:
        raise FileNotFoundError("jax_eval requires a restore source via local checkpoint or training.resume_checkpoint_path.")
    model = _restore_model(model, restore_payload.model_weights)

    eval_iter = iter(
        _make_eval_dataset(cfg).to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, int(cfg.training.loader_workers)),
                prefetch_buffer_size=32,
            )
        )
    )

    @eqx.filter_jit
    def eval_step(model, state, batch):
        def seq_eval(one_seq):
            loss, metrics = model.loss_for_sequence(one_seq, state)
            return loss, metrics

        losses, metrics = jax.vmap(seq_eval)(batch)
        return losses.mean(), metrics

    wandb_run = start_wandb_run(cfg=cfg, artifacts=artifacts, logger=logger, runtime_mode="jax_eval")
    _append_jsonl(
        artifacts.events_path,
        {
            "event": "eval_started",
            "runtime_mode": "jax_eval",
            "checkpoint_step": int(restore_payload.step),
        },
    )

    max_batches = max(1, int(cfg.training.jax_eval_batches))
    batch_losses: list[float] = []
    nll_curves: list[np.ndarray] = []
    tokens = 0
    started = time.time()

    try:
        for idx, batch in enumerate(eval_iter):
            if idx >= max_batches:
                break
            state = state.set(model.step_index, jnp.array(restore_payload.step, dtype=jnp.int32))
            loss_value, metrics = eval_step(model, state, batch)
            nll = np.asarray(jax.device_get(metrics[MetaModel.MetricType.token_nll_loss]))
            # [batch, chunk, token] or [batch, token]
            if nll.ndim == 3:
                nll = nll.reshape(nll.shape[0], -1)
            nll_curves.append(nll.mean(axis=0))
            batch_losses.append(float(jax.device_get(loss_value)))
            tokens += int(batch.input_ids.size)

        if not batch_losses:
            raise ValueError("jax_eval produced no batches")

        mean_nll = np.mean(np.stack(nll_curves, axis=0), axis=0)
        np.save(artifacts.run_dir / "per_position_nll.npy", mean_nll)

        elapsed = max(float(time.time() - started), 1e-9)
        summary = {
            "step": int(restore_payload.step),
            "runtime_mode": "jax_eval",
            "eval_loss": float(np.mean(batch_losses)),
            "eval_batches": int(len(batch_losses)),
            "eval_tokens": int(tokens),
            "tokens_per_second": float(tokens / elapsed),
            "elapsed_seconds": elapsed,
            "per_position_nll_path": "per_position_nll.npy",
        }
        _append_jsonl(artifacts.metrics_path, summary)
        log_wandb_metrics(
            wandb_run,
            step=int(restore_payload.step),
            metrics={
                "eval/loss": summary["eval_loss"],
                "eval/batches": summary["eval_batches"],
                "eval/tokens_per_second": summary["tokens_per_second"],
            },
            logger=logger,
        )
        _append_jsonl(
            artifacts.events_path,
            {
                "event": "eval_finished",
                "runtime_mode": "jax_eval",
                "elapsed_seconds": elapsed,
                "eval_batches": int(len(batch_losses)),
                "eval_tokens": int(tokens),
            },
        )
    finally:
        finish_wandb_run(wandb_run, logger=logger)
