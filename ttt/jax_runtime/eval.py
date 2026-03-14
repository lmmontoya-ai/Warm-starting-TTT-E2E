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

from ttt.config import Config
from ttt.runtime import RunArtifacts
from ttt.utils.jax_utils import initialize_distibuted, set_random_seed

from .checkpoint import resolve_restore_payload, unify_dict_with_eqx_module
from .loop import Evaluator
from .model.transformer import MetaModel
from .sharding import ModelSharding, put_replicated
from .wandb_utils import finish_wandb_run, log_wandb_metrics, start_wandb_run


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _restore_model(model: MetaModel, restored_weights):
    dynamic = model.weights()
    restored_dynamic, _, _ = unify_dict_with_eqx_module(restored_weights, dynamic)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(restored_dynamic, static)


def run(cfg: Config, artifacts: RunArtifacts, logger: logging.Logger) -> None:
    initialize_distibuted(cfg.backend)
    cfg.model.seq_len = int(cfg.training.seq_length)
    key = set_random_seed(int(cfg.training.model_seed))
    model_sharding = ModelSharding(cfg)
    data_sharding = model_sharding.data_sharding()
    replicated_sharding = model_sharding.replicated_sharding()
    n_data_parallel = int(cfg.training.n_data_parallel)

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = put_replicated(state, replicated_sharding)
        model = model_sharding.shard_params(model)
        return model, state

    @eqx.filter_jit
    def restore_model_weights(model: MetaModel, restored_weights):
        model = _restore_model(model, restored_weights)
        return model_sharding.shard_params(model)

    with model_sharding.mesh:
        model, state = create_sharded_model_and_state()
        if not str(cfg.training.resume_checkpoint_path).strip() and not str(cfg.training.resume_exp_name).strip():
            cfg.training.resume_checkpoint_path = str(cfg.checkpoint.checkpoint_dir)
        restore_payload = resolve_restore_payload(
            cfg=cfg,
            current_checkpoint_dir=Path(cfg.checkpoint.checkpoint_dir),
            targets={"model_weights": model.weights()},
        )
        if restore_payload is None:
            raise FileNotFoundError("jax_eval requires a restore source via local checkpoint or training.resume_checkpoint_path.")
        model = restore_model_weights(model, restore_payload.model_weights)

        wandb_run = start_wandb_run(cfg=cfg, artifacts=artifacts, logger=logger, runtime_mode="jax_eval")
        _append_jsonl(
            artifacts.events_path,
            {
                "event": "eval_started",
                "runtime_mode": "jax_eval",
                "checkpoint_step": int(restore_payload.step),
            },
        )

        started = time.time()
        state = put_replicated(
            state.set(model.step_index, jnp.array(restore_payload.step, dtype=jnp.int32)),
            replicated_sharding,
        )
        evaluator = Evaluator(cfg=cfg, data_sharding=data_sharding, n_data_parallel=n_data_parallel)

        try:
            summary = evaluator.evaluate(model=model, state=state, artifacts_dir=artifacts.run_dir)
            head_mean = float(jnp.mean(jnp.asarray(summary.token_nll_curve[: max(1, summary.token_nll_curve.shape[0] // 4)])))
            tail_mean = float(jnp.mean(jnp.asarray(summary.token_nll_curve[-max(1, summary.token_nll_curve.shape[0] // 4) :])))
            summary = {
                "step": int(restore_payload.step),
                "runtime_mode": "jax_eval",
                "eval_loss": float(summary.loss_scalar),
                "eval_loss_ce": float(summary.loss_ce),
                "eval_batches": int(summary.batches),
                "eval_tokens": int(summary.tokens),
                "tokens_per_second": float(summary.tokens / summary.elapsed_seconds),
                "elapsed_seconds": float(summary.elapsed_seconds),
                "per_position_nll_path": "per_position_nll.npy",
                "per_position_nll_head_mean": head_mean,
                "per_position_nll_tail_mean": tail_mean,
            }
            _append_jsonl(artifacts.metrics_path, summary)
            log_wandb_metrics(
                wandb_run,
                step=int(restore_payload.step),
                metrics={
                    "eval/loss": summary["eval_loss"],
                    "eval/loss_ce": summary["eval_loss_ce"],
                    "eval/batches": summary["eval_batches"],
                    "eval/tokens_per_second": summary["tokens_per_second"],
                    "eval/checkpoint_step": summary["step"],
                    "eval/elapsed_seconds": summary["elapsed_seconds"],
                    "eval/per_position_nll_head_mean": summary["per_position_nll_head_mean"],
                    "eval/per_position_nll_tail_mean": summary["per_position_nll_tail_mean"],
                },
                logger=logger,
            )
            _append_jsonl(
                artifacts.events_path,
                {
                    "event": "eval_finished",
                    "runtime_mode": "jax_eval",
                    "elapsed_seconds": float(time.time() - started),
                    "eval_batches": int(summary["eval_batches"]),
                    "eval_tokens": int(summary["eval_tokens"]),
                },
            )
        finally:
            finish_wandb_run(wandb_run, logger=logger)
