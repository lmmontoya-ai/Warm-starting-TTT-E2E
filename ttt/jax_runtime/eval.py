"""Native JAX evaluation runtime."""

from __future__ import annotations

import json
import logging
import time

from ttt.config import Config
from ttt.dataloader import build_batch_iterator
from ttt.runtime import RunArtifacts

from .checkpoint import JaxCheckpointer
from .loop import batch_to_arrays, eval_step
from .model.transformer import derive_model_spec
from .wandb_utils import finish_wandb_run, log_wandb_metrics, start_wandb_run


def _append_jsonl(path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def run(cfg: Config, artifacts: RunArtifacts, logger: logging.Logger) -> None:
    model_spec = derive_model_spec(cfg)
    checkpointer = JaxCheckpointer(checkpoint_dir=cfg.checkpoint.checkpoint_dir)
    restored = checkpointer.load(step=cfg.training.resume_step)
    if restored is None:
        raise FileNotFoundError(
            f"No checkpoint available for jax_eval at {cfg.checkpoint.checkpoint_dir}"
        )

    iterator = build_batch_iterator(
        dataset_path=str(cfg.training.dataset_path),
        split=str(cfg.training.eval_split),
        seq_len=int(cfg.training.seq_length),
        global_batch_size=int(cfg.training.eval_batch_size),
        repeat=False,
        shuffle=False,
        seed=int(cfg.training.data_seed),
        dummy_dataset=bool(cfg.training.dummy_dataset),
        vocab_size=int(cfg.model.vocab_size),
    )

    max_batches = max(1, int(cfg.training.jax_eval_batches))
    max_seq_tokens = int(cfg.training.jax_max_seq_tokens)
    use_prime = bool(model_spec.use_prime and str(cfg.training.train_mode) == "meta")

    wandb_run = start_wandb_run(
        cfg=cfg,
        artifacts=artifacts,
        logger=logger,
        runtime_mode="jax_eval",
    )

    _append_jsonl(
        artifacts.events_path,
        {
            "event": "eval_started",
            "runtime_mode": "jax_eval",
            "checkpoint_step": int(restored.step),
            "max_batches": max_batches,
            "wandb_enabled": bool(wandb_run is not None),
        },
    )

    losses: list[float] = []
    accuracies: list[float] = []
    tokens = 0
    started = time.time()

    try:
        for idx, batch in enumerate(iterator):
            if idx >= max_batches:
                break
            input_ids, targets = batch_to_arrays(batch, max_seq_tokens=max_seq_tokens)
            metrics = eval_step(
                params=restored.params,
                spec=model_spec,
                input_ids=input_ids,
                targets=targets,
                use_prime=use_prime,
            )
            losses.append(float(metrics.loss))
            accuracies.append(float(metrics.accuracy))
            tokens += int(input_ids.size)

            log_wandb_metrics(
                wandb_run,
                step=idx,
                metrics={
                    "eval/batch_loss": float(metrics.loss),
                    "eval/batch_accuracy": float(metrics.accuracy),
                    "eval/tokens_seen": tokens,
                },
                logger=logger,
            )

        elapsed = max(float(time.time() - started), 1e-9)
        if not losses:
            raise ValueError("jax_eval produced no batches")

        summary = {
            "step": int(restored.step),
            "runtime_mode": "jax_eval",
            "eval_loss": float(sum(losses) / len(losses)),
            "eval_accuracy": float(sum(accuracies) / len(accuracies)),
            "eval_batches": int(len(losses)),
            "eval_tokens": int(tokens),
            "tokens_per_second": float(tokens / elapsed),
            "elapsed_seconds": elapsed,
        }
        _append_jsonl(artifacts.metrics_path, summary)
        log_wandb_metrics(
            wandb_run,
            step=int(restored.step),
            metrics={
                "eval/loss": summary["eval_loss"],
                "eval/accuracy": summary["eval_accuracy"],
                "eval/batches": summary["eval_batches"],
                "eval/tokens_per_second": summary["tokens_per_second"],
                "eval/elapsed_seconds": summary["elapsed_seconds"],
            },
            logger=logger,
        )

        _append_jsonl(
            artifacts.events_path,
            {
                "event": "eval_finished",
                "runtime_mode": "jax_eval",
                "elapsed_seconds": elapsed,
                "eval_batches": int(len(losses)),
                "eval_tokens": int(tokens),
            },
        )

        logger.info(
            "JAX eval finished: loss=%.6f acc=%.4f batches=%d tokens=%d",
            summary["eval_loss"],
            summary["eval_accuracy"],
            summary["eval_batches"],
            summary["eval_tokens"],
        )
    finally:
        finish_wandb_run(wandb_run, logger=logger)
