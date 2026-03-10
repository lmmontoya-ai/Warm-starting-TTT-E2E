"""Lightweight W&B integration for JAX runtime."""

from __future__ import annotations

import os
from typing import Any

from ttt.config import Config
from ttt.runtime import RunArtifacts


def _is_disabled(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null", "false", "0"}


def _run_name(cfg: Config, runtime_mode: str) -> str:
    parts = [
        str(cfg.training.paper_run_id).strip(),
        str(cfg.training.stage_id).strip(),
        str(cfg.training.run_id).strip(),
        runtime_mode,
    ]
    parts = [p for p in parts if p]
    return "/".join(parts) if parts else str(cfg.training.exp_name)


def start_wandb_run(
    *,
    cfg: Config,
    artifacts: RunArtifacts,
    logger,
    runtime_mode: str,
):
    if not bool(cfg.training.log_wandb):
        return None

    project = str(cfg.training.wandb_project).strip()
    if _is_disabled(project):
        return None

    try:
        import wandb
    except Exception as exc:
        logger.warning("W&B import failed; continuing without GUI logging: %s", exc)
        return None

    key = str(cfg.training.wandb_key).strip()
    if key and key.lower() not in {"none", "env"}:
        os.environ.setdefault("WANDB_API_KEY", key)

    entity = str(cfg.training.wandb_entity).strip()
    if _is_disabled(entity):
        entity = None

    tags = [
        str(cfg.training.runtime_mode),
        str(cfg.training.train_mode),
        str(cfg.training.stage_id),
    ]
    tags = [t for t in tags if t]

    config = {
        "paper_run_id": str(cfg.training.paper_run_id),
        "stage_id": str(cfg.training.stage_id),
        "run_id": str(cfg.training.run_id),
        "exp_name": str(cfg.training.exp_name),
        "runtime_mode": runtime_mode,
        "train_mode": str(cfg.training.train_mode),
        "seq_length": int(cfg.training.seq_length),
        "global_batch_size": int(cfg.training.global_batch_size),
        "total_steps": int(cfg.training.total_steps),
    }

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=_run_name(cfg, runtime_mode),
            group=str(cfg.training.paper_run_id).strip() or None,
            job_type=runtime_mode,
            tags=tags,
            dir=str(artifacts.run_dir),
            config=config,
            reinit=True,
        )
    except Exception as exc:
        logger.warning("W&B init failed; continuing without GUI logging: %s", exc)
        return None

    return run


def log_wandb_metrics(run, *, step: int, metrics: dict[str, Any], logger) -> None:
    if run is None:
        return

    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)):
            payload[key] = value
    if not payload:
        return

    try:
        run.log(payload, step=int(step))
    except Exception as exc:
        logger.warning("W&B metric logging failed at step=%d: %s", step, exc)


def finish_wandb_run(run, logger) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        logger.warning("W&B finish failed: %s", exc)
