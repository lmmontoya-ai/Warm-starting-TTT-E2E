"""Phase-1 training entrypoint for local reproduction work.

This entrypoint now supports:
1) config composition and run artifact generation, and
2) a deterministic local simulation loop for experiment orchestration
   (`training.dummy_dataset=true`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from pprint import pformat

import hydra
from omegaconf import OmegaConf

from ttt.config import Config, register_configs
from ttt.runtime import (
    Phase1Simulator,
    Phase1TokenStatsTrainer,
    prepare_run_artifacts,
)
from ttt.research.tracking import detect_git_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_configs()


def _mask_secrets(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k).lower()
            if key in {"wandb_key", "api_key", "token", "access_token"}:
                out[k] = "***"
            else:
                out[k] = _mask_secrets(v)
        return out
    if isinstance(value, list):
        return [_mask_secrets(v) for v in value]
    return value


def _phase1_run(cfg: Config) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    logger.info(
        "Launching Phase-1 runtime with config:\n%s",
        pformat(_mask_secrets(cfg_dict)),
    )

    artifacts = prepare_run_artifacts(cfg)

    summary_lines = [
        "Phase 1 runtime initialized.",
        f"run_dir={artifacts.run_dir}",
        f"train_mode={cfg.training.train_mode}",
        f"dataset_name={cfg.training.dataset_name}",
        f"dataset_path={cfg.training.dataset_path}",
        f"seq_length={cfg.training.seq_length}",
        f"global_batch_size={cfg.training.global_batch_size}",
        f"resolved_config={artifacts.resolved_config_path}",
    ]
    for line in summary_lines:
        logger.info(line)

    runtime_mode = str(cfg.training.runtime_mode)
    logger.info("runtime_mode=%s", runtime_mode)

    git = detect_git_state(Path(__file__).resolve().parents[1])
    if bool(git.dirty) and not bool(cfg.training.allow_dirty_repo):
        raise RuntimeError(
            "Repository is dirty and training.allow_dirty_repo=false. "
            "Commit or stash changes before launching runs."
        )

    if runtime_mode == "simulate":
        if not cfg.training.dummy_dataset:
            logger.info(
                "Dummy dataset mode is disabled; runtime stopped after setup. "
                "Set training.dummy_dataset=true to run the phase1 simulator loop."
            )
            return

        simulator = Phase1Simulator(cfg=cfg, artifacts=artifacts, logger=logger)
        simulator.run()
        return

    if runtime_mode == "token_stats":
        trainer = Phase1TokenStatsTrainer(cfg=cfg, artifacts=artifacts, logger=logger)
        trainer.run()
        return

    if runtime_mode in {"jax_train", "jax_eval"}:
        if runtime_mode == "jax_train":
            from ttt.jax_runtime.train import run as jax_train_run

            jax_train_run(cfg=cfg, artifacts=artifacts, logger=logger)
        else:
            from ttt.jax_runtime.eval import run as jax_eval_run

            jax_eval_run(cfg=cfg, artifacts=artifacts, logger=logger)
        return

    raise ValueError(
        f"Unsupported runtime_mode={runtime_mode!r}. "
        "Expected one of: simulate, token_stats, jax_train, jax_eval."
    )


@hydra.main(
    version_base=None,
    config_path=str((Path(__file__).resolve().parents[1] / "configs").resolve()),
    config_name="config",
)
def main(cfg: Config) -> None:
    _phase1_run(cfg)


if __name__ == "__main__":
    main()
