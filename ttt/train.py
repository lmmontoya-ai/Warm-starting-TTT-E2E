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
from ttt.runtime import Phase1Simulator, prepare_run_artifacts

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_configs()


def _phase1_run(cfg: Config) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    logger.info("Launching Phase-1 runtime with config:\n%s", pformat(cfg_dict))

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

    if not cfg.training.dummy_dataset:
        logger.info(
            "Dummy dataset mode is disabled; runtime stopped after setup. "
            "Set training.dummy_dataset=true to run the phase1 simulator loop."
        )
        return

    simulator = Phase1Simulator(cfg=cfg, artifacts=artifacts, logger=logger)
    simulator.run()


@hydra.main(
    version_base=None,
    config_path=str((Path(__file__).resolve().parents[1] / "configs").resolve()),
    config_name="config",
)
def main(cfg: Config) -> None:
    _phase1_run(cfg)


if __name__ == "__main__":
    main()
