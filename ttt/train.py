"""Phase-1 training entrypoint for local reproduction work.

This is intentionally a lightweight scaffold: it validates config composition,
materializes run directories, and writes resolved configs so experiments can be
tracked while core model/training internals are incrementally implemented in
this repository.
"""

from __future__ import annotations

import logging
from pathlib import Path
from pprint import pformat

import hydra
from omegaconf import OmegaConf

from ttt.config import Config, register_configs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_configs()


def _phase1_run(cfg: Config) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    logger.info("Launching Phase-1 scaffold with config:\n%s", pformat(cfg_dict))

    run_dir = Path(cfg.training.exp_dir) / cfg.training.exp_folder / cfg.training.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "phase1_resolved_config.yaml"
    unresolved_config_path = run_dir / "phase1_unresolved_config.yaml"

    resolved_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    unresolved_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=False))

    summary_lines = [
        "Phase 1 scaffold completed.",
        f"run_dir={run_dir}",
        f"train_mode={cfg.training.train_mode}",
        f"dataset_name={cfg.training.dataset_name}",
        f"dataset_path={cfg.training.dataset_path}",
        f"seq_length={cfg.training.seq_length}",
        f"global_batch_size={cfg.training.global_batch_size}",
        f"resolved_config={resolved_config_path}",
    ]

    for line in summary_lines:
        logger.info(line)


@hydra.main(
    version_base=None,
    config_path=str((Path(__file__).resolve().parents[1] / "configs").resolve()),
    config_name="config",
)
def main(cfg: Config) -> None:
    _phase1_run(cfg)


if __name__ == "__main__":
    main()
