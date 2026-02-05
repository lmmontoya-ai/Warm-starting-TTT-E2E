# scripts/01_config_inspect.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Hydra config resolution for the TTT-E2E repo."
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable), e.g. +deploy=interactive",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the full composed config as YAML.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from ttt.config import register_configs  # pylint: disable=import-error

    register_configs()
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=args.override)

    # Provide a concise summary of the resolution graph
    summary = {
        "training.dataset_name": cfg.training.dataset_name,
        "training.dataset_path": cfg.training.dataset_path,
        "deploy_paths.data": {
            "dclm_filter_8k": getattr(cfg.deploy_paths.data, "dclm_filter_8k", None),
            "books3": getattr(cfg.deploy_paths.data, "books3", None),
        },
        "checkpoint.checkpoint_dir": cfg.checkpoint.checkpoint_dir,
        "checkpoint.resume_checkpoint_dir": cfg.checkpoint.resume_checkpoint_dir,
        "backend": {
            "distributed": cfg.backend.distributed,
            "num_devices": cfg.backend.num_devices,
        },
    }

    print("\n=== Hydra Config Summary ===")
    print(OmegaConf.to_yaml(summary, resolve=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(OmegaConf.to_yaml(cfg, resolve=False))
        print(f"Wrote full config to: {args.out}")

    if not args.override:
        print(
            "\nTip: Try with overrides, for example:\n"
            "  python scripts/01_config_inspect.py \\\n"
            "    --override +deploy=interactive \\\n"
            "    --override deploy_paths.data.dclm_filter_8k=/data/dclm_filter_8k \\\n"
            "    --override deploy_paths.data.books3=/data/books3 \\\n"
            "    --override deploy_paths.checkpoint=/checkpoints\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
