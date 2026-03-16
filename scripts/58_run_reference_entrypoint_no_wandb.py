#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


class _NoOpWandbLogger:
    def __init__(self, *args, **kwargs):
        self.is_master = True
        self.enabled = False
        self.preexisting = False
        self.run = None

    def log(self, metrics: dict, step: int):
        return None

    def save(self, path: str | Path, base_path: str | Path = "./"):
        return None

    def log_token_nll_loss(self, token_nll_loss, step: int, k: str):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reference entrypoint from the read-only snapshot with a no-op "
            "W&B logger so smoke tests remain faithful without external logging."
        )
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--module", default="ttt.train")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_root = args.reference_root.resolve()
    os.chdir(reference_root)
    sys.path.insert(0, str(reference_root))

    # Prevent any latent W&B auto-init from interacting with the host env.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")

    module = importlib.import_module(args.module)
    try:
        wandb_utils = importlib.import_module("ttt.infra.wandb_utils")
        wandb_utils.WandbLogger = _NoOpWandbLogger
    except ModuleNotFoundError:
        pass
    module.WandbLogger = _NoOpWandbLogger

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    sys.argv = [args.module, *overrides]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
