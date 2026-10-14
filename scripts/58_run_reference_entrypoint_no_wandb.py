#!/usr/bin/env python3
"""Run an entrypoint of the original TTT-E2E code with W&B replaced by a no-op logger.

Used by the upstream parity scripts 41, 50 and 56. --reference-root must point
at a checkout of https://github.com/test-time-training/e2e.

This script runs inside the upstream project's environment, whose top-level
package is also named ``ttt``, so it must not import anything from this repo.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

UPSTREAM_REPO_URL = "https://github.com/test-time-training/e2e"


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
            "Run an entrypoint of the original TTT-E2E code with a no-op W&B logger, "
            "so smoke runs need no W&B account."
        )
    )
    parser.add_argument("--reference-root", type=Path, required=True, help="Upstream TTT-E2E checkout.")
    parser.add_argument("--module", default="ttt.train")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_root = args.reference_root.resolve()
    if not (reference_root / "ttt" / "train.py").is_file():
        print(
            f"Upstream TTT-E2E checkout not found at {reference_root} (missing ttt/train.py). "
            f"Clone {UPSTREAM_REPO_URL} there first.",
            file=sys.stderr,
        )
        return 2
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
