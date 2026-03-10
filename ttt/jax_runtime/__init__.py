"""Native JAX runtime package."""

from .eval import run as run_eval
from .train import run as run_train

__all__ = ["run_train", "run_eval"]
