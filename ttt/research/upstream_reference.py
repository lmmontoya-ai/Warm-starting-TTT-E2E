"""Locate a checkout of the original TTT-E2E code for the upstream parity scripts.

scripts/41, 50, 56 and 58 run the original authors' trainer
(https://github.com/test-time-training/e2e) on the same data and checkpoints
as this repo's runtime. The checkout is not part of this repo; by default the
scripts look for it at ``ttte2e_reference/e2e`` (gitignored), or at
``--reference-root``.
"""

from __future__ import annotations

from pathlib import Path


UPSTREAM_REPO_URL = "https://github.com/test-time-training/e2e"
DEFAULT_REFERENCE_SUBDIR = Path("ttte2e_reference") / "e2e"
REQUIRED_FILES = ("pyproject.toml", "ttt/train.py")


class UpstreamReferenceError(RuntimeError):
    """Raised when the upstream TTT-E2E checkout is missing or incomplete."""


def resolve_reference_root(repo_root: Path, override: Path | None = None) -> Path:
    """Return the upstream checkout root, or raise with instructions to clone it."""

    root = (override if override is not None else repo_root / DEFAULT_REFERENCE_SUBDIR).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise UpstreamReferenceError(
            f"Upstream TTT-E2E checkout not found at {root} (missing: {', '.join(missing)}).\n"
            "This parity check runs the original authors' code, which is not part of this repo. "
            "Clone it first:\n"
            f"    git clone {UPSTREAM_REPO_URL} {repo_root / DEFAULT_REFERENCE_SUBDIR}\n"
            "or pass --reference-root PATH to an existing checkout."
        )
    return root
