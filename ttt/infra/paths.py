"""Validation for the run and folder names that become path components.

Checkpoints live at ``<checkpoint_root>/<exp_folder>/<exp_name>`` and run
directories at ``<exp_dir>/<paper_run_id>/<stage_id>/<run_id>``. The Hydra
config builds the checkpoint path by string concatenation, while the research
orchestrator builds it with ``pathlib``. Those two only agree when every
component is a plain folder name: an absolute ``exp_folder`` is silently
re-rooted by ``pathlib`` on POSIX but produces an invalid path such as
``<root>/C:\\...`` on Windows. The contract is therefore that these values are
names, never paths; callers choose locations with the root arguments
(``--checkpoint-root``/``training.checkpoint_path`` and
``--exp-dir``/``training.exp_dir``). The rules are checked identically on every
OS so a value accepted on Linux is also accepted on Windows.
"""

from __future__ import annotations

_FORBIDDEN_CHARS = ("/", "\\", ":")


def path_name_error(value: object, *, field: str) -> str | None:
    """Return a human-readable error for an invalid name, or ``None`` if valid."""

    if not isinstance(value, str):
        return f"{field} must be a string folder name, got {type(value).__name__}."
    if not value:
        return f"{field} must be a non-empty folder name."
    if value != value.strip():
        return f"{field} must not have leading or trailing whitespace: {value!r}."
    if value in (".", ".."):
        return f"{field} must be a folder name, not {value!r}."
    bad = [ch for ch in _FORBIDDEN_CHARS if ch in value]
    if bad:
        return (
            f"{field} must be a single folder name, not a path: {value!r} "
            f"(contains {' '.join(repr(ch) for ch in bad)})."
        )
    return None


def require_path_name(value: object, *, field: str, hint: str = "") -> str:
    """Return ``value`` if it is a valid folder name, else raise ``ValueError``."""

    error = path_name_error(value, field=field)
    if error is not None:
        raise ValueError(f"{error} {hint}".strip())
    return str(value)
