"""Warm-start restore coverage reporting and validation gates.

A params warm start restores a parent checkpoint into a model whose
architecture may differ from the parent. Three things can go wrong silently:

* a tensor exists in both trees but with different shapes, so the target keeps
  its fresh initialization (this is how the revision-v2 conversions lost every
  feed-forward block when the FFN width changed from 2048 to 1664);
* a tensor path exists only in the target, which is expected for genuinely new
  modules but is a bug when it happens to most of the model (key-path drift);
* the restore "succeeds" but the model behaves like a random init.

This module turns the loader's missed/mismatched lists into a parameter-count
report grouped by component, and provides the gates that the JAX trainer
enforces. It is pure Python so it can be unit tested without JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class WarmStartValidationError(RuntimeError):
    """Raised when a params warm start does not look like a warm start."""


COMPONENT_ORDER = ("embedding", "attention", "ffn", "ffn_prime", "norm", "other")


def classify_param_path(path: str) -> str:
    """Map a flattened parameter key path onto a coarse component label."""

    lowered = path.lower()
    if "prime" in lowered:
        return "ffn_prime"
    if "feed_forward" in lowered or ".mlp" in lowered:
        return "ffn"
    if "embed" in lowered or "wte" in lowered or "lm_head" in lowered:
        return "embedding"
    if "seq_modeling" in lowered or "attention" in lowered or "attn" in lowered:
        return "attention"
    if "norm" in lowered:
        return "norm"
    return "other"


@dataclass(frozen=True)
class RestoreCoverageReport:
    mode: str
    total_params: int
    restored_params: int
    mismatched_params: int
    missed_params: int
    mismatched_paths: tuple[str, ...]
    missed_paths: tuple[str, ...]
    by_component: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def fresh_params(self) -> int:
        return self.mismatched_params + self.missed_params

    @property
    def fresh_fraction(self) -> float:
        return self.fresh_params / self.total_params if self.total_params else 0.0

    @property
    def mismatched_fraction(self) -> float:
        return self.mismatched_params / self.total_params if self.total_params else 0.0

    @property
    def missed_fraction(self) -> float:
        return self.missed_params / self.total_params if self.total_params else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "mode": self.mode,
            "total_params": self.total_params,
            "restored_params": self.restored_params,
            "fresh_params": self.fresh_params,
            "mismatched_params": self.mismatched_params,
            "missed_params": self.missed_params,
            "restored_fraction": 1.0 - self.fresh_fraction,
            "fresh_fraction": self.fresh_fraction,
            "mismatched_fraction": self.mismatched_fraction,
            "missed_fraction": self.missed_fraction,
            "mismatched_paths": list(self.mismatched_paths),
            "missed_paths": list(self.missed_paths),
            "by_component": {k: dict(v) for k, v in self.by_component.items()},
        }

    def summary_lines(self) -> list[str]:
        lines = [
            (
                f"restore mode={self.mode} total={self.total_params/1e6:.2f}M "
                f"restored={self.restored_params/1e6:.2f}M ({100*(1-self.fresh_fraction):.1f}%) "
                f"fresh={self.fresh_params/1e6:.2f}M "
                f"[shape-mismatch {self.mismatched_params/1e6:.2f}M, new {self.missed_params/1e6:.2f}M]"
            )
        ]
        for component in COMPONENT_ORDER:
            row = self.by_component.get(component)
            if not row:
                continue
            lines.append(
                f"  {component:10s} restored={row.get('restored', 0)/1e6:8.2f}M "
                f"mismatched={row.get('mismatched', 0)/1e6:8.2f}M new={row.get('missed', 0)/1e6:8.2f}M"
            )
        return lines


def _path_key(path: str) -> str:
    # Mismatch entries are formatted as "<path>: checkpoint=(..) target=(..)".
    return path.split(":", 1)[0].strip()


def build_coverage_report(
    *,
    mode: str,
    target_leaves: Iterable[tuple[str, int]],
    missed: Iterable[str],
    mismatched: Iterable[str],
) -> RestoreCoverageReport:
    """Build a coverage report from target leaf sizes plus the loader's lists.

    ``target_leaves`` yields ``(key_path, num_elements)`` for every array leaf
    of the target model weights. ``missed`` holds key paths absent from the
    checkpoint; ``mismatched`` holds the loader's shape-mismatch entries.
    """

    missed_set = {_path_key(p) for p in missed}
    mismatched_set = {_path_key(p) for p in mismatched}
    total = restored = n_missed = n_mismatched = 0
    by_component: dict[str, dict[str, int]] = {}
    for path, size in target_leaves:
        size = int(size)
        total += size
        if path in mismatched_set:
            tag = "mismatched"
            n_mismatched += size
        elif path in missed_set:
            tag = "missed"
            n_missed += size
        else:
            tag = "restored"
            restored += size
        bucket = by_component.setdefault(classify_param_path(path), {})
        bucket[tag] = bucket.get(tag, 0) + size
    return RestoreCoverageReport(
        mode=mode,
        total_params=total,
        restored_params=restored,
        mismatched_params=n_mismatched,
        missed_params=n_missed,
        mismatched_paths=tuple(sorted(mismatched_set)),
        missed_paths=tuple(sorted(missed_set)),
        by_component=by_component,
    )


def coverage_report_from_tree(tree: Any, *, mode: str, missed: Iterable[str], mismatched: Iterable[str]) -> RestoreCoverageReport:
    """Convenience wrapper that flattens a JAX pytree of target weights."""

    import jax

    leaves: list[tuple[str, int]] = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if hasattr(value, "shape"):
            size = 1
            for dim in value.shape:
                size *= int(dim)
            leaves.append((jax.tree_util.keystr(path), size))
    return build_coverage_report(mode=mode, target_leaves=leaves, missed=missed, mismatched=mismatched)


def check_restore_report(
    report: RestoreCoverageReport,
    *,
    allow_shape_mismatch: bool,
    max_new_param_fraction: float,
    label: str = "warm start",
) -> None:
    """Raise ``WarmStartValidationError`` if the restore is not a real warm start."""

    problems: list[str] = []
    if report.mismatched_params > 0 and not allow_shape_mismatch:
        sample = ", ".join(report.mismatched_paths[:4])
        problems.append(
            f"{report.mismatched_params/1e6:.2f}M parameters ({100*report.mismatched_fraction:.1f}%) were left at "
            f"fresh initialization because their checkpoint shapes do not match the target model "
            f"(e.g. {sample}). Align the target architecture with the parent (for example keep "
            f"model.intermediate_size equal to the seed's width) or set "
            f"training.warmstart_allow_shape_mismatch=true to accept a partial warm start on purpose."
        )
    if max_new_param_fraction >= 0 and report.missed_fraction > max_new_param_fraction:
        sample = ", ".join(report.missed_paths[:4])
        problems.append(
            f"{100*report.missed_fraction:.1f}% of parameters have no counterpart in the checkpoint "
            f"(limit {100*max_new_param_fraction:.1f}%; e.g. {sample}). This usually means the checkpoint "
            f"key paths do not match the model and the restore silently did nothing. Raise "
            f"training.warmstart_max_new_param_fraction only if the new modules are intentional."
        )
    if problems:
        raise WarmStartValidationError(f"{label} failed restore-coverage validation:\n- " + "\n- ".join(problems))


def check_initial_loss(loss: float, *, max_initial_loss: float, step: int, label: str = "warm start") -> None:
    """Raise if the first logged loss of a params warm start looks like random init."""

    if max_initial_loss <= 0:
        return
    if loss > max_initial_loss:
        raise WarmStartValidationError(
            f"{label} first training loss at step {step} is {loss:.3f}, above the limit "
            f"training.warmstart_max_initial_loss={max_initial_loss:.3f}. A correctly restored language model "
            f"should not start near random-init loss; check restore_report.json in the run directory. Set "
            f"training.warmstart_max_initial_loss=0 to disable this gate."
        )
