"""Budget estimation and accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import count_visible_gpus
from .types import BudgetSpec


@dataclass(frozen=True)
class BudgetUsage:
    tokens_planned: int
    gpu_hours_planned: float
    tokens_observed: int
    gpu_hours_observed: float

    def to_dict(self) -> dict:
        return {
            "tokens_planned": self.tokens_planned,
            "gpu_hours_planned": self.gpu_hours_planned,
            "tokens_observed": self.tokens_observed,
            "gpu_hours_observed": self.gpu_hours_observed,
        }



def estimate_tokens(*, seq_length: int, global_batch_size: int, total_steps: int) -> int:
    return max(0, int(seq_length)) * max(0, int(global_batch_size)) * max(0, int(total_steps))



def estimate_gpu_hours_from_wall(*, wall_seconds: float, num_gpus: int | None = None) -> float:
    gpus = num_gpus if num_gpus is not None else count_visible_gpus()
    return max(0.0, float(wall_seconds)) * max(1, int(gpus)) / 3600.0



def build_budget_manifest(
    *,
    budget_spec: BudgetSpec,
    tokens_planned: int,
    gpu_hours_planned: float,
    tokens_observed: int,
    gpu_hours_observed: float,
) -> dict:
    return {
        "schema_version": budget_spec.schema_version,
        "budget_spec": budget_spec.to_dict(),
        "usage": BudgetUsage(
            tokens_planned=tokens_planned,
            gpu_hours_planned=gpu_hours_planned,
            tokens_observed=tokens_observed,
            gpu_hours_observed=gpu_hours_observed,
        ).to_dict(),
        "token_cap_exceeded": bool(
            budget_spec.token_cap > 0 and tokens_observed > budget_spec.token_cap
        ),
        "gpu_hours_cap_exceeded": bool(
            budget_spec.gpu_hours_cap > 0 and gpu_hours_observed > budget_spec.gpu_hours_cap
        ),
    }
