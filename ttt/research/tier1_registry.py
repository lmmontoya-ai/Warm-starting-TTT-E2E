"""Strict loader and DAG validator for the Tier-1 inference-stage registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ALLOWED_ACTIONS = {
    "prereg_gate",
    "prepare_t1a",
    "prepare_t1b",
    "prepare_t1c_smoke",
    "validate_t1c",
    "prepare_t1c",
    "restore",
    "eval_t1a",
    "eval_t1b",
    "eval_t1c",
    "eval_t1c_760_smoke",
    "report",
}


@dataclass(frozen=True)
class Tier1Stage:
    stage_id: str
    action: str
    depends_on: tuple[str, ...]
    required: bool
    condition: str = ""
    all_model_checkpoints: bool = False


def load_tier1_registry(path: Path) -> list[Tier1Stage]:
    raw = OmegaConf.to_container(
        OmegaConf.load(path.expanduser().resolve()), resolve=True
    )
    if not isinstance(raw, dict) or str(raw.get("schema_version")) != "1.0":
        raise ValueError("Unsupported Tier-1 registry schema")
    rows = raw.get("stages")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Tier-1 registry has no stages")
    stages: list[Tier1Stage] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Tier-1 registry stage must be a mapping")
        stage = Tier1Stage(
            stage_id=str(row.get("stage_id", "")),
            action=str(row.get("action", "")),
            depends_on=tuple(str(x) for x in row.get("depends_on", [])),
            required=bool(row.get("required", True)),
            condition=str(row.get("condition", "")),
            all_model_checkpoints=bool(row.get("all_model_checkpoints", False)),
        )
        if not stage.stage_id or stage.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid Tier-1 registry stage: {row}")
        if (
            stage.action
            in {
                "restore",
                "eval_t1a",
                "eval_t1b",
                "eval_t1c",
                "eval_t1c_760_smoke",
            }
            and not stage.condition
        ):
            raise ValueError(f"Stage {stage.stage_id} requires a condition")
        stages.append(stage)
    _validate_dag(stages)
    return stages


def _validate_dag(stages: list[Tier1Stage]) -> None:
    ids = [stage.stage_id for stage in stages]
    if len(ids) != len(set(ids)):
        raise ValueError("Tier-1 registry stage ids are not unique")
    known = set(ids)
    for stage in stages:
        missing = sorted(set(stage.depends_on) - known)
        if missing:
            raise ValueError(
                f"Stage {stage.stage_id} has unknown dependencies: {missing}"
            )
    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {stage.stage_id: stage for stage in stages}

    def visit(stage_id: str) -> None:
        if stage_id in visited:
            return
        if stage_id in visiting:
            raise ValueError(
                f"Tier-1 registry contains a dependency cycle at {stage_id}"
            )
        visiting.add(stage_id)
        for dependency in by_id[stage_id].depends_on:
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in ids:
        visit(stage_id)


def dependency_closure(
    stages: list[Tier1Stage], selected: set[str]
) -> list[Tier1Stage]:
    by_id = {stage.stage_id: stage for stage in stages}
    unknown = sorted(selected - set(by_id))
    if unknown:
        raise KeyError(f"Unknown Tier-1 stages: {unknown}")
    included: set[str] = set()

    def include(stage_id: str) -> None:
        if stage_id in included:
            return
        for dependency in by_id[stage_id].depends_on:
            include(dependency)
        included.add(stage_id)

    for stage_id in selected:
        include(stage_id)
    return [stage for stage in stages if stage.stage_id in included]


def registry_plan_payload(stages: list[Tier1Stage]) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": stage.stage_id,
            "action": stage.action,
            "condition": stage.condition,
            "depends_on": list(stage.depends_on),
            "required": stage.required,
            "all_model_checkpoints": stage.all_model_checkpoints,
        }
        for stage in stages
    ]
