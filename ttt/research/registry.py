"""Registry loader for warm-start staged experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .types import DatasetRef, EvalSpec, StageSpec


@dataclass(frozen=True)
class Registry:
    paper_id: str
    schema_version: str
    datasets: dict[str, DatasetRef]
    eval_profiles: dict[str, EvalSpec]
    stages: list[StageSpec]

    def stage_map(self) -> dict[str, StageSpec]:
        return {s.stage_id: s for s in self.stages}

    def stage(self, stage_id: str) -> StageSpec:
        mapping = self.stage_map()
        if stage_id not in mapping:
            raise KeyError(f"Unknown stage_id: {stage_id}")
        return mapping[stage_id]



def _require_dict(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping for {label}, got {type(payload)}")
    return payload



def load_registry(path: str | Path) -> Registry:
    # Keep unresolved interpolation literals (e.g. ${deploy_paths...}) in registry
    # metadata; concrete values are injected later by orchestrator options.
    raw = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=False)
    root = _require_dict(raw, "registry")

    paper_id = str(root.get("paper_id", "warmstart"))
    schema_version = str(root.get("schema_version", "1.0"))

    datasets_payload = _require_dict(root.get("datasets", {}), "datasets")
    datasets: dict[str, DatasetRef] = {}
    for dataset_id, payload in datasets_payload.items():
        dct = _require_dict(payload, f"datasets.{dataset_id}")
        if "dataset_id" not in dct:
            dct["dataset_id"] = str(dataset_id)
        datasets[str(dataset_id)] = DatasetRef.from_dict(dct)

    eval_payload = _require_dict(root.get("eval_profiles", {}), "eval_profiles")
    eval_profiles: dict[str, EvalSpec] = {}
    for eval_id, payload in eval_payload.items():
        dct = _require_dict(payload, f"eval_profiles.{eval_id}")
        if "eval_id" not in dct:
            dct["eval_id"] = str(eval_id)
        eval_profiles[str(eval_id)] = EvalSpec.from_dict(dct)

    stages_raw = root.get("stages", [])
    if not isinstance(stages_raw, list):
        raise ValueError(f"Expected list for stages, got {type(stages_raw)}")

    stages = [StageSpec.from_dict(_require_dict(s, "stage")) for s in stages_raw]
    _validate_registry(stages=stages, datasets=datasets, eval_profiles=eval_profiles)

    return Registry(
        paper_id=paper_id,
        schema_version=schema_version,
        datasets=datasets,
        eval_profiles=eval_profiles,
        stages=stages,
    )



def _validate_registry(
    *,
    stages: list[StageSpec],
    datasets: dict[str, DatasetRef],
    eval_profiles: dict[str, EvalSpec],
) -> None:
    ids = [s.stage_id for s in stages]
    dup_ids = {x for x in ids if ids.count(x) > 1}
    if dup_ids:
        raise ValueError(f"Duplicate stage_ids in registry: {sorted(dup_ids)}")

    stage_ids = {s.stage_id for s in stages}
    for stage in stages:
        for parent in stage.required_parent_checkpoint_ids:
            if parent and parent not in stage_ids and not parent.startswith("import-"):
                raise ValueError(
                    f"Stage {stage.stage_id} references unknown parent checkpoint id: {parent}"
                )
        for dataset_id in stage.dataset_ids:
            if dataset_id not in datasets:
                raise ValueError(
                    f"Stage {stage.stage_id} references missing dataset id: {dataset_id}"
                )

    for eval_id, eval_spec in eval_profiles.items():
        if not eval_spec.contexts:
            raise ValueError(f"eval profile {eval_id} has empty contexts")
        if not eval_spec.datasets:
            raise ValueError(f"eval profile {eval_id} has empty datasets")



def select_stages(
    registry: Registry,
    *,
    model_key: str = "all",
    path_group: str = "all",
    stage_ids: set[str] | None = None,
) -> list[StageSpec]:
    out: list[StageSpec] = []
    for stage in registry.stages:
        if model_key != "all" and stage.model_key != model_key:
            continue
        if path_group != "all" and stage.path_group != path_group:
            continue
        if stage_ids is not None and stage.stage_id not in stage_ids:
            continue
        out.append(stage)
    return out
