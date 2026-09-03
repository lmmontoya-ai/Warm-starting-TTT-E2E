"""Static warm-start compatibility checks over registry stages.

The revision-v2 program lost every feed-forward block of the "warm-started"
models because the converted configs changed ``model.intermediate_size`` while
the loader silently skipped shape-mismatched tensors. These checks catch that
class of mistake at planning time, before any GPU is rented:

* for every stage that restores parameters from a parent stage, compose both
  Hydra configs and compare the model fields that determine parameter shapes;
* report any difference as a finding, distinguishing fields that only add new
  modules (allowed) from fields that change the shape of inherited tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .registry import Registry
from .types import StageSpec


# Model fields whose change alters the shape of tensors the parent also has.
SHAPE_AFFECTING_FIELDS: tuple[str, ...] = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "tie_word_embeddings",
)

# Fields that add or remove modules without reshaping inherited tensors.
MODULE_ADDING_FIELDS: tuple[str, ...] = (
    "prime",
    "suffix_len",
    "prime_intermediate_size",
    "feed_forward_prime",
)

# Non-shape fields where a silent change between parent and child is a common
# source of "restored but behaves like random init" bugs.
BEHAVIOUR_FIELDS: tuple[str, ...] = (
    "rope_theta",
    "qk_norm",
    "pre_norm",
    "post_norm",
    "rms_norm_eps",
)


@dataclass(frozen=True)
class WarmStartFinding:
    stage_id: str
    parent_stage_id: str
    severity: str  # "error" | "warning" | "info"
    field_name: str
    parent_value: Any
    child_value: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "parent_stage_id": self.parent_stage_id,
            "severity": self.severity,
            "field": self.field_name,
            "parent_value": self.parent_value,
            "child_value": self.child_value,
            "message": self.message,
        }


@dataclass(frozen=True)
class WarmStartShapeReport:
    findings: tuple[WarmStartFinding, ...]
    checked_pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[WarmStartFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "status": "PASS" if self.ok else "FAIL",
            "checked_pairs": [list(pair) for pair in self.checked_pairs],
            "findings": [f.to_dict() for f in self.findings],
        }


def _override_value(overrides: Iterable[str], key: str) -> str | None:
    value: str | None = None
    for item in overrides:
        if "=" not in item:
            continue
        left, right = item.split("=", 1)
        if left.strip().lstrip("+") == key:
            value = right.strip()
    return value


def stage_restores_params(stage: StageSpec, experiment_training: dict[str, Any]) -> bool:
    """True when the stage warm-starts parameters from a parent checkpoint."""

    load_part = _override_value(stage.extra_overrides, "training.load_part")
    if load_part is None:
        load_part = str(experiment_training.get("load_part", "none"))
    return load_part in ("params", "all")


def parent_stage_for(stage: StageSpec, stage_map: dict[str, StageSpec], experiment_training: dict[str, Any]) -> StageSpec | None:
    """Resolve the parent stage whose checkpoint the stage restores from."""

    resume_name = _override_value(stage.extra_overrides, "training.resume_exp_name")
    if resume_name is None:
        resume_name = str(experiment_training.get("resume_exp_name", "") or "")
    if resume_name:
        for candidate in stage_map.values():
            if candidate.exp_name == resume_name:
                return candidate
    if stage.required_parent_checkpoint_ids:
        return stage_map.get(stage.required_parent_checkpoint_ids[0])
    return None


def compose_model_section(repo_root: Path, experiment: str, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Compose a Hydra experiment and return its resolved ``model`` and ``training`` sections."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from ttt.config import register_configs

    register_configs()
    model_overrides = [o for o in overrides if o.split("=", 1)[0].lstrip("+").startswith(("model.", "training."))]
    # Dataset roots are mandatory interpolation targets in the deploy profile;
    # they do not affect model shapes, so placeholder paths keep composition pure.
    placeholders = ["deploy_paths.data.dclm_filter_8k=/dev/null", "deploy_paths.data.books3=/dev/null"]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        cfg = compose(
            config_name="config",
            overrides=["+deploy=interactive", f"+experiment={experiment}", *placeholders, *model_overrides],
        )
    model = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=False)
    training = OmegaConf.to_container(cfg.training, resolve=False, throw_on_missing=False)
    assert isinstance(model, dict) and isinstance(training, dict)
    return {"model": dict(model), "training": dict(training)}


def compare_model_sections(
    *,
    stage_id: str,
    parent_stage_id: str,
    parent_model: dict[str, Any],
    child_model: dict[str, Any],
) -> list[WarmStartFinding]:
    findings: list[WarmStartFinding] = []
    for name in SHAPE_AFFECTING_FIELDS:
        p_val, c_val = parent_model.get(name), child_model.get(name)
        if p_val != c_val:
            findings.append(
                WarmStartFinding(
                    stage_id=stage_id,
                    parent_stage_id=parent_stage_id,
                    severity="error",
                    field_name=f"model.{name}",
                    parent_value=p_val,
                    child_value=c_val,
                    message=(
                        f"model.{name} changes from {p_val!r} to {c_val!r}; every tensor whose shape depends on it "
                        "will be silently left at fresh initialization by the params restore."
                    ),
                )
            )
    for name in MODULE_ADDING_FIELDS:
        p_val, c_val = parent_model.get(name), child_model.get(name)
        if p_val != c_val:
            findings.append(
                WarmStartFinding(
                    stage_id=stage_id,
                    parent_stage_id=parent_stage_id,
                    severity="info",
                    field_name=f"model.{name}",
                    parent_value=p_val,
                    child_value=c_val,
                    message=f"model.{name} changes from {p_val!r} to {c_val!r}; this adds or removes modules (expected for conversions).",
                )
            )
    for name in BEHAVIOUR_FIELDS:
        p_val, c_val = parent_model.get(name), child_model.get(name)
        if p_val != c_val:
            findings.append(
                WarmStartFinding(
                    stage_id=stage_id,
                    parent_stage_id=parent_stage_id,
                    severity="warning",
                    field_name=f"model.{name}",
                    parent_value=p_val,
                    child_value=c_val,
                    message=(
                        f"model.{name} changes from {p_val!r} to {c_val!r}; the restored weights were trained under "
                        "the parent value, so the child does not start from the parent's function."
                    ),
                )
            )
    return findings


def check_registry_warmstart_shapes(
    *,
    registry: Registry,
    repo_root: Path,
    stage_ids: Iterable[str] | None = None,
) -> WarmStartShapeReport:
    """Compare every params-restoring stage against its parent stage's model config."""

    stage_map = registry.stage_map()
    selected = [stage_map[s] for s in stage_ids] if stage_ids is not None else list(stage_map.values())
    findings: list[WarmStartFinding] = []
    checked: list[tuple[str, str]] = []
    composed: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def _compose(experiment: str, overrides: Iterable[str]) -> dict[str, Any]:
        key = (experiment, tuple(overrides))
        if key not in composed:
            composed[key] = compose_model_section(repo_root, experiment, overrides)
        return composed[key]

    for stage in selected:
        child = _compose(stage.experiment, stage.extra_overrides)
        if not stage_restores_params(stage, child["training"]):
            continue
        parent = parent_stage_for(stage, stage_map, child["training"])
        if parent is None:
            findings.append(
                WarmStartFinding(
                    stage_id=stage.stage_id,
                    parent_stage_id="",
                    severity="warning",
                    field_name="training.resume_exp_name",
                    parent_value=None,
                    child_value=child["training"].get("resume_exp_name"),
                    message="stage restores parameters but its parent stage could not be resolved from the registry.",
                )
            )
            continue
        parent_cfg = _compose(parent.experiment, parent.extra_overrides)
        checked.append((stage.stage_id, parent.stage_id))
        findings.extend(
            compare_model_sections(
                stage_id=stage.stage_id,
                parent_stage_id=parent.stage_id,
                parent_model=parent_cfg["model"],
                child_model=child["model"],
            )
        )
    return WarmStartShapeReport(findings=tuple(findings), checked_pairs=tuple(checked))


def render_shape_report(report: WarmStartShapeReport) -> str:
    lines = [f"WARM-START SHAPE CHECK: {'PASS' if report.ok else 'FAIL'} ({len(report.checked_pairs)} parent/child pairs)"]
    for stage_id, parent_id in report.checked_pairs:
        lines.append(f"  checked {stage_id} <- {parent_id}")
    for finding in report.findings:
        lines.append(
            f"  [{finding.severity.upper()}] {finding.stage_id} <- {finding.parent_stage_id or '?'}: "
            f"{finding.field_name} {finding.parent_value!r} -> {finding.child_value!r}"
        )
    return "\n".join(lines)
