"""Typed contracts for warm-start research orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "1.0"



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class CheckpointRef:
    schema_version: str = SCHEMA_VERSION
    checkpoint_id: str = ""
    exp_folder: str = ""
    exp_name: str = ""
    step: int | None = None
    checkpoint_path: str = ""
    payload_sha256: str = ""
    parent_checkpoint_id: str = ""
    source_model_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "CheckpointRef":
        return CheckpointRef(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            checkpoint_id=str(payload.get("checkpoint_id", "")),
            exp_folder=str(payload.get("exp_folder", "")),
            exp_name=str(payload.get("exp_name", "")),
            step=payload.get("step"),
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            payload_sha256=str(payload.get("payload_sha256", "")),
            parent_checkpoint_id=str(payload.get("parent_checkpoint_id", "")),
            source_model_id=str(payload.get("source_model_id", "")),
        )


@dataclass(frozen=True)
class DatasetRef:
    schema_version: str = SCHEMA_VERSION
    dataset_id: str = ""
    path: str = ""
    split: str = "train"
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    num_tokens: int = 0
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "DatasetRef":
        return DatasetRef(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            dataset_id=str(payload.get("dataset_id", "")),
            path=str(payload.get("path", "")),
            split=str(payload.get("split", "train")),
            tokenizer_id=str(payload.get("tokenizer_id", "")),
            tokenizer_revision=str(payload.get("tokenizer_revision", "")),
            num_tokens=int(payload.get("num_tokens", 0) or 0),
            sha256=str(payload.get("sha256", "")),
        )


@dataclass(frozen=True)
class BudgetSpec:
    schema_version: str = SCHEMA_VERSION
    budget_id: str = ""
    pretrain_steps: int = 0
    adapt_steps: int = 0
    ext_steps: int = 0
    gpu_hours_cap: float = 0.0
    token_cap: int = 0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "BudgetSpec":
        return BudgetSpec(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            budget_id=str(payload.get("budget_id", "")),
            pretrain_steps=int(payload.get("pretrain_steps", 0) or 0),
            adapt_steps=int(payload.get("adapt_steps", 0) or 0),
            ext_steps=int(payload.get("ext_steps", 0) or 0),
            gpu_hours_cap=float(payload.get("gpu_hours_cap", 0.0) or 0.0),
            token_cap=int(payload.get("token_cap", 0) or 0),
            seed=int(payload.get("seed", 0) or 0),
        )


@dataclass(frozen=True)
class EvalSpec:
    schema_version: str = SCHEMA_VERSION
    eval_id: str = ""
    contexts: list[int] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    eval_split: str = "val"
    eval_batches: int = 0
    eval_batch_size: int = 0
    niah_examples: int = 0
    niah_candidates: int = 0
    niah_positions: list[float] = field(default_factory=list)
    decode_steps: int = 0
    decode_prompts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "EvalSpec":
        return EvalSpec(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            eval_id=str(payload.get("eval_id", "")),
            contexts=[int(x) for x in payload.get("contexts", [])],
            datasets=[str(x) for x in payload.get("datasets", [])],
            eval_split=str(payload.get("eval_split", "val")),
            eval_batches=int(payload.get("eval_batches", 0) or 0),
            eval_batch_size=int(payload.get("eval_batch_size", 0) or 0),
            niah_examples=int(payload.get("niah_examples", 0) or 0),
            niah_candidates=int(payload.get("niah_candidates", 0) or 0),
            niah_positions=[float(x) for x in payload.get("niah_positions", [])],
            decode_steps=int(payload.get("decode_steps", 0) or 0),
            decode_prompts=int(payload.get("decode_prompts", 0) or 0),
        )


@dataclass(frozen=True)
class StageSpec:
    schema_version: str = SCHEMA_VERSION
    stage_id: str = ""
    name: str = ""
    kind: str = ""
    model_key: str = ""
    path_group: str = ""
    experiment: str = ""
    exp_name: str = ""
    train_mode: str = ""
    model_pattern: str = ""
    required_parent_checkpoint_ids: list[str] = field(default_factory=list)
    required_profile_keys: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    budget_policy: str = ""
    expected_outputs: list[str] = field(default_factory=list)
    acceptance_gates: list[str] = field(default_factory=list)
    extra_overrides: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "StageSpec":
        return StageSpec(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            stage_id=str(payload.get("stage_id", "")),
            name=str(payload.get("name", "")),
            kind=str(payload.get("kind", "")),
            model_key=str(payload.get("model_key", "")),
            path_group=str(payload.get("path_group", "")),
            experiment=str(payload.get("experiment", "")),
            exp_name=str(payload.get("exp_name", "")),
            train_mode=str(payload.get("train_mode", "")),
            model_pattern=str(payload.get("model_pattern", "")),
            required_parent_checkpoint_ids=[
                str(x) for x in payload.get("required_parent_checkpoint_ids", [])
            ],
            required_profile_keys=[str(x) for x in payload.get("required_profile_keys", [])],
            dataset_ids=[str(x) for x in payload.get("dataset_ids", [])],
            budget_policy=str(payload.get("budget_policy", "")),
            expected_outputs=[str(x) for x in payload.get("expected_outputs", [])],
            acceptance_gates=[str(x) for x in payload.get("acceptance_gates", [])],
            extra_overrides=[str(x) for x in payload.get("extra_overrides", [])],
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class RunSpec:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    paper_run_id: str = ""
    stage_id: str = ""
    model_key: str = ""
    path_group: str = ""
    exp_folder: str = ""
    exp_name: str = ""
    command: list[str] = field(default_factory=list)
    config_overrides: list[str] = field(default_factory=list)
    checkpoint_parents: list[CheckpointRef] = field(default_factory=list)
    datasets: list[DatasetRef] = field(default_factory=list)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    eval_spec: EvalSpec = field(default_factory=EvalSpec)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_parents"] = [x.to_dict() for x in self.checkpoint_parents]
        payload["datasets"] = [x.to_dict() for x in self.datasets]
        payload["budget"] = self.budget.to_dict()
        payload["eval_spec"] = self.eval_spec.to_dict()
        return payload


@dataclass(frozen=True)
class RunResult:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    stage_id: str = ""
    status: str = "unknown"
    created_at_utc: str = ""
    started_at_utc: str = ""
    finished_at_utc: str = ""
    run_dir: str = ""
    metrics_path: str = ""
    events_path: str = ""
    wall_seconds: float = 0.0
    gpu_hours: float = 0.0
    tokens_seen: int = 0
    checkpoint: CheckpointRef = field(default_factory=CheckpointRef)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint"] = self.checkpoint.to_dict()
        return payload


@dataclass(frozen=True)
class EvalResult:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    stage_id: str = ""
    eval_id: str = ""
    status: str = "unknown"
    created_at_utc: str = ""
    finished_at_utc: str = ""
    eval_manifest_path: str = ""
    raw_json_path: str = ""
    raw_csv_path: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperTableRow:
    schema_version: str = SCHEMA_VERSION
    paper_run_id: str = ""
    metric: str = ""
    stage_from: str = ""
    stage_to: str = ""
    n: int = 0
    mean_from: float = 0.0
    mean_to: float = 0.0
    delta: float = 0.0
    std_delta: float = 0.0
    ci95_low: float = 0.0
    ci95_high: float = 0.0
    unit: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_run_result(
    *,
    run_id: str,
    stage_id: str,
    status: str,
    run_dir: str,
    metrics_path: str,
    events_path: str,
    checkpoint: CheckpointRef,
    wall_seconds: float,
    gpu_hours: float,
    tokens_seen: int,
    error_message: str = "",
    started_at_utc: str = "",
    finished_at_utc: str = "",
) -> RunResult:
    now = utc_now_iso()
    return RunResult(
        run_id=run_id,
        stage_id=stage_id,
        status=status,
        created_at_utc=now,
        started_at_utc=started_at_utc or now,
        finished_at_utc=finished_at_utc or now,
        run_dir=run_dir,
        metrics_path=metrics_path,
        events_path=events_path,
        wall_seconds=wall_seconds,
        gpu_hours=gpu_hours,
        tokens_seen=tokens_seen,
        checkpoint=checkpoint,
        error_message=error_message,
    )
