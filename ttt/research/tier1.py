"""Shared Tier-1 configuration, manifest, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


SCHEMA_VERSION = "1.0"
DEFAULT_CONFIG_PATH = Path("configs/research/tier1_eval.yaml")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_tier1_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Tier-1 config must be a mapping: {resolved}")
    validate_tier1_config(raw)
    raw["_config_path"] = str(resolved)
    raw["_config_sha256"] = sha256_file(resolved)
    return raw


def validate_tier1_config(config: dict[str, Any]) -> None:
    if str(config.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError("Unsupported Tier-1 config schema")
    if str(config.get("paper_run_id")) != "revision_v2_t1_eval_v1":
        raise ValueError("Unexpected Tier-1 paper_run_id")
    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("Tier-1 config requires conditions")
    required_fields = {
        "repo_id",
        "source_paper_run_id",
        "source_stage_id",
        "source_run_id",
        "checkpoint_step",
        "architecture",
        "scale",
    }
    for condition_id, raw in conditions.items():
        if not isinstance(raw, dict):
            raise TypeError(f"Condition {condition_id} must be a mapping")
        missing = sorted(required_fields - set(raw))
        if missing:
            raise ValueError(f"Condition {condition_id} missing: {', '.join(missing)}")
    for section, field in (("t1a", "tasks"), ("t1b", "conditions"), ("t1c", "tasks")):
        if not config.get(section, {}).get(field):
            raise ValueError(f"Tier-1 config {section}.{field} is empty")


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    scale: str
    architecture: str
    repo_id: str
    source_paper_run_id: str
    source_stage_id: str
    source_run_id: str
    checkpoint_step: int
    evidence_fence: str = ""

    @staticmethod
    def from_config(condition_id: str, raw: dict[str, Any]) -> "ConditionSpec":
        return ConditionSpec(
            condition_id=condition_id,
            scale=str(raw["scale"]),
            architecture=str(raw["architecture"]),
            repo_id=str(raw["repo_id"]),
            source_paper_run_id=str(raw["source_paper_run_id"]),
            source_stage_id=str(raw["source_stage_id"]),
            source_run_id=str(raw["source_run_id"]),
            checkpoint_step=int(raw["checkpoint_step"]),
            evidence_fence=str(raw.get("evidence_fence", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def condition_spec(config: dict[str, Any], condition_id: str) -> ConditionSpec:
    conditions = config["conditions"]
    if condition_id not in conditions:
        raise KeyError(f"Unknown Tier-1 condition: {condition_id}")
    return ConditionSpec.from_config(condition_id, conditions[condition_id])


def stable_rank(*, salt: str, task: str, example_id: str) -> str:
    return hashlib.sha256(f"{salt}|{task}|{example_id}".encode("utf-8")).hexdigest()


def make_manifest(
    *,
    manifest_type: str,
    paper_run_id: str,
    provenance: dict[str, Any],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": manifest_type,
        "paper_run_id": paper_run_id,
        "provenance": provenance,
        "example_count": len(examples),
        "examples": examples,
    }
    return {**core, "manifest_sha256": sha256_payload(core)}


def verify_manifest(
    payload: dict[str, Any], *, expected_type: str | None = None
) -> str:
    if str(payload.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    if expected_type is not None and payload.get("manifest_type") != expected_type:
        raise ValueError(
            f"Expected {expected_type}, got {payload.get('manifest_type')}"
        )
    claimed = str(payload.get("manifest_sha256", ""))
    core = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = sha256_payload(core)
    if not claimed or claimed != actual:
        raise ValueError(f"Manifest hash mismatch: claimed={claimed!r} actual={actual}")
    examples = payload.get("examples")
    if not isinstance(examples, list) or int(payload.get("example_count", -1)) != len(
        examples
    ):
        raise ValueError("Manifest example_count mismatch")
    return actual


def expected_restore_paths(
    *, config: dict[str, Any], condition_id: str, exp_dir: Path, checkpoint_root: Path
) -> tuple[Path, Path]:
    spec = condition_spec(config, condition_id)
    paper_run_id = str(config["paper_run_id"])
    run_dir = (
        exp_dir.expanduser().resolve()
        / paper_run_id
        / condition_id
        / spec.source_run_id
    )
    checkpoint_dir = (
        checkpoint_root.expanduser().resolve() / paper_run_id / spec.source_run_id
    )
    return run_dir, checkpoint_dir
