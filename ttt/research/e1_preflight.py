"""E1 pre-flight artifact inventory and gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .lineage import resolve_checkpoint_ref
from .preregistration import compare_fingerprints
from .registry import Registry


CHECKS_PER_CHECKPOINT = ("exists", "fingerprint", "fingerprint_matches_canonical", "restores_clean")
CHECKS_PER_DATA = ("exists", "fingerprint", "fingerprint_matches_canonical")


@dataclass(frozen=True)
class PreflightArtifact:
    artifact_id: str
    kind: str
    path: str
    exists: bool
    fingerprint: dict[str, Any] | str
    fingerprint_matches_canonical: bool
    restores_clean: bool | None
    status: str
    reason: str
    checks_executed: list[str] = field(default_factory=list)
    expected_exp_name: str = ""
    canonical_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightResult:
    schema_version: str
    status: str
    checks_executed_count: int
    artifacts: list[PreflightArtifact]
    output_json: str
    output_md: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _experiment_config_path(repo_root: Path, experiment: str) -> Path:
    return repo_root / "configs" / "experiment" / f"{experiment}.yaml"


def _resume_exp_name(repo_root: Path, experiment: str) -> str:
    path = _experiment_config_path(repo_root, experiment)
    payload = _load_yaml(path)
    training = payload.get("training", {})
    if not isinstance(training, dict):
        raise ValueError(f"Expected training mapping in {path}")
    resume = str(training.get("resume_exp_name", "")).strip()
    if not resume:
        raise ValueError(f"Missing training.resume_exp_name in {path}")
    return resume


def _short_hash(value: str) -> str:
    if not value:
        return ""
    return value[:12]


def _load_fingerprint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected fingerprint object: {path}")
    return payload


def _fingerprint_summary(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset", payload)
    if not isinstance(dataset, dict):
        return {"invalid": True}
    return {
        "dataset_id": str(dataset.get("dataset_id", "")),
        "split": str(dataset.get("split", "")),
        "sha256": _short_hash(str(dataset.get("sha256", ""))),
        "num_tokens": int(dataset.get("num_tokens", 0) or 0),
        "tokenizer_id": str(dataset.get("tokenizer_id", "")),
        "tokenizer_revision": str(dataset.get("tokenizer_revision", "")),
    }


def _is_orbax_checkpoint_payload(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not path.name.isdigit():
        return False
    parent = path.parent
    step_meta = parent / f"step_metadata_{int(path.name):08d}.json"
    if step_meta.exists():
        try:
            payload = json.loads(step_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if str(payload.get("checkpoint_format", "")).lower() == "orbax":
            return True
    return (path / "model_weights").exists()


def _checkpoint_artifact(
    *,
    artifact_id: str,
    checkpoint_root: Path,
    exp_folder: str,
    checkpoint_id: str,
    exp_name: str,
    canonical_source: str,
) -> PreflightArtifact:
    expected_dir = (checkpoint_root / exp_folder / exp_name).expanduser().resolve()
    latest_path = expected_dir / "latest.json"
    checks: list[str] = ["exists"]
    if not latest_path.exists():
        return PreflightArtifact(
            artifact_id=artifact_id,
            kind="checkpoint",
            path=str(latest_path),
            exists=False,
            fingerprint={},
            fingerprint_matches_canonical=False,
            restores_clean=False,
            status="FAIL",
            reason="uncertain provenance: missing local checkpoint latest pointer",
            checks_executed=checks,
            expected_exp_name=exp_name,
            canonical_source=canonical_source,
        )

    try:
        ref = resolve_checkpoint_ref(
            checkpoint_root=checkpoint_root,
            exp_folder=exp_folder,
            checkpoint_id=checkpoint_id,
            exp_name=exp_name,
            allow_missing=False,
        )
        checks.append("fingerprint")
        fingerprint = {
            "checkpoint_id": ref.checkpoint_id,
            "exp_folder": ref.exp_folder,
            "exp_name": ref.exp_name,
            "step": ref.step,
            "payload_sha256": _short_hash(ref.payload_sha256),
        }
        payload_path = Path(ref.checkpoint_path)
        matches = ref.exp_name == exp_name and ref.checkpoint_id == checkpoint_id
        checks.append("fingerprint_matches_canonical")
        restores_clean = _is_orbax_checkpoint_payload(payload_path)
        checks.append("restores_clean")
        status = "PASS" if matches and restores_clean else "FAIL"
        if not matches:
            reason = "checkpoint identity does not match registry/canonical stage"
        elif not restores_clean:
            reason = "checkpoint exists but is not a current-pipeline Orbax payload"
        else:
            reason = ""
        return PreflightArtifact(
            artifact_id=artifact_id,
            kind="checkpoint",
            path=str(payload_path),
            exists=True,
            fingerprint=fingerprint,
            fingerprint_matches_canonical=matches,
            restores_clean=restores_clean,
            status=status,
            reason=reason,
            checks_executed=checks,
            expected_exp_name=exp_name,
            canonical_source=canonical_source,
        )
    except Exception as exc:
        return PreflightArtifact(
            artifact_id=artifact_id,
            kind="checkpoint",
            path=str(latest_path),
            exists=True,
            fingerprint={},
            fingerprint_matches_canonical=False,
            restores_clean=False,
            status="FAIL",
            reason=f"checkpoint check errored: {exc}",
            checks_executed=checks,
            expected_exp_name=exp_name,
            canonical_source=canonical_source,
        )


def _data_artifact(
    *,
    artifact_id: str,
    root: Path,
    canonical_root: Path,
    canonical_source: str,
) -> PreflightArtifact:
    resolved_root = root.expanduser().resolve()
    checks: list[str] = ["exists"]
    split_paths = {
        "train": resolved_root / "train.fingerprint.json",
        "val": resolved_root / "val.fingerprint.json",
    }
    canonical_paths = {
        "train": canonical_root.expanduser().resolve() / "train.fingerprint.json",
        "val": canonical_root.expanduser().resolve() / "val.fingerprint.json",
    }
    missing = [f"{split}:{path}" for split, path in split_paths.items() if not path.exists()]
    if missing:
        return PreflightArtifact(
            artifact_id=artifact_id,
            kind="data",
            path=str(resolved_root),
            exists=False,
            fingerprint={},
            fingerprint_matches_canonical=False,
            restores_clean=None,
            status="FAIL",
            reason="missing data fingerprint sidecar(s): " + "; ".join(missing),
            checks_executed=checks,
            canonical_source=canonical_source,
        )

    try:
        fingerprints = {split: _load_fingerprint(path) for split, path in split_paths.items()}
        checks.append("fingerprint")
        missing_canonical = [
            f"{split}:{path}" for split, path in canonical_paths.items() if not path.exists()
        ]
        if missing_canonical:
            return PreflightArtifact(
                artifact_id=artifact_id,
                kind="data",
                path=str(resolved_root),
                exists=True,
                fingerprint={split: _fingerprint_summary(payload) for split, payload in fingerprints.items()},
                fingerprint_matches_canonical=False,
                restores_clean=None,
                status="FAIL",
                reason="missing canonical fingerprint sidecar(s): " + "; ".join(missing_canonical),
                checks_executed=checks,
                canonical_source=canonical_source,
            )
        canonical = {split: _load_fingerprint(path) for split, path in canonical_paths.items()}
        mismatches: list[str] = []
        for split in ("train", "val"):
            comparison = compare_fingerprints(
                fingerprints[split],
                canonical[split],
                label_left=f"{artifact_id}:{split}",
                label_right=f"canonical:{split}",
            )
            if not comparison.equal:
                mismatches.extend(comparison.differences)
        checks.append("fingerprint_matches_canonical")
        matches = not mismatches
        return PreflightArtifact(
            artifact_id=artifact_id,
            kind="data",
            path=str(resolved_root),
            exists=True,
            fingerprint={split: _fingerprint_summary(payload) for split, payload in fingerprints.items()},
            fingerprint_matches_canonical=matches,
            restores_clean=None,
            status="PASS" if matches else "FAIL",
            reason="" if matches else "fingerprint mismatch: " + "; ".join(mismatches),
            checks_executed=checks,
            canonical_source=canonical_source,
        )
    except Exception as exc:
        return PreflightArtifact(
            artifact_id=artifact_id,
            kind="data",
            path=str(resolved_root),
            exists=True,
            fingerprint={},
            fingerprint_matches_canonical=False,
            restores_clean=None,
            status="FAIL",
            reason=f"data fingerprint check errored: {exc}",
            checks_executed=checks,
            canonical_source=canonical_source,
        )


def _required_checkpoint_specs(*, registry: Registry, repo_root: Path) -> list[dict[str, str]]:
    stages = registry.stage_map()
    s2_minus = stages["S2_MINUS_125M"]
    s2 = stages["S2_125M"]
    s1 = stages["S1_125M"]
    s3 = stages["S3_125M"]

    direct_resume = _resume_exp_name(repo_root, s2_minus.experiment)
    bridge_resume = _resume_exp_name(repo_root, s2.experiment)

    return [
        {
            "artifact_id": "fa_seed_checkpoint",
            "checkpoint_id": "S0_PRETRAIN_FA_125M",
            "exp_name": direct_resume,
            "canonical_source": _experiment_config_path(repo_root, s2_minus.experiment).as_posix(),
        },
        {
            "artifact_id": "s2_bridge_output_checkpoint",
            "checkpoint_id": "S2_ADAPT_125M",
            "exp_name": bridge_resume,
            "canonical_source": _experiment_config_path(repo_root, s2.experiment).as_posix(),
        },
        {
            "artifact_id": "s1_final_checkpoint",
            "checkpoint_id": "S1_125M",
            "exp_name": s1.exp_name,
            "canonical_source": "CANONICAL_RESULTS.md",
        },
        {
            "artifact_id": "s2_final_checkpoint",
            "checkpoint_id": "S2_125M",
            "exp_name": s2.exp_name,
            "canonical_source": "CANONICAL_RESULTS.md; required for current-pipeline S2 re-eval",
        },
        {
            "artifact_id": "s3_final_checkpoint",
            "checkpoint_id": "S3_125M",
            "exp_name": s3.exp_name,
            "canonical_source": "CANONICAL_RESULTS.md",
        },
    ]


def run_e1_preflight(
    *,
    repo_root: Path,
    registry: Registry,
    checkpoint_root: Path,
    exp_folder: str,
    books_root: Path,
    dclm_root: Path,
    canonical_books_root: Path | None,
    canonical_dclm_root: Path | None,
    output_dir: Path,
) -> PreflightResult:
    artifacts: list[PreflightArtifact] = []
    for spec in _required_checkpoint_specs(registry=registry, repo_root=repo_root):
        artifacts.append(
            _checkpoint_artifact(
                artifact_id=spec["artifact_id"],
                checkpoint_root=checkpoint_root.expanduser().resolve(),
                exp_folder=exp_folder,
                checkpoint_id=spec["checkpoint_id"],
                exp_name=spec["exp_name"],
                canonical_source=spec["canonical_source"],
            )
        )

    artifacts.append(
        _data_artifact(
            artifact_id="books32k_data_surface",
            root=books_root,
            canonical_root=canonical_books_root or books_root,
            canonical_source="CANONICAL_RESULTS.md; configs/research/warmstart_registry.yaml: books3",
        )
    )
    artifacts.append(
        _data_artifact(
            artifact_id="dclm8k_data_surface",
            root=dclm_root,
            canonical_root=canonical_dclm_root or dclm_root,
            canonical_source="CANONICAL_RESULTS.md; configs/research/warmstart_registry.yaml: dclm_filter_8k",
        )
    )

    status = "PASS" if artifacts and all(artifact.status == "PASS" for artifact in artifacts) else "FAIL"
    checks_executed_count = sum(len(artifact.checks_executed) for artifact in artifacts)
    required_count = sum(
        len(CHECKS_PER_CHECKPOINT if artifact.kind == "checkpoint" else CHECKS_PER_DATA)
        for artifact in artifacts
    )
    if checks_executed_count < required_count:
        status = "FAIL"

    output_dir = output_dir.expanduser().resolve()
    output_json = output_dir / "preflight_manifest.json"
    output_md = output_dir / "preflight_manifest.md"
    result = PreflightResult(
        schema_version="1.0",
        status=status,
        checks_executed_count=checks_executed_count,
        artifacts=artifacts,
        output_json=str(output_json),
        output_md=str(output_md),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_preflight_markdown(result), encoding="utf-8")
    return result


def render_preflight_markdown(result: PreflightResult) -> str:
    lines = [
        "# E1 Preflight Manifest",
        "",
        f"PREFLIGHT: {result.status} ({result.checks_executed_count} checks executed)",
        "",
        "| artifact | kind | exists | fingerprint | matches canonical | restores clean | status | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in result.artifacts:
        if isinstance(artifact.fingerprint, dict):
            fp = json.dumps(artifact.fingerprint, sort_keys=True)
        else:
            fp = str(artifact.fingerprint)
        if len(fp) > 96:
            fp = fp[:93] + "..."
        restores = "" if artifact.restores_clean is None else str(artifact.restores_clean)
        reason = artifact.reason.replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    artifact.artifact_id,
                    artifact.kind,
                    str(artifact.exists),
                    f"`{fp}`",
                    str(artifact.fingerprint_matches_canonical),
                    restores,
                    artifact.status,
                    reason,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(f"PREFLIGHT: {result.status} ({result.checks_executed_count} checks executed)")
    lines.append("")
    return "\n".join(lines)


def assert_preflight_pass(result: PreflightResult) -> None:
    if result.status != "PASS":
        failures = [
            f"{artifact.artifact_id}: {artifact.reason or artifact.status}"
            for artifact in result.artifacts
            if artifact.status != "PASS"
        ]
        raise RuntimeError("E1 preflight failed:\n" + "\n".join(failures))
