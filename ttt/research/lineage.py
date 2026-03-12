"""Lineage validation and checkpoint/profile resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .types import CheckpointRef, StageSpec



def file_sha256(path: str | Path) -> str:
    target = Path(path)
    if target.is_dir():
        return directory_sha256(target)
    h = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_sha256(path: str | Path) -> str:
    root = Path(path)
    h = hashlib.sha256()
    for child in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = child.relative_to(root).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        with child.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()



def _load_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw



def resolve_checkpoint_ref(
    *,
    checkpoint_root: Path,
    exp_folder: str,
    checkpoint_id: str,
    exp_name: str | None = None,
    allow_missing: bool = False,
) -> CheckpointRef:
    name = exp_name or checkpoint_id
    latest_path = checkpoint_root / exp_folder / name / "latest.json"
    if not latest_path.exists():
        if allow_missing:
            return CheckpointRef(
                checkpoint_id=checkpoint_id,
                exp_folder=exp_folder,
                exp_name=name,
            )
        raise FileNotFoundError(
            f"Missing required parent checkpoint latest pointer: {latest_path}"
        )
    latest = _load_json(latest_path)
    step = latest.get("step")
    rel_path = latest.get("path")
    if not isinstance(step, int) or not isinstance(rel_path, str):
        raise ValueError(f"Invalid latest checkpoint record: {latest_path}")

    ckpt_path = (latest_path.parent / rel_path).resolve()
    if not ckpt_path.exists():
        if allow_missing:
            return CheckpointRef(
                checkpoint_id=checkpoint_id,
                exp_folder=exp_folder,
                exp_name=name,
                step=step,
            )
        raise FileNotFoundError(f"Missing checkpoint payload file: {ckpt_path}")

    payload_sha = file_sha256(ckpt_path)
    return CheckpointRef(
        checkpoint_id=checkpoint_id,
        exp_folder=exp_folder,
        exp_name=name,
        step=step,
        checkpoint_path=str(ckpt_path),
        payload_sha256=payload_sha,
    )



def resolve_stage_parents(
    *,
    stage: StageSpec,
    stage_map: dict[str, StageSpec],
    checkpoint_root: Path,
    exp_folder: str,
    allow_missing: bool = False,
) -> list[CheckpointRef]:
    refs: list[CheckpointRef] = []
    for parent_id in stage.required_parent_checkpoint_ids:
        parent_stage = stage_map.get(parent_id)
        exp_name = parent_stage.exp_name if parent_stage is not None else parent_id
        refs.append(
            resolve_checkpoint_ref(
                checkpoint_root=checkpoint_root,
                exp_folder=exp_folder,
                checkpoint_id=parent_id,
                exp_name=exp_name,
                allow_missing=allow_missing,
            )
        )
    return refs



def validate_stage_profiles(
    *,
    stage: StageSpec,
    profile_root: Path,
) -> list[str]:
    missing: list[str] = []
    for profile_key in stage.required_profile_keys:
        profile_path = profile_root / profile_key / "model_profile.json"
        if not profile_path.exists():
            missing.append(str(profile_path))
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "Missing required external model profile files:\n" + joined
        )
    return missing



def build_checkpoint_manifest(
    *,
    run_checkpoint: CheckpointRef,
    parent_checkpoints: list[CheckpointRef],
) -> dict:
    return {
        "schema_version": run_checkpoint.schema_version,
        "run_checkpoint": run_checkpoint.to_dict(),
        "parent_checkpoints": [p.to_dict() for p in parent_checkpoints],
    }
