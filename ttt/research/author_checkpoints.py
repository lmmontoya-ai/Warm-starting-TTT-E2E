from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuthorCheckpointSpec:
    key: str
    source_uri: str
    default_step: int
    description: str


AUTHOR_CHECKPOINTS: dict[str, AuthorCheckpointSpec] = {
    "760m_fa": AuthorCheckpointSpec(
        key="760m_fa",
        source_uri="gs://ttt-e2e-checkpoints/760m_full_attn_pretrain_dclm_8k_1x_cc",
        default_step=28999,
        description="Author-shared 760M full-attention pretrain checkpoint on DCLM @ 8K",
    ),
    "760m_e2e": AuthorCheckpointSpec(
        key="760m_e2e",
        source_uri="gs://ttt-e2e-checkpoints/760m_ttt_e2e_pretrain_dclm_8k_1x_cc",
        default_step=28999,
        description="Author-shared 760M TTT-E2E pretrain checkpoint on DCLM @ 8K",
    ),
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def resolve_gcloud() -> list[str]:
    gcloud = shutil.which("gcloud")
    if gcloud:
        return [gcloud, "storage"]
    fallback = Path.home() / "Downloads" / "google-cloud-sdk" / "bin" / "gcloud"
    if fallback.exists():
        return [str(fallback), "storage"]
    raise FileNotFoundError("Could not find 'gcloud' in PATH or under ~/Downloads/google-cloud-sdk/bin.")


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_capture(cmd: list[str]) -> str:
    print("+", shlex.join(cmd))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def gcloud_ls(uri: str, *, billing_project: str | None = None) -> list[str]:
    cmd = resolve_gcloud() + ["ls"]
    if billing_project:
        cmd += ["--billing-project", billing_project]
    cmd.append(uri)
    stdout = run_capture(cmd)
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def gcloud_cp(
    source: str,
    dest: Path,
    *,
    recursive: bool = False,
    billing_project: str | None = None,
    dry_run: bool = False,
) -> None:
    cmd = resolve_gcloud() + ["cp"]
    if recursive:
        cmd.append("-r")
    if billing_project:
        cmd += ["--billing-project", billing_project]
    cmd += [source, str(dest)]
    run(cmd, dry_run=dry_run)


def gcloud_cat(uri: str, *, billing_project: str | None = None) -> str:
    cmd = resolve_gcloud() + ["cat", uri]
    if billing_project:
        cmd += ["--billing-project", billing_project]
    return run_capture(cmd)


def gcloud_du(uri: str, *, billing_project: str | None = None) -> int | None:
    cmd = resolve_gcloud() + ["du", "-s"]
    if billing_project:
        cmd += ["--billing-project", billing_project]
    cmd.append(uri)
    stdout = run_capture(cmd).strip()
    if not stdout:
        return None
    try:
        return int(stdout.split()[0])
    except (IndexError, ValueError):
        return None


def select_specs(raw: str) -> list[AuthorCheckpointSpec]:
    if raw == "all":
        return [AUTHOR_CHECKPOINTS["760m_fa"], AUTHOR_CHECKPOINTS["760m_e2e"]]
    values = [item.strip() for item in raw.split(",") if item.strip()]
    specs: list[AuthorCheckpointSpec] = []
    for value in values:
        if value not in AUTHOR_CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint key: {value}")
        specs.append(AUTHOR_CHECKPOINTS[value])
    return specs


def find_latest_step(spec: AuthorCheckpointSpec, *, billing_project: str | None = None) -> int:
    entries = gcloud_ls(spec.source_uri, billing_project=billing_project)
    step_values: list[int] = []
    for entry in entries:
        candidate = entry.rstrip("/").split("/")[-1]
        try:
            step_values.append(int(candidate))
        except ValueError:
            continue
    if not step_values:
        raise FileNotFoundError(f"No numeric step directories found under {spec.source_uri}")
    return max(step_values)


def artifact_root(base_root: Path, spec: AuthorCheckpointSpec) -> Path:
    return base_root / spec.key


def raw_step_dir(base_root: Path, spec: AuthorCheckpointSpec, step: int) -> Path:
    return artifact_root(base_root, spec) / "raw_orbax" / str(step)


def manifest_path(base_root: Path, spec: AuthorCheckpointSpec) -> Path:
    return artifact_root(base_root, spec) / "artifact_manifest.json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def build_base_manifest(
    *,
    spec: AuthorCheckpointSpec,
    step: int,
    billing_project: str | None,
    remote_bytes: int | None,
    local_step_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "checkpoint_key": spec.key,
        "description": spec.description,
        "source_uri": spec.source_uri,
        "step": int(step),
        "billing_project": billing_project or "",
        "remote_step_uri": f"{spec.source_uri}/{step}",
        "remote_size_bytes": remote_bytes,
        "local_artifact_root": str(local_step_dir.parents[1]),
        "local_raw_step_dir": str(local_step_dir),
        "layout": {
            "raw_orbax_subdir": f"raw_orbax/{step}",
            "manifest_filename": "artifact_manifest.json",
        },
        "local_runtime_compatibility": {
            "status": "not_yet_compatible",
            "reason": (
                "Current local jax_runtime checkpoint and model tree are not architectural "
                "parity with the author Orbax checkpoint. This artifact is preserved as raw "
                "Orbax plus manifests for verified transport and later parity work."
            ),
        },
    }


def spec_to_dict(spec: AuthorCheckpointSpec) -> dict[str, Any]:
    return asdict(spec)
