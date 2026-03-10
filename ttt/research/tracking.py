"""Run tracking and manifest writing utilities."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import EvalResult, RunResult, RunSpec, StageSpec, utc_now_iso


@dataclass(frozen=True)
class GitState:
    commit: str
    branch: str
    dirty: bool



def _run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()



def detect_git_state(cwd: Path) -> GitState:
    commit = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    dirty_raw = _run_git(["status", "--porcelain"], cwd=cwd)
    return GitState(commit=commit, branch=branch, dirty=bool(dirty_raw))



def ensure_run_dir(*, exp_dir: Path, paper_run_id: str, stage_id: str, run_id: str) -> Path:
    run_dir = exp_dir / paper_run_id / stage_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")



def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")



def write_command_script(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join(_shell_quote(part) for part in command)
    content = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + quoted + "\n"
    path.write_text(content)
    os.chmod(path, 0o755)



def _shell_quote(raw: str) -> str:
    if raw == "":
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/:=,+"
    if all(c in safe for c in raw):
        return raw
    return "'" + raw.replace("'", "'\"'\"'") + "'"



def environment_manifest(*, repo_root: Path) -> dict[str, Any]:
    git = detect_git_state(repo_root)
    try:
        nvidia = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        nvidia_smi = nvidia.stdout.strip() if nvidia.returncode == 0 else ""
    except FileNotFoundError:
        nvidia_smi = ""

    return {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "git_commit": git.commit,
        "git_branch": git.branch,
        "git_dirty": git.dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": nvidia_smi,
    }



def write_stage_manifest(path: Path, stage: StageSpec, *, repo_root: Path | None = None) -> None:
    payload = {
        "schema_version": stage.schema_version,
        "created_at_utc": utc_now_iso(),
        "stage": stage.to_dict(),
    }
    if repo_root is not None:
        git = detect_git_state(repo_root)
        payload.update(
            {
                "git_commit": git.commit,
                "git_branch": git.branch,
                "git_dirty": git.dirty,
            }
        )
    write_json(path, payload)



def write_run_manifest(path: Path, run_spec: RunSpec, *, repo_root: Path) -> None:
    git = detect_git_state(repo_root)
    payload = run_spec.to_dict()
    payload.update(
        {
            "created_at_utc": utc_now_iso(),
            "git_commit": git.commit,
            "git_branch": git.branch,
            "git_dirty": git.dirty,
        }
    )
    write_json(path, payload)



def write_run_result(path: Path, run_result: RunResult) -> None:
    write_json(path, run_result.to_dict())



def write_eval_manifest(path: Path, eval_result: EvalResult, *, repo_root: Path) -> None:
    git = detect_git_state(repo_root)
    payload = eval_result.to_dict()
    payload.update(
        {
            "git_commit": git.commit,
            "git_branch": git.branch,
            "git_dirty": git.dirty,
        }
    )
    write_json(path, payload)
