#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _print(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key.strip()] = value
    return env


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    _print("+ " + shlex.join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True, env=env)


def run_capture(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    return run(cmd, capture_output=True, env=env).stdout


def ssh_cmd(identity_file: Path, port: int, user_host: str, remote_cmd: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        str(identity_file),
        "-p",
        str(port),
        user_host,
        remote_cmd,
    ]


def rsync_cmd(identity_file: Path, port: int, src: str, dest: str, *, delete: bool = False) -> list[str]:
    cmd = [
        "rsync",
        "-az",
    ]
    if delete:
        cmd.append("--delete")
    cmd.extend(
        [
            "-e",
            f"ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i {shlex.quote(str(identity_file))} -p {port}",
            src,
            dest,
        ]
    )
    return cmd


@dataclass
class RemoteStatus:
    alive: bool
    stage: str
    process_lines: list[str]
    latest_metrics: dict[str, object]
    gpu_lines: list[str]


def fetch_remote_status(
    *,
    identity_file: Path,
    port: int,
    user_host: str,
    paper_run_id: str,
    remote_repo_root: str,
) -> RemoteStatus:
    remote_cmd = f"""python - <<'PY'
import json
import pathlib
import subprocess

repo = pathlib.Path({remote_repo_root!r})
paper_run_id = {paper_run_id!r}
process = subprocess.run(
    "ps -eo pid,etime,pcpu,pmem,cmd | grep -E 'bin/train \\\\+deploy=interactive|35_run_125m_ladder.py|34_eval_matrix_jax.py|23_warmstart_registry.py' | grep -v grep",
    shell=True,
    capture_output=True,
    text=True,
)
process_lines = [line for line in process.stdout.splitlines() if line.strip()]
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
gpu_lines = [line for line in gpu.stdout.splitlines() if line.strip()]
exp_root = repo / "experiments" / paper_run_id
latest_metrics = {{}}
stage = ""
if exp_root.exists():
    stage_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for stage_dir in stage_dirs:
        run_dirs = sorted([p for p in stage_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        for run_dir in run_dirs:
            metrics_path = run_dir / "metrics.jsonl"
            if metrics_path.exists():
                lines = [line for line in metrics_path.read_text().splitlines() if line.strip()]
                if lines:
                    latest_metrics = json.loads(lines[-1])
                    stage = stage_dir.name
payload = {{
    "alive": bool(process_lines),
    "stage": stage,
    "process_lines": process_lines,
    "gpu_lines": gpu_lines,
    "latest_metrics": latest_metrics,
}}
print(json.dumps(payload))
PY"""
    output = run_capture(ssh_cmd(identity_file, port, user_host, remote_cmd))
    payload = json.loads(output)
    return RemoteStatus(
        alive=bool(payload.get("alive")),
        stage=str(payload.get("stage", "")),
        process_lines=[str(x) for x in payload.get("process_lines", [])],
        latest_metrics=dict(payload.get("latest_metrics", {})),
        gpu_lines=[str(x) for x in payload.get("gpu_lines", [])],
    )


def sync_remote_tree(
    *,
    identity_file: Path,
    port: int,
    user_host: str,
    remote_path: str,
    local_path: Path,
    delete: bool = False,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = rsync_cmd(identity_file, port, f"{user_host}:{remote_path}", str(local_path), delete=delete)
    proc = run(cmd, check=False, capture_output=True)
    if proc.returncode in {0}:
        return
    stderr = proc.stderr or ""
    stdout = proc.stdout or ""
    missing_markers = (
        "No such file or directory",
        "link_stat",
        "change_dir",
        "No such file",
    )
    if proc.returncode in {23, 24} and any(marker in stderr or marker in stdout for marker in missing_markers):
        _print(f"Skipping missing remote path during sync: {remote_path}")
        return
    transient_markers = (
        "Operation timed out",
        "Broken pipe",
        "unexpected end of file",
        "Connection reset by peer",
    )
    if proc.returncode in {255, 12, 30, 35} and any(marker in stderr or marker in stdout for marker in transient_markers):
        _print(f"Transient sync failure for {remote_path}; will retry next poll.")
        return
    raise RuntimeError(
        f"rsync failed for {remote_path} -> {local_path} "
        f"(code={proc.returncode})\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )


def pod_status(pod_id: str) -> dict:
    return json.loads(run_capture(["prime", "pods", "status", pod_id, "--output", "json"]))


def terminate_pod(pod_id: str) -> bool:
    proc = run(["prime", "pods", "terminate", pod_id, "--yes"], check=False, capture_output=True)
    if proc.returncode == 0:
        return True
    _print(f"pod terminate failed: {proc.stdout}\n{proc.stderr}")
    return False


def upload_to_hf(
    *,
    repo_id: str,
    local_root: Path,
    subdir: str,
    private: bool,
    token: str | None,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    target_root = local_root / subdir
    if not target_root.exists():
        raise FileNotFoundError(f"Missing local upload root: {target_root}")
    _print(f"Uploading {target_root} -> {repo_id}")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=target_root,
        private=private,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Babysit the live 125M ladder on a Prime pod.")
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--ssh-host", required=True, help="Remote SSH target like root@ip")
    parser.add_argument("--ssh-port", required=True, type=int)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--remote-repo-root", default="/root/Reproducing-TTT-E2E")
    parser.add_argument("--local-artifact-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--hf-repo-id", default="")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--skip-hf-upload", action="store_true")
    parser.add_argument("--terminate-on-finish", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_values = load_env_file(REPO_ROOT / ".env")
    hf_repo_id = args.hf_repo_id.strip() or env_values.get("HF_RESULTS_REPO", "").strip()
    hf_token = args.hf_token.strip() or env_values.get("HF_TOKEN", "").strip() or None

    local_root = args.local_artifact_root.expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    local_live_root = local_root / args.paper_run_id
    local_live_root.mkdir(parents=True, exist_ok=True)
    summary_path = local_live_root / "babysit_summary.json"
    history_path = local_live_root / "babysit_history.jsonl"

    _print(f"Babysitting paper run {args.paper_run_id} on pod {args.pod_id}")
    completed = False
    final_stage = ""
    final_metrics: dict[str, object] = {}

    while True:
        status = fetch_remote_status(
            identity_file=args.identity_file,
            port=args.ssh_port,
            user_host=args.ssh_host,
            paper_run_id=args.paper_run_id,
            remote_repo_root=args.remote_repo_root,
        )
        pod = pod_status(args.pod_id)
        snapshot = {
            "timestamp": utc_now(),
            "pod_status": pod.get("status"),
            "run_alive": status.alive,
            "stage": status.stage,
            "latest_metrics": status.latest_metrics,
            "gpu_lines": status.gpu_lines,
            "process_lines": status.process_lines,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        _print(
            f"pod={pod.get('status')} alive={status.alive} stage={status.stage or '-'} "
            f"step={status.latest_metrics.get('step', '-') if status.latest_metrics else '-'} "
            f"loss_ce={status.latest_metrics.get('loss_ce', '-') if status.latest_metrics else '-'}"
        )

        # Incremental sync for durability.
        sync_remote_tree(
            identity_file=args.identity_file,
            port=args.ssh_port,
            user_host=args.ssh_host,
            remote_path=f"/root/runlogs/{args.paper_run_id}.log",
            local_path=local_live_root / f"{args.paper_run_id}.log",
        )
        sync_remote_tree(
            identity_file=args.identity_file,
            port=args.ssh_port,
            user_host=args.ssh_host,
            remote_path=f"{args.remote_repo_root}/experiments/{args.paper_run_id}/",
            local_path=local_live_root / "experiments",
            delete=True,
        )
        sync_remote_tree(
            identity_file=args.identity_file,
            port=args.ssh_port,
            user_host=args.ssh_host,
            remote_path=f"{args.remote_repo_root}/reports/paper/{args.paper_run_id}/",
            local_path=local_live_root / "reports",
            delete=True,
        )
        sync_remote_tree(
            identity_file=args.identity_file,
            port=args.ssh_port,
            user_host=args.ssh_host,
            remote_path=f"{args.remote_repo_root}/checkpoints/{args.paper_run_id}/",
            local_path=local_live_root / "checkpoints",
            delete=True,
        )

        if not status.alive:
            completed = True
            final_stage = status.stage
            final_metrics = status.latest_metrics
            break
        time.sleep(max(30, args.poll_seconds))

    # Final comprehensive sync.
    for remote_path, local_path in [
        (f"/root/runlogs/{args.paper_run_id}.log", local_live_root / f"{args.paper_run_id}.log"),
        (f"{args.remote_repo_root}/experiments/{args.paper_run_id}/", local_live_root / "experiments"),
        (f"{args.remote_repo_root}/checkpoints/{args.paper_run_id}/", local_live_root / "checkpoints"),
        (f"{args.remote_repo_root}/reports/paper/{args.paper_run_id}/", local_live_root / "reports"),
    ]:
        sync_remote_tree(
            identity_file=args.identity_file,
            port=args.ssh_port,
            user_host=args.ssh_host,
            remote_path=remote_path,
            local_path=local_path,
            delete=True,
        )

    if completed and not args.skip_hf_upload:
        if not hf_repo_id:
            raise RuntimeError("HF_RESULTS_REPO is not configured and --hf-repo-id was not provided.")
        upload_to_hf(
            repo_id=hf_repo_id,
            local_root=local_root,
            subdir=args.paper_run_id,
            private=bool(args.hf_private),
            token=hf_token,
        )

    terminated = False
    if args.terminate_on_finish:
        terminated = terminate_pod(args.pod_id)

    write_json(
        summary_path,
        {
            "paper_run_id": args.paper_run_id,
            "pod_id": args.pod_id,
            "completed": completed,
            "final_stage": final_stage,
            "final_metrics": final_metrics,
            "hf_repo_id": hf_repo_id,
            "terminated": terminated,
            "finished_at": utc_now(),
        },
    )
    _print(f"Wrote babysit summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
