#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def ssh_cmd(identity_file: Path, port: int, user_host: str, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=10",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        str(identity_file),
        "-p",
        str(port),
        user_host,
        remote_command,
    ]


def remote_json(identity_file: Path, port: int, user_host: str, remote_python: str) -> dict:
    proc = run(
        ssh_cmd(
            identity_file,
            port,
            user_host,
            f"python - <<'PY'\n{remote_python}\nPY",
        ),
        capture_output=True,
    )
    return json.loads(proc.stdout)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def remote_status(identity_file: Path, port: int, user_host: str, paper_run_id: str, remote_repo_root: str) -> dict:
    remote_py = f"""
import json
import pathlib
import subprocess

repo = pathlib.Path({remote_repo_root!r})
paper_run_id = {paper_run_id!r}
run_dir = repo / "experiments" / paper_run_id / "S0_PRETRAIN_FA_125M" / "pretrain-125m-fa"
ckpt_dir = repo / "checkpoints" / paper_run_id / "pretrain-125m-fa"

metrics = []
metrics_path = run_dir / "metrics.jsonl"
if metrics_path.exists():
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
latest = metrics[-1] if metrics else {{}}

latest_step = None
latest_json = ckpt_dir / "latest.json"
if latest_json.exists():
    latest_step = json.loads(latest_json.read_text()).get("step")

run_result = None
run_result_path = run_dir / "run_result.json"
if run_result_path.exists():
    run_result = json.loads(run_result_path.read_text())

proc = subprocess.run(
    "ps -eo pid,etime,pcpu,pmem,args | grep -E 'pretrain-125m-fa' | grep -v grep",
    shell=True,
    capture_output=True,
    text=True,
)

payload = {{
    "latest_metrics": latest,
    "latest_checkpoint_step": latest_step,
    "has_run_result": run_result is not None,
    "run_result": run_result,
    "active_process_lines": [line for line in proc.stdout.splitlines() if line.strip()],
}}
print(json.dumps(payload))
"""
    return remote_json(identity_file, port, user_host, remote_py)


def sync_experiment(identity_file: Path, port: int, user_host: str, remote_repo_root: str, paper_run_id: str, local_run_root: Path) -> None:
    local_run_root.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i {identity_file} -p {port}",
            f"{user_host}:{remote_repo_root}/experiments/{paper_run_id}/",
            str(local_run_root / "experiments"),
        ],
        check=False,
    )


def remote_run(identity_file: Path, port: int, user_host: str, cmd: str) -> None:
    run(ssh_cmd(identity_file, port, user_host, cmd))


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor only the 125M FA pretrain stage, then eval/export/terminate.")
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--remote-repo-root", default="/root/Reproducing-TTT-E2E")
    parser.add_argument("--local-run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--eval-batches", type=int, default=8)
    args = parser.parse_args()

    env = load_env_file(REPO_ROOT / ".env")
    hf_repo = env.get("HF_RESULTS_REPO", "").strip()
    hf_token = env.get("HF_TOKEN", "").strip()
    if not hf_repo:
        raise SystemExit("Missing HF_RESULTS_REPO in .env")

    while True:
        status = remote_status(args.identity_file, args.ssh_port, args.ssh_host, args.paper_run_id, args.remote_repo_root)
        latest = status.get("latest_metrics", {})
        latest_step = int(latest.get("step", -1)) if latest else -1
        latest_loss = latest.get("loss_ce", "-")
        latest_ckpt = status.get("latest_checkpoint_step")
        active = status.get("active_process_lines", [])
        print(
            json.dumps(
                {
                    "latest_step": latest_step,
                    "loss_ce": latest_loss,
                    "latest_checkpoint_step": latest_ckpt,
                    "active_processes": len(active),
                }
            ),
            flush=True,
        )
        sync_experiment(args.identity_file, args.ssh_port, args.ssh_host, args.remote_repo_root, args.paper_run_id, args.local_run_root)

        run_result = status.get("run_result") if status.get("has_run_result") else None
        if run_result and str(run_result.get("status")) == "succeeded":
            break
        time.sleep(max(30, args.poll_seconds))

    eval_cmd = f"""bash -lc '
set -euo pipefail
cd {shlex.quote(args.remote_repo_root)}
. .venv/bin/activate
set -a
source .env.runtime
set +a
uv run --exact python scripts/34_eval_matrix_jax.py \
  --paper-run-id {shlex.quote(args.paper_run_id)} \
  --exp-dir {shlex.quote(args.remote_repo_root + "/experiments")} \
  --checkpoint-root {shlex.quote(args.remote_repo_root + "/checkpoints")} \
  --exp-folder {shlex.quote(args.paper_run_id)} \
  --stages S0_PRETRAIN_FA_125M \
  --runs pretrain-125m-fa \
  --contexts 8192 \
  --datasets dclm_filter_8k \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3 \
  --eval-split val \
  --eval-batches {int(args.eval_batches)}
'"""
    remote_run(args.identity_file, args.ssh_port, args.ssh_host, eval_cmd)

    export_cmd = f"""bash -lc '
set -euo pipefail
cd {shlex.quote(args.remote_repo_root)}
. .venv/bin/activate
set -a
source .env.runtime
set +a
uv run --exact python scripts/40_export_stage_to_hf.py \
  --paper-run-id {shlex.quote(args.paper_run_id)} \
  --stage-id S0_PRETRAIN_FA_125M \
  --run-id pretrain-125m-fa \
  --repo-id {shlex.quote(hf_repo)} \
  --token {shlex.quote(hf_token)}
'"""
    remote_run(args.identity_file, args.ssh_port, args.ssh_host, export_cmd)

    sync_experiment(args.identity_file, args.ssh_port, args.ssh_host, args.remote_repo_root, args.paper_run_id, args.local_run_root)
    run(["prime", "pods", "terminate", args.pod_id, "--yes"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
