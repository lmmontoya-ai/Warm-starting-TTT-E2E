#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key.strip()] = value
    return env


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    log("+ " + shlex.join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)


def run_capture(cmd: list[str]) -> str:
    return run(cmd, capture_output=True).stdout


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
    cmd = ["rsync", "-az"]
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


def sync_tree(identity_file: Path, port: int, user_host: str, remote_path: str, local_path: Path, *, delete: bool = False) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    proc = run(rsync_cmd(identity_file, port, f"{user_host}:{remote_path}", str(local_path), delete=delete), check=False, capture_output=True)
    if proc.returncode == 0:
        return
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode in {23, 24, 255, 12, 30, 35}:
        for marker in [
            "No such file",
            "No such file or directory",
            "Operation timed out",
            "Broken pipe",
            "unexpected end of file",
            "Connection reset by peer",
            "link_stat",
            "change_dir",
        ]:
            if marker in text:
                log(f"Non-fatal sync issue for {remote_path}: {marker}")
                return
    raise RuntimeError(f"rsync failed for {remote_path}: {text}")


def prime_status(pod_id: str) -> dict:
    return json.loads(run_capture(["prime", "pods", "status", pod_id, "--output", "json"]))


def remote_json(identity_file: Path, port: int, user_host: str, remote_py: str) -> dict:
    payload = run_capture(ssh_cmd(identity_file, port, user_host, f"python - <<'PY'\n{remote_py}\nPY"))
    return json.loads(payload)


def remote_status(identity_file: Path, port: int, user_host: str, paper_run_id: str, remote_repo_root: str) -> dict:
    remote_py = f"""
import json
import pathlib
import subprocess

repo = pathlib.Path({remote_repo_root!r})
paper_run_id = {paper_run_id!r}
exp_root = repo / "experiments" / paper_run_id
ckpt_root = repo / "checkpoints" / paper_run_id / "pretrain-125m-fa"
all_ckpt_root = repo / "checkpoints" / paper_run_id
launch_summary = repo / "reports" / "paper" / paper_run_id / "launch" / "launcher_summary.json"

def latest_metrics():
    latest = {{}}
    latest_stage = ""
    if exp_root.exists():
        for stage_dir in sorted([p for p in exp_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            for run_dir in sorted([p for p in stage_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                metrics = run_dir / "metrics.jsonl"
                if metrics.exists():
                    lines = [line for line in metrics.read_text().splitlines() if line.strip()]
                    if lines:
                        latest = json.loads(lines[-1])
                        latest_stage = stage_dir.name
    return latest_stage, latest

proc = subprocess.run(
    "ps -eo pid,etime,pcpu,pmem,args | grep -E 'pretrain-125m-fa|35_run_125m_ladder.py|23_warmstart_registry.py|34_eval_matrix_jax.py' | grep -v grep",
    shell=True, capture_output=True, text=True
)
process_lines = [line for line in proc.stdout.splitlines() if line.strip()]
latest_stage, latest = latest_metrics()
latest_ckpt = None
latest_json = ckpt_root / "latest.json"
if latest_json.exists():
    latest_ckpt = json.loads(latest_json.read_text()).get("step")
checkpoint_latest_by_run = {{}}
if all_ckpt_root.exists():
    for run_dir in sorted([p for p in all_ckpt_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        run_latest_json = run_dir / "latest.json"
        if not run_latest_json.exists():
            continue
        try:
            checkpoint_latest_by_run[run_dir.name] = int(json.loads(run_latest_json.read_text()).get("step"))
        except Exception:
            continue
payload = {{
    "process_lines": process_lines,
    "latest_stage": latest_stage,
    "latest_metrics": latest,
    "latest_checkpoint_step": latest_ckpt,
    "checkpoint_latest_by_run": checkpoint_latest_by_run,
    "launcher_summary_exists": launch_summary.exists(),
}}
if launch_summary.exists():
    payload["launcher_summary"] = json.loads(launch_summary.read_text())
print(json.dumps(payload))
"""
    return remote_json(identity_file, port, user_host, remote_py)


def remote_launch_resume_pretrain(
    identity_file: Path,
    port: int,
    user_host: str,
    *,
    paper_run_id: str,
    remote_repo_root: str,
) -> None:
    cmd = f"""bash -s <<'REMOTE'
set -euo pipefail
cd {shlex.quote(remote_repo_root)}
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
. .venv/bin/activate
set -a
source .env.runtime
set +a
mkdir -p /root/runlogs
RESUME_ARGS=""
if [ -f {shlex.quote(remote_repo_root + "/checkpoints/" + paper_run_id + "/pretrain-125m-fa/latest.json")} ]; then
  RESUME_ARGS="training.resume_exp_name=pretrain-125m-fa training.load_part=all"
fi
nohup uv run --exact train \\
  +deploy=interactive \\
  +experiment=125m/pretrain/pretrain-125m-fa \\
  training.exp_folder={shlex.quote(paper_run_id)} \\
  training.exp_dir={shlex.quote(remote_repo_root + "/experiments")} \\
  training.exp_name=pretrain-125m-fa \\
  training.total_steps=4800 \\
  training.runtime_mode=jax_train \\
  training.wandb_entity=luism31 \\
  training.wandb_project=ttt-e2e-warmstart \\
  training.wandb_key=env \\
  deploy_paths.data.dclm_filter_8k=/root/ttt-e2e-data/dclm_filter_8k \\
  deploy_paths.data.books3=/root/ttt-e2e-data/books3 \\
  deploy_paths.checkpoint={shlex.quote(remote_repo_root + "/checkpoints")} \\
  training.checkpoint_path={shlex.quote(remote_repo_root + "/checkpoints")} \\
  training.paper_run_id={shlex.quote(paper_run_id)} \\
  training.stage_id=S0_PRETRAIN_FA_125M \\
  training.run_id=pretrain-125m-fa \\
  training.save_milestone_freq=120 \\
  $RESUME_ARGS \\
  > /root/runlogs/{shlex.quote(paper_run_id)}_resume_s0.log 2>&1 < /dev/null &
echo $!
REMOTE"""
    run(ssh_cmd(identity_file, port, user_host, cmd))


def remote_launch_ladder(
    identity_file: Path,
    port: int,
    user_host: str,
    *,
    paper_run_id: str,
    remote_repo_root: str,
    protocol: str,
    ext_global_batch_size: int | None,
    preserve_ext_token_budget: bool,
    base_ext_global_batch_size: int,
) -> None:
    extra_args: list[str] = ["--protocol", protocol]
    if ext_global_batch_size is not None:
        extra_args.extend(["--ext-global-batch-size", str(ext_global_batch_size)])
    if preserve_ext_token_budget:
        extra_args.append("--preserve-ext-token-budget")
    extra_args.extend(["--base-ext-global-batch-size", str(base_ext_global_batch_size)])
    extra_arg_block = " \\\n  ".join(shlex.quote(x) for x in extra_args)
    cmd = f"""bash -s <<'REMOTE'
set -euo pipefail
cd {shlex.quote(remote_repo_root)}
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
. .venv/bin/activate
set -a
source .env.runtime
set +a
mkdir -p /root/runlogs
nohup uv run --exact python scripts/35_run_125m_ladder.py \\
  --paper-run-id {shlex.quote(paper_run_id)} \\
  --exp-folder {shlex.quote(paper_run_id)} \\
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \\
  --books-root /root/ttt-e2e-data/books3 \\
  --skip-existing \\
  {extra_arg_block} \\
  > /root/runlogs/{shlex.quote(paper_run_id)}_resume_wrapper.log 2>&1 < /dev/null &
echo $!
REMOTE"""
    run(ssh_cmd(identity_file, port, user_host, cmd))


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_checkpoint_runs(checkpoint_root: Path) -> Iterable[tuple[str, Path, int]]:
    if not checkpoint_root.exists():
        return []
    rows: list[tuple[str, Path, int]] = []
    for run_dir in sorted([p for p in checkpoint_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        latest_json = run_dir / "latest.json"
        if not latest_json.exists():
            continue
        try:
            step = int(json.loads(latest_json.read_text(encoding="utf-8")).get("step") or 0)
        except Exception:
            continue
        rows.append((run_dir.name, run_dir, step))
    return rows


def _iter_experiment_runs(experiment_root: Path) -> Iterable[tuple[str, str, Path]]:
    if not experiment_root.exists():
        return []
    rows: list[tuple[str, str, Path]] = []
    for stage_dir in sorted([p for p in experiment_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for run_dir in sorted([p for p in stage_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            rows.append((stage_dir.name, run_dir.name, run_dir))
    return rows


def _checkpoint_metadata_name(step: int) -> str:
    return f"step_metadata_{step:08d}.json"


def sync_latest_checkpoint_snapshots(
    identity_file: Path,
    port: int,
    user_host: str,
    remote_repo_root: str,
    paper_run_id: str,
    local_run_root: Path,
    latest_by_run: dict[str, int],
) -> None:
    local_checkpoint_root = local_run_root / "checkpoints"
    for run_name, step in sorted(latest_by_run.items()):
        if int(step) < 0:
            continue
        step = int(step)
        local_run_dir = local_checkpoint_root / run_name
        local_run_dir.mkdir(parents=True, exist_ok=True)
        remote_run_dir = f"{remote_repo_root}/checkpoints/{paper_run_id}/{run_name}"

        sync_tree(identity_file, port, user_host, f"{remote_run_dir}/latest.json", local_run_dir / "latest.json")
        metadata_name = _checkpoint_metadata_name(step)
        sync_tree(identity_file, port, user_host, f"{remote_run_dir}/{metadata_name}", local_run_dir / metadata_name)
        # Do not mirror heavy checkpoint payloads back to the local machine.
        # Durability for full checkpoint contents is handled on the pod and via
        # stage-level HF exports; locally we keep only lightweight pointers.
        keep_files = {
            "latest.json",
            metadata_name,
        }
        for stale in local_run_dir.iterdir():
            if stale.name in keep_files:
                continue
            if stale.is_dir():
                subprocess.run(["rm", "-rf", str(stale)], check=False)
            elif stale.is_file() and stale.name.startswith("step_metadata_"):
                stale.unlink(missing_ok=True)


def _iter_successful_local_stage_runs(experiment_root: Path) -> Iterable[tuple[str, str]]:
    for stage_name, run_name, run_dir in _iter_experiment_runs(experiment_root):
        payload = _load_json_file(run_dir / "run_result.json")
        status = str(payload.get("status", ""))
        if status in {"succeeded", "dry_run"}:
            yield stage_name, run_name


def remote_export_stage_to_hf(
    identity_file: Path,
    port: int,
    user_host: str,
    *,
    paper_run_id: str,
    stage_id: str,
    run_id: str,
    repo_id: str,
    token: str | None,
) -> None:
    token_value = token or ""
    cmd = f"""bash -s <<'REMOTE'
set -euo pipefail
cd /root/Reproducing-TTT-E2E
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
. .venv/bin/activate
uv run --exact python scripts/40_export_stage_to_hf.py \\
  --paper-run-id {shlex.quote(paper_run_id)} \\
  --stage-id {shlex.quote(stage_id)} \\
  --run-id {shlex.quote(run_id)} \\
  --repo-id {shlex.quote(repo_id)} \\
  --token {shlex.quote(token_value)}
REMOTE"""
    run(ssh_cmd(identity_file, port, user_host, cmd))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous controller for the 125M Prime ladder.")
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--paper-run-id", default="prime_125m_ladder_20260312a")
    parser.add_argument("--remote-repo-root", default="/root/Reproducing-TTT-E2E")
    parser.add_argument("--local-run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--initial-launch", choices=["wrapper", "pretrain"], default="wrapper")
    parser.add_argument("--protocol", choices=["faithful", "revised"], default="faithful")
    parser.add_argument("--ext-global-batch-size", type=int, default=None)
    parser.add_argument("--preserve-ext-token-budget", action="store_true")
    parser.add_argument("--base-ext-global-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_values = load_env_file(REPO_ROOT / ".env")
    hf_repo = env_values.get("HF_RESULTS_REPO", "").strip()
    hf_token = env_values.get("HF_TOKEN", "").strip() or None
    local_run_root = args.local_run_root.expanduser().resolve()
    local_run_root.mkdir(parents=True, exist_ok=True)
    fa_complete_step = 4799
    export_state_path = local_run_root / ".hf_stage_export_state.json"
    export_state = _load_json_file(export_state_path)
    exported_stage_runs = set(export_state.get("exported_stage_runs", []))

    launched_wrapper = False
    launched_anything = False
    while True:
        status = remote_status(args.identity_file, args.ssh_port, args.ssh_host, args.paper_run_id, args.remote_repo_root)
        pod = prime_status(args.pod_id)
        log(
            f"pod={pod.get('status')} active_procs={len(status.get('process_lines', []))} "
            f"stage={status.get('latest_stage','-')} step={status.get('latest_metrics',{}).get('step','-')} "
            f"loss_ce={status.get('latest_metrics',{}).get('loss_ce','-')}"
        )

        # Lightweight sync only every poll.
        sync_tree(args.identity_file, args.ssh_port, args.ssh_host, f"/root/runlogs/{args.paper_run_id}.log", local_run_root / f"{args.paper_run_id}.log")
        sync_tree(args.identity_file, args.ssh_port, args.ssh_host, f"{args.remote_repo_root}/experiments/{args.paper_run_id}/", local_run_root / "experiments", delete=True)
        sync_latest_checkpoint_snapshots(
            args.identity_file,
            args.ssh_port,
            args.ssh_host,
            args.remote_repo_root,
            args.paper_run_id,
            local_run_root,
            status.get("checkpoint_latest_by_run", {}),
        )
        if hf_repo:
            for stage_name, run_name in _iter_successful_local_stage_runs(local_run_root / "experiments"):
                key = f"{stage_name}/{run_name}"
                if key in exported_stage_runs:
                    continue
                log(f"Exporting completed stage to HF: {key}")
                remote_export_stage_to_hf(
                    args.identity_file,
                    args.ssh_port,
                    args.ssh_host,
                    paper_run_id=args.paper_run_id,
                    stage_id=stage_name,
                    run_id=run_name,
                    repo_id=hf_repo,
                    token=hf_token,
                )
                exported_stage_runs.add(key)
                _write_json_file(export_state_path, {"exported_stage_runs": sorted(exported_stage_runs)})

        if status.get("launcher_summary_exists"):
            summary = status.get("launcher_summary", {})
            rows = summary.get("rows", [])
            if rows and all(int(row.get("returncode", 1)) == 0 for row in rows):
                sync_tree(args.identity_file, args.ssh_port, args.ssh_host, f"{args.remote_repo_root}/reports/paper/{args.paper_run_id}/", local_run_root / "reports", delete=True)
                run(["prime", "pods", "terminate", args.pod_id, "--yes"], check=False)
                log("Controller finished successfully.")
                return 0

        process_lines = status.get("process_lines", [])
        latest_ckpt = int(status.get("latest_checkpoint_step") or 0)
        latest_step = int(status.get("latest_metrics", {}).get("step") or -1)
        fa_complete = latest_ckpt >= fa_complete_step or latest_step >= fa_complete_step

        if process_lines:
            time.sleep(max(30, args.poll_seconds))
            continue

        if not fa_complete and not launched_anything:
            if args.initial_launch == "wrapper":
                log("No active process; launching full ladder wrapper from scratch.")
                remote_launch_ladder(
                    args.identity_file,
                    args.ssh_port,
                    args.ssh_host,
                    paper_run_id=args.paper_run_id,
                    remote_repo_root=args.remote_repo_root,
                    protocol=args.protocol,
                    ext_global_batch_size=args.ext_global_batch_size,
                    preserve_ext_token_budget=args.preserve_ext_token_budget,
                    base_ext_global_batch_size=args.base_ext_global_batch_size,
                )
                launched_wrapper = True
            else:
                log("No active process; launching pretrain stage directly.")
                remote_launch_resume_pretrain(
                    args.identity_file,
                    args.ssh_port,
                    args.ssh_host,
                    paper_run_id=args.paper_run_id,
                    remote_repo_root=args.remote_repo_root,
                )
            launched_anything = True
            time.sleep(20)
            continue

        if fa_complete and not launched_wrapper:
            log("FA pretrain complete; launching skip-existing ladder wrapper.")
            remote_launch_ladder(
                args.identity_file,
                args.ssh_port,
                args.ssh_host,
                paper_run_id=args.paper_run_id,
                remote_repo_root=args.remote_repo_root,
                protocol=args.protocol,
                ext_global_batch_size=args.ext_global_batch_size,
                preserve_ext_token_budget=args.preserve_ext_token_budget,
                base_ext_global_batch_size=args.base_ext_global_batch_size,
            )
            launched_wrapper = True
            launched_anything = True
            time.sleep(20)
            continue

        if not fa_complete:
            log("Detected missing active process before FA pretrain completion; relaunching resume.")
            remote_launch_resume_pretrain(
                args.identity_file,
                args.ssh_port,
                args.ssh_host,
                paper_run_id=args.paper_run_id,
                remote_repo_root=args.remote_repo_root,
            )
            time.sleep(20)
            continue

        time.sleep(max(30, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
