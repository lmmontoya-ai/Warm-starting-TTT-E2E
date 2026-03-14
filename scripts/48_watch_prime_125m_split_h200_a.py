#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "artifacts" / "prime_watch"
CANONICAL_PAPER_RUN_ID = "protocol_r_125m_main_v1"
REMOTE_REPO_ROOT = f"/root/{REPO_ROOT.name}"
REMOTE_DATA_ROOT = "/root/ttt-e2e-data"
MINIMAL_SYNC_PATHS = [
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("ttt"),
    Path("configs"),
    Path("scripts/23_warmstart_registry.py"),
    Path("scripts/40_export_stage_to_hf.py"),
    Path("scripts/46_restore_stage_from_hf.py"),
    Path("scripts/47_run_125m_split_batch.py"),
]
HF_STAGE_PREFIXES = {
    "S0_125M": "ext-125m-fa-32K",
    "S1_125M": "ext-125m-swa-32K-from-fa",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _print(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
    env: dict[str, str] | None = None,
):
    _print("+ " + shlex.join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text, env=env)


def _run_capture(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    return _run(cmd, capture_output=True, env=env).stdout


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key] = value
    return env


def _safe_int(value, fallback: int) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return fallback


def _prime_ssh_key_path() -> Path:
    output = _run_capture(["prime", "config", "view"])
    for line in output.splitlines():
        if "SSH Key Path" not in line:
            continue
        parts = [part.strip() for part in line.split("│") if part.strip()]
        if parts:
            path = Path(parts[-1]).expanduser()
            if path.exists():
                return path
    raise FileNotFoundError("Could not determine Prime SSH key path from `prime config view`.")


def _ssh_base_cmd(ssh_key: Path, *, scp: bool = False, port: str | None = None) -> list[str]:
    cmd = ["scp" if scp else "ssh"]
    cmd.extend(
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=120",
            "-i",
            str(ssh_key),
        ]
    )
    if port:
        cmd.extend(["-P" if scp else "-p", port])
    return cmd


def _rsync_ssh_transport(ssh_key: Path, *, port: str | None = None) -> str:
    parts = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=120",
        "-i",
        shlex.quote(str(ssh_key)),
    ]
    if port:
        parts.extend(["-p", shlex.quote(port)])
    return " ".join(parts)


def _parse_ssh_target(ssh_target: str) -> tuple[str, str | None]:
    ssh_target = ssh_target.strip()
    if " -p " in ssh_target:
        host, port = ssh_target.rsplit(" -p ", 1)
        return host.strip(), port.strip()
    return ssh_target, None


def _remote_shell(ssh_key: Path, ssh_target: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    host, port = _parse_ssh_target(ssh_target)
    cmd = _ssh_base_cmd(ssh_key, scp=False, port=port)
    cmd.extend([host, command])
    return _run(cmd, check=check, capture_output=True)


def _sync_repo(ssh_key: Path, ssh_target: str) -> None:
    host, port = _parse_ssh_target(ssh_target)
    transport = _rsync_ssh_transport(ssh_key, port=port)
    for rel_path in MINIMAL_SYNC_PATHS:
        local_path = REPO_ROOT / rel_path
        if not local_path.exists():
            raise FileNotFoundError(f"Required sync path is missing: {local_path}")
        remote_parent = f"{REMOTE_REPO_ROOT}/{rel_path.parent.as_posix()}" if rel_path.parent != Path(".") else REMOTE_REPO_ROOT
        _remote_shell(
            ssh_key,
            ssh_target,
            f"set -euo pipefail; mkdir -p {shlex.quote(remote_parent)}",
        )
        cmd = ["rsync", "-az", "--delete", "-e", transport]
        if local_path.is_dir():
            remote_dir = f"{host}:{remote_parent}/{local_path.name}/"
            cmd.extend([f"{local_path}/", remote_dir])
        else:
            remote_file = f"{host}:{REMOTE_REPO_ROOT}/{rel_path.as_posix()}"
            cmd.extend([str(local_path), remote_file])
        _run(cmd)


def _write_remote_env(ssh_key: Path, ssh_target: str, env_values: dict[str, str]) -> None:
    host, port = _parse_ssh_target(ssh_target)
    keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "B2_ENDPOINT_URL",
        "B2_BUCKET",
        "B2_DATASET_PREFIX",
        "HF_TOKEN",
        "HF_RESULTS_REPO",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    ]
    lines = []
    for key in keys:
        value = env_values.get(key, "")
        if value:
            lines.append(f"{key}={shlex.quote(value)}")
    payload = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        _run(
            _ssh_base_cmd(ssh_key, scp=True, port=port)
            + [str(temp_path), f"{host}:{REMOTE_REPO_ROOT}/.env.runtime"]
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _bootstrap_remote(ssh_key: Path, ssh_target: str) -> None:
    command = (
        "set -euo pipefail; "
        "sudo apt-get update -y >/dev/null; "
        "sudo apt-get install -y curl rsync git unzip awscli >/dev/null; "
        "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi; "
        "command -v aws >/dev/null 2>&1; aws --version >/dev/null 2>&1; "
        f"mkdir -p {shlex.quote(REMOTE_REPO_ROOT)} {shlex.quote(REMOTE_DATA_ROOT)} /root/runlogs"
    )
    _remote_shell(ssh_key, ssh_target, command)


def _restore_datasets(ssh_key: Path, ssh_target: str) -> None:
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(REMOTE_REPO_ROOT)}; "
        "set -a; source .env.runtime; set +a; "
        f"mkdir -p {shlex.quote(REMOTE_DATA_ROOT + '/dclm_filter_8k')} "
        f"{shlex.quote(REMOTE_DATA_ROOT + '/books3')}; "
        "aws s3 sync "
        "\"s3://$B2_BUCKET/$B2_DATASET_PREFIX/paper_budget_125m_val-full/dclm_filter_8k\" "
        f"{shlex.quote(REMOTE_DATA_ROOT + '/dclm_filter_8k')} "
        "--endpoint-url \"$B2_ENDPOINT_URL\" "
        "--region \"$AWS_DEFAULT_REGION\" "
        "--only-show-errors; "
        "aws s3 sync "
        "\"s3://$B2_BUCKET/$B2_DATASET_PREFIX/paper_budget_125m_val-full/books3\" "
        f"{shlex.quote(REMOTE_DATA_ROOT + '/books3')} "
        "--endpoint-url \"$B2_ENDPOINT_URL\" "
        "--region \"$AWS_DEFAULT_REGION\" "
        "--only-show-errors; "
        f"test -f {shlex.quote(REMOTE_DATA_ROOT + '/dclm_filter_8k/train/zarr.json')}; "
        f"test -f {shlex.quote(REMOTE_DATA_ROOT + '/books3/train/zarr.json')}; "
        f"test -f {shlex.quote(REMOTE_DATA_ROOT + '/dclm_filter_8k/train.fingerprint.json')}; "
        f"test -f {shlex.quote(REMOTE_DATA_ROOT + '/books3/train.fingerprint.json')}"
    )
    _remote_shell(ssh_key, ssh_target, command)


def _load_availability() -> list[dict]:
    payload = json.loads(
        _run_capture(
            [
                "prime",
                "availability",
                "list",
                "--gpu-type",
                "H200_141GB",
                "--gpu-count",
                "8",
                "--no-group-similar",
                "--output",
                "json",
            ]
        )
    )
    rows = payload.get("gpu_resources", [])
    if not isinstance(rows, list):
        raise ValueError("Prime availability payload missing gpu_resources list.")
    return rows


def _select_target(rows: list[dict]) -> dict | None:
    matches: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("is_spot")):
            continue
        try:
            count = int(row.get("gpu_count", 0) or 0)
        except Exception:
            continue
        gpu_type = str(row.get("gpu_type", ""))
        if count != 8:
            continue
        if "H200 141GB" not in gpu_type:
            continue
        matches.append(row)
    matches.sort(key=lambda item: float(item.get("price_value", 1e9) or 1e9))
    return matches[0] if matches else None


def _create_pod(row: dict, *, pod_name: str, disk_size_gb: int, image: str, log_path: Path) -> str:
    vcpus = _safe_int(row.get("vcpus"), 96)
    memory = _safe_int(row.get("memory_gb"), 1024)
    cmd = [
        "prime",
        "pods",
        "create",
        "--id",
        str(row["id"]),
        "--name",
        pod_name,
        "--disk-size",
        str(disk_size_gb),
        "--vcpus",
        str(vcpus),
        "--memory",
        str(memory),
        "--image",
        image,
        "--yes",
    ]
    proc = _run(cmd, capture_output=True)
    created_id = None
    for line in proc.stdout.splitlines():
        if "Successfully created pod" in line:
            created_id = line.rsplit(" ", 1)[-1].strip()
            break
    if not created_id:
        raise RuntimeError(f"Could not parse created pod id from Prime output:\n{proc.stdout}\n{proc.stderr}")
    _append_jsonl(log_path, {"event": "pod_created", "row": row, "pod_id": created_id})
    return created_id


def _pod_status(pod_id: str) -> dict:
    return json.loads(_run_capture(["prime", "pods", "status", pod_id, "--output", "json"]))


def _wait_for_ssh_target(pod_id: str, *, timeout_seconds: int, poll_seconds: int, log_path: Path) -> str | None:
    started = time.time()
    while time.time() - started < timeout_seconds:
        status = _pod_status(pod_id)
        _append_jsonl(
            log_path,
            {
                "event": "pod_status",
                "pod_id": pod_id,
                "status": status.get("status"),
                "ssh": status.get("ssh"),
                "ip": status.get("ip"),
            },
        )
        ssh_target = str(status.get("ssh", "")).strip()
        if ssh_target and ssh_target != "N/A":
            return ssh_target
        time.sleep(max(1, poll_seconds))
    return None


def _terminate_pod(pod_id: str, *, retries: int, sleep_seconds: int, log_path: Path) -> bool:
    for attempt in range(1, retries + 1):
        proc = subprocess.run(
            ["prime", "pods", "terminate", pod_id, "--yes"],
            capture_output=True,
            text=True,
        )
        _append_jsonl(
            log_path,
            {
                "event": "pod_terminate_attempt",
                "pod_id": pod_id,
                "attempt": attempt,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )
        if proc.returncode == 0:
            return True
        time.sleep(max(1, sleep_seconds))
    return False


def _launch_batch(ssh_key: Path, ssh_target: str) -> int:
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(REMOTE_REPO_ROOT)}; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "cat > /root/runlogs/h200_a_wrapper.sh <<'SH'\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(REMOTE_REPO_ROOT)}\n"
        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
        "uv venv --python 3.13 >/dev/null 2>&1\n"
        ". .venv/bin/activate\n"
        "uv sync --exact --python 3.13 >/dev/null 2>&1\n"
        "set -a\n"
        "source .env.runtime\n"
        "set +a\n"
        "uv run --exact python scripts/47_run_125m_split_batch.py "
        "--batch h200_a "
        f"--paper-run-id {CANONICAL_PAPER_RUN_ID} "
        "--repo-id \"$HF_RESULTS_REPO\" "
        "--token \"$HF_TOKEN\" "
        f"--dclm-root {shlex.quote(REMOTE_DATA_ROOT + '/dclm_filter_8k')} "
        f"--books-root {shlex.quote(REMOTE_DATA_ROOT + '/books3')} "
        "--wandb-entity \"$WANDB_ENTITY\" "
        "--wandb-project \"$WANDB_PROJECT\" "
        "--wandb-key env "
        "--skip-existing\n"
        "SH\n"
        "chmod +x /root/runlogs/h200_a_wrapper.sh; "
        "rm -f /root/runlogs/h200_a.exit /root/runlogs/h200_a.pid; "
        "nohup bash -lc '/root/runlogs/h200_a_wrapper.sh > /root/runlogs/h200_a.log 2>&1; "
        "rc=$?; printf \"%s\\n\" \"$rc\" > /root/runlogs/h200_a.exit' "
        "</dev/null >/dev/null 2>&1 & "
        "echo $! > /root/runlogs/h200_a.pid; "
        "cat /root/runlogs/h200_a.pid"
    )
    proc = _remote_shell(ssh_key, ssh_target, command)
    pid_text = proc.stdout.strip().splitlines()[-1].strip()
    return int(pid_text)


def _remote_status_snapshot(ssh_key: Path, ssh_target: str, pid: int) -> dict:
    py = (
        "import json, pathlib\n"
        "def load(path):\n"
        "    p = pathlib.Path(path)\n"
        "    if not p.exists():\n"
        "        return None\n"
        "    try:\n"
        "        return json.loads(p.read_text())\n"
        "    except Exception:\n"
        "        return {'path': str(p), 'error': 'decode_failed'}\n"
        "summary = {}\n"
        f"for run_id in ['ext-125m-fa-32K', 'ext-125m-swa-32K-from-fa', 'pretrain-125m-fa']:\n"
        f"    ckpt = pathlib.Path('{REMOTE_REPO_ROOT}/checkpoints/{CANONICAL_PAPER_RUN_ID}') / run_id / 'latest.json'\n"
        "    summary[run_id] = load(ckpt)\n"
        f"for stage, run_id in [('S0_125M','ext-125m-fa-32K'),('S1_125M','ext-125m-swa-32K-from-fa')]:\n"
        f"    exp = pathlib.Path('{REMOTE_REPO_ROOT}/experiments/{CANONICAL_PAPER_RUN_ID}') / stage / run_id / 'hf_export_manifest.json'\n"
        "    summary[f'{stage}__export'] = exp.exists()\n"
        f"exit_file = pathlib.Path('/root/runlogs/h200_a.exit')\n"
        "summary['exit_code'] = exit_file.read_text().strip() if exit_file.exists() else None\n"
        "print(json.dumps(summary))\n"
    )
    ps_cmd = f"ps -p {pid} -o pid="
    proc_alive = _remote_shell(ssh_key, ssh_target, ps_cmd, check=False)
    alive = proc_alive.returncode == 0 and bool(proc_alive.stdout.strip())
    proc = _remote_shell(ssh_key, ssh_target, f"python3 - <<'PY'\n{py}PY")
    payload = json.loads(proc.stdout.strip() or "{}")
    payload["alive"] = alive
    return payload


def _sync_lightweight_results(ssh_key: Path, ssh_target: str, local_out_root: Path) -> None:
    host, port = _parse_ssh_target(ssh_target)
    transport = _rsync_ssh_transport(ssh_key, port=port)
    sync_pairs = [
        (
            f"{REMOTE_REPO_ROOT}/reports/paper/{CANONICAL_PAPER_RUN_ID}/split_batches",
            local_out_root / "reports" / "split_batches",
        ),
        (
            f"{REMOTE_REPO_ROOT}/experiments/{CANONICAL_PAPER_RUN_ID}/S0_125M",
            local_out_root / "experiments" / "S0_125M",
        ),
        (
            f"{REMOTE_REPO_ROOT}/experiments/{CANONICAL_PAPER_RUN_ID}/S1_125M",
            local_out_root / "experiments" / "S1_125M",
        ),
        (
            f"{REMOTE_REPO_ROOT}/experiments/{CANONICAL_PAPER_RUN_ID}/S1_125M_GATE",
            local_out_root / "experiments" / "S1_125M_GATE",
        ),
        (
            "/root/runlogs",
            local_out_root / "runlogs",
        ),
    ]
    for remote_dir, local_dir in sync_pairs:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-az",
            "-e",
            transport,
            f"{host}:{remote_dir}/",
            f"{local_dir}/",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            _print(f"Lightweight sync skipped for {remote_dir}: {proc.stderr.strip()}")


def _verify_hf_exports(repo_id: str, token: str) -> dict[str, bool]:
    api = HfApi(token=token or None)
    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    out: dict[str, bool] = {}
    for stage_id, run_id in HF_STAGE_PREFIXES.items():
        prefix = f"{CANONICAL_PAPER_RUN_ID}/stages/{stage_id}/{run_id}/"
        required = {
            prefix + "hf_export_manifest.json",
            prefix + "checkpoint/latest.json",
        }
        out[stage_id] = required.issubset(repo_files)
    return out


@dataclass
class WatchConfig:
    interval_seconds: int
    disk_size_gb: int
    ready_timeout_seconds: int
    ready_poll_seconds: int
    image: str
    pod_name_prefix: str
    monitor_poll_seconds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch Prime for the next on-demand 8x H200 141GB row, run the 125M Protocol R "
            "h200_a split batch on the pod, verify stage exports on HF, sync lightweight "
            "artifacts locally, and terminate the pod."
        )
    )
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--disk-size-gb", type=int, default=500)
    parser.add_argument("--ready-timeout-seconds", type=int, default=1800)
    parser.add_argument("--ready-poll-seconds", type=int, default=15)
    parser.add_argument("--monitor-poll-seconds", type=int, default=60)
    parser.add_argument("--image", default="ubuntu_22_cuda_12")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--pod-name-prefix", default="ttt-e2e-125m-h200a")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_values = os.environ.copy()
    env_values.update(_load_env_file(args.env_file))
    missing = [key for key in ["HF_TOKEN", "HF_RESULTS_REPO", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_API_KEY"] if not env_values.get(key)]
    if missing:
        raise RuntimeError(f"Missing required env values for h200_a watcher: {', '.join(missing)}")
    ssh_key = _prime_ssh_key_path()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    watch_log = args.log_dir / "h200_a_watch.jsonl"

    cfg = WatchConfig(
        interval_seconds=args.interval_seconds,
        disk_size_gb=args.disk_size_gb,
        ready_timeout_seconds=args.ready_timeout_seconds,
        ready_poll_seconds=args.ready_poll_seconds,
        image=args.image,
        pod_name_prefix=args.pod_name_prefix,
        monitor_poll_seconds=args.monitor_poll_seconds,
    )

    attempt = 0
    while True:
        attempt += 1
        rows = _load_availability()
        target = _select_target(rows)
        _append_jsonl(
            watch_log,
            {
                "event": "availability_check",
                "attempt": attempt,
                "target_found": bool(target),
                "target": target,
            },
        )
        if target is None:
            _print("No on-demand 8x H200 141GB row available yet.")
            if args.once:
                return 1
            time.sleep(max(1, cfg.interval_seconds))
            continue

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pod_name = f"{cfg.pod_name_prefix}-{stamp}"
        local_out_root = args.log_dir / f"h200_a_{stamp}"
        pod_id = None
        try:
            _print(
                f"Target row found: {target['id']} {target.get('price_per_hour')} "
                f"{target.get('gpu_type')} {target.get('provider')} {target.get('location')}"
            )
            pod_id = _create_pod(
                target,
                pod_name=pod_name,
                disk_size_gb=cfg.disk_size_gb,
                image=cfg.image,
                log_path=watch_log,
            )
            ssh_target = _wait_for_ssh_target(
                pod_id,
                timeout_seconds=cfg.ready_timeout_seconds,
                poll_seconds=cfg.ready_poll_seconds,
                log_path=watch_log,
            )
            if not ssh_target:
                raise TimeoutError(f"Pod {pod_id} never exposed SSH target.")

            _bootstrap_remote(ssh_key, ssh_target)
            _sync_repo(ssh_key, ssh_target)
            _write_remote_env(ssh_key, ssh_target, env_values)
            _restore_datasets(ssh_key, ssh_target)
            pid = _launch_batch(ssh_key, ssh_target)
            _append_jsonl(watch_log, {"event": "batch_started", "pod_id": pod_id, "pid": pid})

            while True:
                snap = _remote_status_snapshot(ssh_key, ssh_target, pid)
                snap["event"] = "batch_poll"
                snap["pod_id"] = pod_id
                _append_jsonl(watch_log, snap)
                if not snap.get("alive"):
                    break
                time.sleep(max(5, cfg.monitor_poll_seconds))

            _sync_lightweight_results(ssh_key, ssh_target, local_out_root)
            hf_verify = _verify_hf_exports(env_values["HF_RESULTS_REPO"], env_values["HF_TOKEN"])
            _append_jsonl(
                watch_log,
                {
                    "event": "hf_verify",
                    "pod_id": pod_id,
                    "verification": hf_verify,
                    "local_out_root": str(local_out_root),
                },
            )
            if not all(hf_verify.values()):
                raise RuntimeError(f"HF verification failed for one or more stages: {hf_verify}")

            _append_jsonl(
                watch_log,
                {
                    "event": "watch_complete",
                    "pod_id": pod_id,
                    "local_out_root": str(local_out_root),
                    "verification": hf_verify,
                },
            )
            _print("h200_a watch completed successfully.")
            if pod_id:
                _terminate_pod(pod_id, retries=20, sleep_seconds=30, log_path=watch_log)
            return 0
        except Exception as exc:
            _append_jsonl(
                watch_log,
                {
                    "event": "watch_error",
                    "attempt": attempt,
                    "pod_id": pod_id,
                    "error": str(exc),
                },
            )
            _print(f"Watch attempt failed: {exc}")
        finally:
            if pod_id:
                _terminate_pod(pod_id, retries=20, sleep_seconds=30, log_path=watch_log)

        if args.once:
            return 1
        time.sleep(max(1, cfg.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
