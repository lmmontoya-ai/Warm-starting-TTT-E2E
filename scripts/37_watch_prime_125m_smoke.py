#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "artifacts" / "prime_watch"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _print(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def _run(cmd: list[str], *, check: bool = True, capture_output: bool = False, text: bool = True, env: dict[str, str] | None = None):
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


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_availability() -> list[dict]:
    payload = json.loads(
        _run_capture(
            [
                "prime",
                "availability",
                "list",
                "--gpu-type",
                "H100_80GB",
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


def _select_target(rows: list[dict], *, max_price: float, provider_substring: str) -> dict | None:
    matches: list[dict] = []
    provider_substring = provider_substring.lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("is_spot", False)):
            continue
        if int(row.get("gpu_count", 0) or 0) != 8:
            continue
        if "H100" not in str(row.get("gpu_type", "")):
            continue
        provider = str(row.get("provider", "")).lower()
        if provider_substring and provider_substring not in provider:
            continue
        price = float(row.get("price_value", 1e9) or 1e9)
        if price > max_price:
            continue
        matches.append(row)
    matches.sort(key=lambda item: float(item.get("price_value", 1e9) or 1e9))
    return matches[0] if matches else None


def _prime_ssh_key_path() -> Path:
    output = _run_capture(["prime", "config", "view"])
    for line in output.splitlines():
        if "SSH Key Path" in line:
            parts = [part.strip() for part in line.split("│") if part.strip()]
            if parts:
                candidate = parts[-1]
                path = Path(candidate).expanduser()
                if path.exists():
                    return path
    raise FileNotFoundError("Could not determine Prime SSH key path from `prime config view`.")


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


def _create_pod(row: dict, *, pod_name: str, disk_size_gb: int, log_path: Path) -> str:
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
        "--yes",
    ]
    proc = _run(cmd, capture_output=True)
    created_id = None
    for line in proc.stdout.splitlines():
        if "Successfully created pod" in line:
            created_id = line.rsplit(" ", 1)[-1].strip()
            break
    if not created_id:
        raise RuntimeError(f"Could not parse created pod id from Prime output:\n{proc.stdout}")
    _append_jsonl(log_path, {"event": "pod_created", "row": row, "pod_id": created_id})
    return created_id


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


def _remote_shell(ssh_key: Path, ssh_target: str, command: str, *, env: dict[str, str] | None = None) -> None:
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        str(ssh_key),
        ssh_target,
        command,
    ]
    _run(cmd, env=env)


def _remote_shell_capture(ssh_key: Path, ssh_target: str, command: str) -> str:
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        str(ssh_key),
        ssh_target,
        command,
    ]
    return _run_capture(cmd)


def _sync_repo(ssh_key: Path, ssh_target: str, remote_repo_root: str) -> None:
    exclude = [
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "artifacts",
        "experiments",
        "checkpoints",
        "reports",
        "data",
        ".DS_Store",
    ]
    cmd = [
        "rsync",
        "-az",
        "--delete",
    ]
    for item in exclude:
        cmd.extend(["--exclude", item])
    cmd.extend(
        [
            "-e",
            f"ssh -o StrictHostKeyChecking=no -i {shlex.quote(str(ssh_key))}",
            f"{REPO_ROOT}/",
            f"{ssh_target}:{remote_repo_root}/",
        ]
    )
    _run(cmd)


def _write_remote_env(
    ssh_key: Path,
    ssh_target: str,
    remote_repo_root: str,
    env_values: dict[str, str],
) -> None:
    keys = [
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "B2_ENDPOINT_URL",
        "B2_BUCKET",
        "B2_DATASET_PREFIX",
        "HF_TOKEN",
        "HF_USERNAME",
        "HF_RESULTS_REPO",
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
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                str(ssh_key),
                str(temp_path),
                f"{ssh_target}:{remote_repo_root}/.env.runtime",
            ]
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _bootstrap_remote(ssh_key: Path, ssh_target: str, remote_repo_root: str) -> None:
    command = (
        "set -euo pipefail; "
        "sudo apt-get update -y >/dev/null; "
        "sudo apt-get install -y curl rsync git >/dev/null; "
        "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi; "
        f"mkdir -p {shlex.quote(remote_repo_root)}"
    )
    _remote_shell(ssh_key, ssh_target, command)


def _restore_smoke_dataset(ssh_key: Path, ssh_target: str, remote_repo_root: str, remote_data_root: str) -> None:
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(remote_repo_root)}; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "set -a; source .env.runtime; set +a; "
        f"mkdir -p {shlex.quote(remote_data_root)}/books3; "
        "uv venv >/dev/null; "
        ". .venv/bin/activate; "
        "uv sync --exact >/dev/null; "
        "uv run --exact python scripts/28_fetch_b2_dataset.py "
        f"--dest-root {shlex.quote(remote_data_root)} "
        "--datasets dclm_filter_8k "
        "--splits train "
        "--b2-prefix \"$B2_DATASET_PREFIX/paper_budget_125m_val-full\" "
        "--no-progress"
    )
    _remote_shell(ssh_key, ssh_target, command)


def _run_smoke(
    ssh_key: Path,
    ssh_target: str,
    remote_repo_root: str,
    remote_data_root: str,
    *,
    paper_run_id: str,
    exp_folder: str,
    pretrain_steps: int,
    accum_steps: int,
    seq_length: int,
    global_batch_size: int,
) -> None:
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(remote_repo_root)}; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "set -a; source .env.runtime; set +a; "
        ". .venv/bin/activate; "
        "uv run --exact python scripts/23_warmstart_registry.py "
        f"--paper-run-id {shlex.quote(paper_run_id)} "
        "--registry ./configs/research/warmstart_registry.yaml "
        "--stage-ids S0_PRETRAIN_FA_125M "
        "--runtime-mode jax_train "
        f"--exp-folder {shlex.quote(exp_folder)} "
        "--exp-dir ./experiments "
        "--checkpoint-root ./checkpoints "
        "--profile-root ./artifacts/external_models "
        f"--dclm-root {shlex.quote(remote_data_root + '/dclm_filter_8k')} "
        f"--books-root {shlex.quote(remote_data_root + '/books3')} "
        f"--pretrain-steps {pretrain_steps} "
        "--adapt-steps 1 "
        "--ext-steps 1 "
        "--seed 0 "
        "--save-milestone-freq 1 "
        f"--global-batch-size {global_batch_size} "
        f"--accum-steps {accum_steps} "
        f"--seq-length {seq_length} "
        "--wandb-entity \"$WANDB_ENTITY\" "
        "--wandb-project \"$WANDB_PROJECT\" "
        "--wandb-key env"
    )
    _remote_shell(ssh_key, ssh_target, command)


def _sync_results_back(
    ssh_key: Path,
    ssh_target: str,
    remote_repo_root: str,
    paper_run_id: str,
    exp_folder: str,
    local_out_root: Path,
) -> None:
    local_out_root.mkdir(parents=True, exist_ok=True)
    for remote_rel, local_rel in [
        (f"{remote_repo_root}/experiments/{paper_run_id}", local_out_root / "experiments"),
        (f"{remote_repo_root}/checkpoints/{exp_folder}", local_out_root / "checkpoints"),
        (f"{remote_repo_root}/reports/paper/{paper_run_id}", local_out_root / "reports"),
    ]:
        cmd = [
            "rsync",
            "-az",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -i {shlex.quote(str(ssh_key))}",
            f"{ssh_target}:{remote_rel}/",
            f"{local_rel}/",
        ]
        try:
            _run(cmd)
        except subprocess.CalledProcessError:
            continue


def _smoke_succeeded(local_out_root: Path, paper_run_id: str) -> bool:
    metrics = list((local_out_root / "experiments").rglob("metrics.jsonl"))
    if not metrics:
        return False
    latest = sorted(metrics)[-1]
    text = latest.read_text(encoding="utf-8")
    return '"runtime_mode": "jax_train"' in text and '"loss"' in text


@dataclass
class WatchConfig:
    interval_seconds: int
    max_price: float
    provider_substring: str
    disk_size_gb: int
    ready_timeout_seconds: int
    ready_poll_seconds: int
    pretrain_steps: int
    accum_steps: int
    seq_length: int
    global_batch_size: int
    pod_name_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch Prime for the cheap non-spot 8x H100 row and run the 125M FA smoke until success."
    )
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-price", type=float, default=14.40)
    parser.add_argument("--provider-substring", default="prime")
    parser.add_argument("--disk-size-gb", type=int, default=500)
    parser.add_argument("--ready-timeout-seconds", type=int, default=1200)
    parser.add_argument("--ready-poll-seconds", type=int, default=15)
    parser.add_argument("--pretrain-steps", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--pod-name-prefix", default="ttt-e2e-125m-smoke")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_values = os.environ.copy()
    env_values.update(_load_env_file(args.env_file))
    ssh_key = _prime_ssh_key_path()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    watch_log = args.log_dir / "watch.jsonl"

    cfg = WatchConfig(
        interval_seconds=args.interval_seconds,
        max_price=args.max_price,
        provider_substring=args.provider_substring,
        disk_size_gb=args.disk_size_gb,
        ready_timeout_seconds=args.ready_timeout_seconds,
        ready_poll_seconds=args.ready_poll_seconds,
        pretrain_steps=args.pretrain_steps,
        accum_steps=args.accum_steps,
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        pod_name_prefix=args.pod_name_prefix,
    )

    attempt = 0
    while True:
        attempt += 1
        rows = _load_availability()
        target = _select_target(rows, max_price=cfg.max_price, provider_substring=cfg.provider_substring)
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
            _print("No matching cheap Prime-owned 8x H100 row yet.")
            if args.once:
                return 1
            time.sleep(max(1, cfg.interval_seconds))
            continue

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pod_name = f"{cfg.pod_name_prefix}-{stamp}"
        paper_run_id = f"prime_125m_fa_smoke_{stamp}"
        exp_folder = paper_run_id
        local_out_root = args.log_dir / paper_run_id
        pod_id = None
        try:
            _print(f"Target row found: {target['id']} {target['price_per_hour']} {target['provider']} {target['location']}")
            pod_id = _create_pod(target, pod_name=pod_name, disk_size_gb=cfg.disk_size_gb, log_path=watch_log)
            ssh_target = _wait_for_ssh_target(
                pod_id,
                timeout_seconds=cfg.ready_timeout_seconds,
                poll_seconds=cfg.ready_poll_seconds,
                log_path=watch_log,
            )
            if not ssh_target:
                raise TimeoutError(f"Pod {pod_id} never exposed SSH target.")

            remote_repo_root = f"/root/{REPO_ROOT.name}"
            remote_data_root = "/root/ttt-e2e-data"
            _bootstrap_remote(ssh_key, ssh_target, remote_repo_root)
            _sync_repo(ssh_key, ssh_target, remote_repo_root)
            _write_remote_env(ssh_key, ssh_target, remote_repo_root, env_values)
            _restore_smoke_dataset(ssh_key, ssh_target, remote_repo_root, remote_data_root)
            _run_smoke(
                ssh_key,
                ssh_target,
                remote_repo_root,
                remote_data_root,
                paper_run_id=paper_run_id,
                exp_folder=exp_folder,
                pretrain_steps=cfg.pretrain_steps,
                accum_steps=cfg.accum_steps,
                seq_length=cfg.seq_length,
                global_batch_size=cfg.global_batch_size,
            )
            _sync_results_back(ssh_key, ssh_target, remote_repo_root, paper_run_id, exp_folder, local_out_root)
            success = _smoke_succeeded(local_out_root, paper_run_id)
            _append_jsonl(
                watch_log,
                {
                    "event": "smoke_complete",
                    "pod_id": pod_id,
                    "paper_run_id": paper_run_id,
                    "success": success,
                },
            )
            if success:
                _print(f"Smoke succeeded: {paper_run_id}")
                if pod_id:
                    _terminate_pod(pod_id, retries=20, sleep_seconds=30, log_path=watch_log)
                return 0
            _print(f"Smoke did not produce success artifacts: {paper_run_id}")
        except Exception as exc:
            _append_jsonl(
                watch_log,
                {
                    "event": "watch_error",
                    "attempt": attempt,
                    "pod_id": pod_id,
                    "paper_run_id": paper_run_id,
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
