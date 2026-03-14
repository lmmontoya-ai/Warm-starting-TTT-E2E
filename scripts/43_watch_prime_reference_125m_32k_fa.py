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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "artifacts" / "prime_watch"
DEFAULT_SEED_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "live_runs"
    / "prime_125m_ladder_20260312a"
    / "checkpoints"
    / "pretrain-125m-fa"
)
MINIMAL_SYNC_PATHS = [
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("ttt"),
    Path("configs"),
    Path("scripts/28_fetch_b2_dataset.py"),
    Path("scripts/40_diagnose_125m_32k_fa_oom.py"),
    Path("scripts/41_run_reference_125m_32k_fa_smoke.py"),
    Path("ttte2e_reference/e2e"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _print(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


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


def _ssh_base_cmd(ssh_key: Path, *, scp: bool = False, port: str | None = None) -> list[str]:
    cmd = ["scp" if scp else "ssh"]
    if scp:
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
            cmd.extend(["-P", port])
        return cmd
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
        cmd.extend(["-p", port])
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


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def _load_availability() -> list[dict]:
    payload = json.loads(
        _run_capture(
            [
                "prime",
                "availability",
                "list",
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
        try:
            count = int(row.get("gpu_count", 0) or 0)
        except Exception:
            continue
        gpu_type = str(row.get("gpu_type", ""))
        if count != 8:
            continue
        if "A100 80GB" not in gpu_type and "H100 80GB" not in gpu_type:
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
                path = Path(parts[-1]).expanduser()
                if path.exists():
                    return path
    raise FileNotFoundError("Could not determine Prime SSH key path from `prime config view`.")


def _safe_int(value, fallback: int) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return fallback


def _create_pod(row: dict, *, pod_name: str, disk_size_gb: int, image: str, log_path: Path) -> str:
    vcpus = _safe_int(row.get("vcpus"), 96)
    memory = _safe_int(row.get("memory_gb"), 640)
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


def _parse_ssh_target(ssh_target: str) -> tuple[str, str | None]:
    ssh_target = ssh_target.strip()
    if " -p " in ssh_target:
        host, port = ssh_target.rsplit(" -p ", 1)
        return host.strip(), port.strip()
    return ssh_target, None


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
    host, port = _parse_ssh_target(ssh_target)
    cmd = _ssh_base_cmd(ssh_key, scp=False, port=port)
    cmd.extend([host, command])
    _run(cmd, env=env)


def _sync_repo(ssh_key: Path, ssh_target: str, remote_repo_root: str) -> None:
    host, port = _parse_ssh_target(ssh_target)
    transport = _rsync_ssh_transport(ssh_key, port=port)
    for rel_path in MINIMAL_SYNC_PATHS:
        local_path = REPO_ROOT / rel_path
        if not local_path.exists():
            raise FileNotFoundError(f"Required sync path is missing: {local_path}")
        remote_parent = f"{remote_repo_root}/{rel_path.parent.as_posix()}" if rel_path.parent != Path(".") else remote_repo_root
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
            remote_file = f"{host}:{remote_repo_root}/{rel_path.as_posix()}"
            cmd.extend([str(local_path), remote_file])
        _run(cmd)


def _write_remote_env(ssh_key: Path, ssh_target: str, remote_repo_root: str, env_values: dict[str, str]) -> None:
    host, port = _parse_ssh_target(ssh_target)
    keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "B2_ENDPOINT_URL",
        "B2_BUCKET",
        "B2_DATASET_PREFIX",
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
            + [
                str(temp_path),
                f"{host}:{remote_repo_root}/.env.runtime",
            ]
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _bootstrap_remote(ssh_key: Path, ssh_target: str, remote_repo_root: str) -> None:
    command = (
        "set -euo pipefail; "
        "sudo apt-get update -y >/dev/null; "
        "sudo apt-get install -y curl rsync git unzip awscli >/dev/null; "
        "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi; "
        "command -v aws >/dev/null 2>&1; "
        "aws --version >/dev/null 2>&1; "
        f"mkdir -p {shlex.quote(remote_repo_root)}"
    )
    _remote_shell(ssh_key, ssh_target, command)


def _restore_books3_train(ssh_key: Path, ssh_target: str, remote_repo_root: str, remote_data_root: str) -> None:
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(remote_repo_root)}; "
        "set -a; source .env.runtime; set +a; "
        f"mkdir -p {shlex.quote(remote_data_root + '/books3/train')}; "
        "aws s3 sync "
        "\"s3://$B2_BUCKET/$B2_DATASET_PREFIX/paper_budget_125m_val-full/books3/train\" "
        f"{shlex.quote(remote_data_root + '/books3/train')} "
        "--endpoint-url \"$B2_ENDPOINT_URL\" "
        "--region \"$AWS_DEFAULT_REGION\" "
        "--only-show-errors; "
        f"test -f {shlex.quote(remote_data_root + '/books3/train/zarr.json')}"
    )
    _remote_shell(ssh_key, ssh_target, command)


def _sync_seed_checkpoint(
    ssh_key: Path,
    ssh_target: str,
    local_seed_root: Path,
    remote_repo_root: str,
    remote_seed_name: str,
) -> str:
    host, port = _parse_ssh_target(ssh_target)
    if not local_seed_root.exists():
        raise FileNotFoundError(f"Local seed checkpoint root not found: {local_seed_root}")
    remote_seed_root = f"{remote_repo_root}/checkpoints/{remote_seed_name}"
    _remote_shell(
        ssh_key,
        ssh_target,
        f"set -euo pipefail; mkdir -p {shlex.quote(remote_seed_root)}",
    )
    cmd = [
        "rsync",
        "-az",
        "-e",
        _rsync_ssh_transport(ssh_key, port=port),
        f"{local_seed_root}/",
        f"{host}:{remote_seed_root}/",
    ]
    _run(cmd)
    return remote_seed_root


def _run_reference_smoke(
    ssh_key: Path,
    ssh_target: str,
    remote_repo_root: str,
    remote_books_root: str,
    remote_seed_root: str,
    run_id: str,
) -> int:
    host, port = _parse_ssh_target(ssh_target)
    log_path = f"{remote_repo_root}/artifacts/reference_smokes/{run_id}/reference_125m_32k_fa_smoke.log"
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(remote_repo_root)}; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "uv venv --python 3.13 >/dev/null 2>&1; "
        ". .venv/bin/activate; "
        "uv sync --exact --python 3.13 >/dev/null 2>&1; "
        "uv run --exact python scripts/41_run_reference_125m_32k_fa_smoke.py "
        f"--repo-root {shlex.quote(remote_repo_root)} "
        f"--books-root {shlex.quote(remote_books_root)} "
        f"--checkpoint-root {shlex.quote(remote_repo_root + '/checkpoints')} "
        f"--resume-checkpoint-dir {shlex.quote(remote_seed_root)} "
        f"--exp-folder reference_smokes "
        f"--exp-name {shlex.quote(run_id)} "
        f"--log-path {shlex.quote(log_path)}"
    )
    proc = subprocess.run(
        _ssh_base_cmd(ssh_key, scp=False, port=port) + [host, command],
        capture_output=True,
        text=True,
    )
    return int(proc.returncode)


def _run_local_diagnosis(
    ssh_key: Path,
    ssh_target: str,
    remote_repo_root: str,
    remote_books_root: str,
    remote_seed_root: str,
    run_id: str,
) -> int:
    host, port = _parse_ssh_target(ssh_target)
    artifact_root = f"{remote_repo_root}/artifacts/oom_diagnosis"
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(remote_repo_root)}; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "uv venv --python 3.13 >/dev/null 2>&1; "
        ". .venv/bin/activate; "
        "uv sync --exact --python 3.13 >/dev/null 2>&1; "
        "uv run --exact python scripts/40_diagnose_125m_32k_fa_oom.py "
        f"--books-root {shlex.quote(remote_books_root)} "
        f"--checkpoint-root {shlex.quote(remote_repo_root + '/checkpoints')} "
        f"--resume-checkpoint-path {shlex.quote(remote_seed_root)} "
        "--resume-checkpoint-format orbax "
        "--resume-step 4560 "
        "--load-part params "
        "--total-steps 2 "
        "--save-milestone-freq 999 "
        "--skip-execute "
        f"--exp-folder oom_diagnosis "
        f"--exp-name {shlex.quote(run_id)} "
        f"--artifact-root {shlex.quote(artifact_root)}"
    )
    proc = subprocess.run(
        _ssh_base_cmd(ssh_key, scp=False, port=port) + [host, command],
        capture_output=True,
        text=True,
    )
    return int(proc.returncode)


def _sync_remote_dir(ssh_key: Path, ssh_target: str, remote_dir: str, local_dir: Path) -> None:
    host, port = _parse_ssh_target(ssh_target)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-az",
        "-e",
        _rsync_ssh_transport(ssh_key, port=port),
        f"{host}:{remote_dir}/",
        f"{local_dir}/",
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        pass


def _sync_results_back(
    ssh_key: Path,
    ssh_target: str,
    remote_repo_root: str,
    local_out_root: Path,
) -> None:
    local_out_root.mkdir(parents=True, exist_ok=True)
    _sync_remote_dir(
        ssh_key,
        ssh_target,
        f"{remote_repo_root}/artifacts/reference_smokes",
        local_out_root / "reference_smokes",
    )
    _sync_remote_dir(
        ssh_key,
        ssh_target,
        f"{remote_repo_root}/artifacts/oom_diagnosis",
        local_out_root / "oom_diagnosis",
    )
    _sync_remote_dir(
        ssh_key,
        ssh_target,
        f"{remote_repo_root}/experiments/reference_smokes",
        local_out_root / "experiments" / "reference_smokes",
    )
    _sync_remote_dir(
        ssh_key,
        ssh_target,
        f"{remote_repo_root}/experiments/oom_diagnosis",
        local_out_root / "experiments" / "oom_diagnosis",
    )


@dataclass
class WatchConfig:
    interval_seconds: int
    disk_size_gb: int
    ready_timeout_seconds: int
    ready_poll_seconds: int
    image: str
    pod_name_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch Prime for the next 8x A100/H100 80GB row, run the reference 125M 32K FA smoke "
            "and the local compile-memory diagnosis on the same pod, sync artifacts locally, and terminate the pod."
        )
    )
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--disk-size-gb", type=int, default=500)
    parser.add_argument("--ready-timeout-seconds", type=int, default=1800)
    parser.add_argument("--ready-poll-seconds", type=int, default=15)
    parser.add_argument("--image", default="ubuntu_22_cuda_12")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--pod-name-prefix", default="ttt-e2e-ref-125m-32k")
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_values = os.environ.copy()
    env_values.update(_load_env_file(args.env_file))
    ssh_key = _prime_ssh_key_path()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    watch_log = args.log_dir / "reference_125m_32k_fa_watch.jsonl"

    cfg = WatchConfig(
        interval_seconds=args.interval_seconds,
        disk_size_gb=args.disk_size_gb,
        ready_timeout_seconds=args.ready_timeout_seconds,
        ready_poll_seconds=args.ready_poll_seconds,
        image=args.image,
        pod_name_prefix=args.pod_name_prefix,
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
            _print("No 8x A100/H100 80GB row available yet.")
            if args.once:
                return 1
            time.sleep(max(1, cfg.interval_seconds))
            continue

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pod_name = f"{cfg.pod_name_prefix}-{stamp}"
        run_id = f"reference_125m_32k_fa_{stamp}"
        local_out_root = args.log_dir / run_id
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

            remote_repo_root = f"/root/{REPO_ROOT.name}"
            remote_data_root = "/root/ttt-e2e-data"

            _bootstrap_remote(ssh_key, ssh_target, remote_repo_root)
            _sync_repo(ssh_key, ssh_target, remote_repo_root)
            _write_remote_env(ssh_key, ssh_target, remote_repo_root, env_values)
            _restore_books3_train(ssh_key, ssh_target, remote_repo_root, remote_data_root)
            remote_seed_root = _sync_seed_checkpoint(
                ssh_key,
                ssh_target,
                args.seed_root,
                remote_repo_root,
                "seed_pretrain_125m_fa_4560",
            )

            ref_rc = _run_reference_smoke(
                ssh_key,
                ssh_target,
                remote_repo_root,
                f"{remote_data_root}/books3",
                remote_seed_root,
                run_id,
            )
            _append_jsonl(
                watch_log,
                {
                    "event": "reference_smoke_finished",
                    "pod_id": pod_id,
                    "run_id": run_id,
                    "returncode": ref_rc,
                },
            )

            diag_rc = _run_local_diagnosis(
                ssh_key,
                ssh_target,
                remote_repo_root,
                f"{remote_data_root}/books3",
                remote_seed_root,
                run_id,
            )
            _append_jsonl(
                watch_log,
                {
                    "event": "local_diagnosis_finished",
                    "pod_id": pod_id,
                    "run_id": run_id,
                    "returncode": diag_rc,
                },
            )

            _sync_results_back(ssh_key, ssh_target, remote_repo_root, local_out_root)
            _append_jsonl(
                watch_log,
                {
                    "event": "watch_complete",
                    "pod_id": pod_id,
                    "run_id": run_id,
                    "reference_returncode": ref_rc,
                    "diagnosis_returncode": diag_rc,
                    "local_out_root": str(local_out_root),
                },
            )
            _print(f"Reference watch completed: {run_id}")
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
