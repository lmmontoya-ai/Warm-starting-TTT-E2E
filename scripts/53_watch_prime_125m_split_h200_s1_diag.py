#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPER_PATH = REPO_ROOT / "scripts" / "48_watch_prime_125m_split_h200_a.py"
_SPEC = importlib.util.spec_from_file_location("_prime_h200_helper", _HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load helper module from {_HELPER_PATH}")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

DEFAULT_LOG_DIR = _MOD.DEFAULT_LOG_DIR
CANONICAL_PAPER_RUN_ID = _MOD.CANONICAL_PAPER_RUN_ID
REMOTE_REPO_ROOT = _MOD.REMOTE_REPO_ROOT
REMOTE_DATA_ROOT = _MOD.REMOTE_DATA_ROOT

HF_STAGE_PREFIX = "ext-125m-swa-32K-from-fa"


def _launch_batch(ssh_key: Path, ssh_target: str) -> int:
    command = (
        "set -euo pipefail; "
        f"cd {shlex.quote(REMOTE_REPO_ROOT)}; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "cat > /root/runlogs/h200_s1_diag_wrapper.sh <<'SH'\n"
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
        "--batch h200_s1_diag "
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
        "chmod +x /root/runlogs/h200_s1_diag_wrapper.sh; "
        "rm -f /root/runlogs/h200_s1_diag.exit /root/runlogs/h200_s1_diag.pid; "
        "nohup bash -lc '/root/runlogs/h200_s1_diag_wrapper.sh > /root/runlogs/h200_s1_diag.log 2>&1; "
        "rc=$?; printf \"%s\\n\" \"$rc\" > /root/runlogs/h200_s1_diag.exit' "
        "</dev/null >/dev/null 2>&1 & "
        "echo $! > /root/runlogs/h200_s1_diag.pid; "
        "cat /root/runlogs/h200_s1_diag.pid"
    )
    proc = _MOD._remote_shell(ssh_key, ssh_target, command)
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
        f"for run_id in ['ext-125m-swa-32K-from-fa', 'pretrain-125m-fa']:\n"
        f"    ckpt = pathlib.Path('{REMOTE_REPO_ROOT}/checkpoints/{CANONICAL_PAPER_RUN_ID}') / run_id / 'latest.json'\n"
        "    summary[run_id] = load(ckpt)\n"
        f"summary['batch_summary'] = load(pathlib.Path('{REMOTE_REPO_ROOT}/reports/paper/{CANONICAL_PAPER_RUN_ID}/split_batches/h200_s1_diag.json'))\n"
        f"summary['s1_export_manifest'] = pathlib.Path('{REMOTE_REPO_ROOT}/experiments/{CANONICAL_PAPER_RUN_ID}/S1_125M/ext-125m-swa-32K-from-fa/hf_export_manifest.json').exists()\n"
        f"summary['reference_swa_smoke'] = load(pathlib.Path('{REMOTE_REPO_ROOT}/artifacts/reference_smokes/125m_32k_swa_protocol_r_b8/reference_swa_smoke_manifest.json'))\n"
        f"summary['local_swa_probe'] = load(pathlib.Path('{REMOTE_REPO_ROOT}/artifacts/oom_diagnosis/local_125m_32k_swa_probe/summary.json'))\n"
        "exit_file = pathlib.Path('/root/runlogs/h200_s1_diag.exit')\n"
        "summary['exit_code'] = exit_file.read_text().strip() if exit_file.exists() else None\n"
        "print(json.dumps(summary))\n"
    )
    proc_alive = _MOD._remote_shell(ssh_key, ssh_target, f"ps -p {pid} -o pid=", check=False)
    alive = proc_alive.returncode == 0 and bool(proc_alive.stdout.strip())
    proc = _MOD._remote_shell(ssh_key, ssh_target, f"python3 - <<'PY'\n{py}PY")
    payload = json.loads(proc.stdout.strip() or "{}")
    payload["alive"] = alive
    return payload


def _sync_lightweight_results(ssh_key: Path, ssh_target: str, local_out_root: Path) -> None:
    host, port = _MOD._parse_ssh_target(ssh_target)
    transport = _MOD._rsync_ssh_transport(ssh_key, port=port)
    sync_pairs = [
        (
            f"{REMOTE_REPO_ROOT}/reports/paper/{CANONICAL_PAPER_RUN_ID}/split_batches",
            local_out_root / "reports" / "split_batches",
        ),
        (
            f"{REMOTE_REPO_ROOT}/artifacts/reference_smokes/125m_32k_swa_protocol_r_b8",
            local_out_root / "artifacts" / "reference_smokes" / "125m_32k_swa_protocol_r_b8",
        ),
        (
            f"{REMOTE_REPO_ROOT}/artifacts/oom_diagnosis/local_125m_32k_swa_probe",
            local_out_root / "artifacts" / "oom_diagnosis" / "local_125m_32k_swa_probe",
        ),
        (
            f"{REMOTE_REPO_ROOT}/experiments/{CANONICAL_PAPER_RUN_ID}/S1_125M",
            local_out_root / "experiments" / "S1_125M",
        ),
        (
            "/root/runlogs",
            local_out_root / "runlogs",
        ),
    ]
    for remote_dir, local_dir in sync_pairs:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["rsync", "-az", "-e", transport, f"{host}:{remote_dir}/", f"{local_dir}/"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            _MOD._print(f"Lightweight sync skipped for {remote_dir}: {proc.stderr.strip()}")


def _summary_allows_full_s1(summary: dict | None) -> bool:
    if not isinstance(summary, dict):
        return False
    rows = summary.get("rows")
    if not isinstance(rows, list):
        return False
    return any(
        isinstance(row, dict)
        and row.get("step_id") == "export_stage"
        and row.get("stage_id") == "S1_125M"
        and row.get("returncode") == 0
        for row in rows
    )


def _verify_hf_export(repo_id: str, token: str) -> bool:
    api = HfApi(token=token or None)
    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    prefix = f"{CANONICAL_PAPER_RUN_ID}/stages/S1_125M/{HF_STAGE_PREFIX}/"
    required = {
        prefix + "hf_export_manifest.json",
        prefix + "checkpoint/latest.json",
    }
    return required.issubset(repo_files)


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
            "h200_s1_diag batch on the pod, sync diagnostic artifacts locally, optionally verify "
            "S1 HF export if the full stage runs, and terminate the pod."
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
    parser.add_argument("--pod-name-prefix", default="ttt-e2e-125m-h200s1diag")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_values = os.environ.copy()
    env_values.update(_MOD._load_env_file(args.env_file))
    missing = [key for key in ["HF_TOKEN", "HF_RESULTS_REPO", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_API_KEY"] if not env_values.get(key)]
    if missing:
        raise RuntimeError(f"Missing required env values for h200_s1_diag watcher: {', '.join(missing)}")
    ssh_key = _MOD._prime_ssh_key_path()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    watch_log = args.log_dir / "h200_s1_diag_watch.jsonl"

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
        rows = _MOD._load_availability()
        target = _MOD._select_target(rows)
        _MOD._append_jsonl(
            watch_log,
            {"event": "availability_check", "attempt": attempt, "target_found": bool(target), "target": target},
        )
        if target is None:
            _MOD._print("No on-demand 8x H200 141GB row available yet.")
            if args.once:
                return 1
            time.sleep(max(1, cfg.interval_seconds))
            continue

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pod_name = f"{cfg.pod_name_prefix}-{stamp}"
        local_out_root = args.log_dir / f"h200_s1_diag_{stamp}"
        pod_id = None
        try:
            _MOD._print(
                f"Target row found: {target['id']} {target.get('price_per_hour')} "
                f"{target.get('gpu_type')} {target.get('provider')} {target.get('location')}"
            )
            pod_id = _MOD._create_pod(
                target,
                pod_name=pod_name,
                disk_size_gb=cfg.disk_size_gb,
                image=cfg.image,
                log_path=watch_log,
            )
            ssh_target = _MOD._wait_for_ssh_target(
                pod_id,
                timeout_seconds=cfg.ready_timeout_seconds,
                poll_seconds=cfg.ready_poll_seconds,
                log_path=watch_log,
            )
            if not ssh_target:
                raise TimeoutError(f"Pod {pod_id} never exposed SSH target.")

            _MOD._bootstrap_remote(ssh_key, ssh_target)
            _MOD._sync_repo(ssh_key, ssh_target)
            _MOD._write_remote_env(ssh_key, ssh_target, env_values)
            _MOD._restore_datasets(ssh_key, ssh_target)
            pid = _launch_batch(ssh_key, ssh_target)
            _MOD._append_jsonl(watch_log, {"event": "batch_started", "pod_id": pod_id, "pid": pid})

            final_snap: dict | None = None
            while True:
                snap = _remote_status_snapshot(ssh_key, ssh_target, pid)
                final_snap = snap
                snap["event"] = "batch_poll"
                snap["pod_id"] = pod_id
                _MOD._append_jsonl(watch_log, snap)
                if not snap.get("alive"):
                    break
                time.sleep(max(5, cfg.monitor_poll_seconds))

            _sync_lightweight_results(ssh_key, ssh_target, local_out_root)
            batch_summary = None if final_snap is None else final_snap.get("batch_summary")
            hf_ok = None
            if _summary_allows_full_s1(batch_summary):
                hf_ok = _verify_hf_export(env_values["HF_RESULTS_REPO"], env_values["HF_TOKEN"])
                _MOD._append_jsonl(
                    watch_log,
                    {"event": "hf_verify", "pod_id": pod_id, "verification": {"S1_125M": hf_ok}, "local_out_root": str(local_out_root)},
                )
                if not hf_ok:
                    raise RuntimeError("HF verification failed for S1_125M export.")

            _MOD._append_jsonl(
                watch_log,
                {
                    "event": "watch_complete",
                    "pod_id": pod_id,
                    "local_out_root": str(local_out_root),
                    "verification": {"S1_125M": hf_ok} if hf_ok is not None else None,
                },
            )
            _MOD._print("h200_s1_diag watch completed successfully.")
            if pod_id:
                _MOD._terminate_pod(pod_id, retries=20, sleep_seconds=30, log_path=watch_log)
            return 0
        except Exception as exc:
            _MOD._append_jsonl(
                watch_log,
                {"event": "watch_error", "attempt": attempt, "pod_id": pod_id, "error": str(exc)},
            )
            _MOD._print(f"Watch attempt failed: {exc}")
        finally:
            if pod_id:
                _MOD._terminate_pod(pod_id, retries=20, sleep_seconds=30, log_path=watch_log)

        if args.once:
            return 1
        time.sleep(max(1, cfg.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
