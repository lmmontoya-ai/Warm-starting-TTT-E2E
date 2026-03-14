#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from ttt.research.types import utc_now_iso


def _resolve_uv_executable() -> str:
    candidates = [shutil.which("uv")]
    home = Path.home()
    candidates.extend(
        [
            str(home / ".local" / "bin" / "uv"),
            "/root/.local/bin/uv",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return str(path)
    raise FileNotFoundError("Could not locate the `uv` executable.")


UV_EXECUTABLE = _resolve_uv_executable()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _visible_devices() -> list[str]:
    current = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if current:
        return [part.strip() for part in current.split(",") if part.strip()]
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True, text=True)
    except Exception:
        return ["0"]
    return [str(i) for i, _ in enumerate(proc.stdout.splitlines())]


def _gpu_snapshot() -> list[dict[str, int | str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except Exception as exc:
        return [{"error": repr(exc)}]
    rows: list[dict[str, int | str]] = []
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": parts[0],
                "memory_used_mib": int(parts[1]),
                "memory_total_mib": int(parts[2]),
                "utilization_gpu_pct": int(parts[3]),
            }
        )
    return rows


def _read_metrics(metrics_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    if not metrics_path.exists():
        return {}, {}
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {}, {}
    return rows[0], rows[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a faithful local 125M 32K SWA 2-step smoke on 8 GPUs, and only if that "
            "fails, run the older 1/2/8 GPU compile diagnosis for explanation."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/oom_diagnosis"))
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--dclm-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--resume-checkpoint-path", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-format", default="orbax")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--exp-folder", default="protocol_r_125m_main_v1_s1_localdiag")
    parser.add_argument("--paper-run-id", default="protocol_r_125m_main_v1_s1_localdiag")
    parser.add_argument("--run-id", default="ext-125m-swa-32K-faithful-gate2")
    parser.add_argument("--stage-id", default="S1_125M_FAITHFUL_GATE")
    parser.add_argument("--device-counts", default="1,2,8")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=32768)
    parser.add_argument("--n-data-parallel", type=int, default=8)
    parser.add_argument("--n-state-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _faithful_gate_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "train",
        "+deploy=interactive",
        "+experiment=125m/pretrained/ext-125m-swa-32K-from-fa",
        "training.runtime_mode=jax_train",
        "training.log_wandb=false",
        "training.wandb_entity=none",
        "training.wandb_project=none",
        "training.wandb_key=env",
        f"training.exp_dir={args.exp_dir}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_name={args.run_id}",
        f"training.paper_run_id={args.paper_run_id}",
        f"training.stage_id={args.stage_id}",
        f"training.run_id={args.run_id}",
        f"training.total_steps={args.steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        f"training.checkpoint_path={args.checkpoint_root}",
        f"training.resume_checkpoint_path={args.resume_checkpoint_path}",
        f"training.resume_checkpoint_format={args.resume_checkpoint_format}",
        "training.load_part=params",
        f"training.global_batch_size={args.global_batch_size}",
        f"training.seq_length={args.seq_length}",
        f"training.n_data_parallel={args.n_data_parallel}",
        f"training.n_state_parallel={args.n_state_parallel}",
        f"deploy_paths.checkpoint={args.checkpoint_root}",
        f"deploy_paths.data.books3={args.books_root}",
    ]
    if args.dclm_root:
        cmd.extend([f"deploy_paths.data.dclm_filter_8k={args.dclm_root}"])
    return cmd


def _run_faithful_gate(args: argparse.Namespace, artifact_dir: Path) -> dict[str, object]:
    log_path = artifact_dir / "faithful_gate.log"
    result_path = artifact_dir / "faithful_gate.result.json"
    all_devices = _visible_devices()
    total_devices = int(args.n_data_parallel) * int(args.n_state_parallel)
    if total_devices > len(all_devices):
        result = {
            "status": "skipped",
            "returncode": 1,
            "timed_out": False,
            "checkpoint_written": False,
            "first_metric_seen": False,
            "completed_steps": 0,
            "peak_gpu_memory_mib": {},
            "reason": f"only {len(all_devices)} visible devices available",
            "created_at_utc": utc_now_iso(),
            "log_path": str(log_path),
        }
        _write_json(result_path, result)
        return result

    selected_devices = all_devices[:total_devices]
    cmd = _faithful_gate_command(args)
    print("$ " + shlex.join(cmd), flush=True)
    if args.dry_run:
        result = {
            "status": "dry_run",
            "returncode": 0,
            "timed_out": False,
            "checkpoint_written": False,
            "first_metric_seen": False,
            "completed_steps": 0,
            "peak_gpu_memory_mib": {},
            "visible_devices": selected_devices,
            "command": cmd,
            "created_at_utc": utc_now_iso(),
            "log_path": str(log_path),
        }
        _write_json(result_path, result)
        return result

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_devices)
    peak_memory_by_gpu: dict[str, int] = {}
    timed_out = False
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=args.repo_root.resolve(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        while proc.poll() is None:
            snapshot = _gpu_snapshot()
            for row in snapshot:
                if "index" not in row:
                    continue
                gpu_idx = str(row["index"])
                peak_memory_by_gpu[gpu_idx] = max(
                    int(row["memory_used_mib"]),
                    peak_memory_by_gpu.get(gpu_idx, 0),
                )
            if time.time() - started > args.timeout_seconds:
                timed_out = True
                proc.kill()
                proc.wait()
                break
            time.sleep(2.0)
    returncode = 124 if timed_out else int(proc.returncode)

    checkpoint_dir = args.checkpoint_root.expanduser().resolve() / args.paper_run_id / args.run_id
    run_dir = args.exp_dir.expanduser().resolve() / args.paper_run_id / args.stage_id / args.run_id
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_written = (checkpoint_dir / "latest.json").exists()
    first_metrics, latest_metrics = _read_metrics(metrics_path)
    first_metric_seen = bool(first_metrics)
    latest_step = int(latest_metrics.get("step", -1)) if latest_metrics else -1
    completed_steps = latest_step + 1 if latest_step >= 0 else 0
    passed = (
        returncode == 0
        and checkpoint_written
        and first_metric_seen
        and latest_step >= max(0, args.steps - 1)
    )
    result = {
        "status": "passed" if passed else ("timeout" if timed_out else "failed"),
        "returncode": returncode,
        "timed_out": timed_out,
        "checkpoint_written": checkpoint_written,
        "first_metric_seen": first_metric_seen,
        "first_metric_step": int(first_metrics.get("step", -1)) if first_metrics else None,
        "completed_steps": completed_steps,
        "latest_step": latest_step,
        "latest_loss_ce": latest_metrics.get("loss_ce"),
        "latest_train_step_seconds": latest_metrics.get("train_step_seconds"),
        "peak_gpu_memory_mib": peak_memory_by_gpu,
        "visible_devices": selected_devices,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "log_path": str(log_path),
        "created_at_utc": utc_now_iso(),
    }
    _write_json(result_path, result)
    return result


def _run_secondary_diagnosis(args: argparse.Namespace, artifact_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    requested = [int(part.strip()) for part in args.device_counts.split(",") if part.strip()]
    all_devices = _visible_devices()
    for count in requested:
        if count > len(all_devices):
            rows.append(
                {
                    "device_count": count,
                    "status": "skipped",
                    "reason": f"only {len(all_devices)} visible devices available",
                }
            )
            continue
        visible = all_devices[:count]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
        diag_dir = artifact_dir / f"devices_{count}"
        cmd = [
            UV_EXECUTABLE,
            "run",
            "--exact",
            "python",
            "scripts/40_diagnose_125m_32k_fa_oom.py",
            "--artifact-dir",
            str(diag_dir),
            "--experiment",
            "125m/pretrained/ext-125m-swa-32K-from-fa",
            "--exp-dir",
            str(args.exp_dir),
            "--exp-folder",
            args.exp_folder,
            "--exp-name",
            f"ext-125m-swa-32K-local-diag-{count}gpu",
            "--stage-id",
            f"S1_125M_DIAG_{count}GPU",
            "--run-id",
            f"ext-125m-swa-32K-local-diag-{count}gpu",
            "--checkpoint-root",
            str(args.checkpoint_root),
            "--books-root",
            str(args.books_root),
            "--resume-checkpoint-path",
            str(args.resume_checkpoint_path),
            "--resume-checkpoint-format",
            args.resume_checkpoint_format,
            "--load-part",
            "params",
            "--global-batch-size",
            str(args.global_batch_size),
            "--seq-length",
            str(args.seq_length),
            "--n-data-parallel",
            str(count),
            "--n-state-parallel",
            "1",
            "--total-steps",
            str(args.steps),
            "--save-milestone-freq",
            str(args.save_milestone_freq),
        ]
        if args.dclm_root:
            cmd.extend(["--dclm-root", str(args.dclm_root)])
        print("$ " + shlex.join(cmd), flush=True)
        if args.dry_run:
            rows.append(
                {
                    "device_count": count,
                    "status": "dry_run",
                    "visible_devices": visible,
                    "command": cmd,
                }
            )
            continue
        try:
            proc = subprocess.run(
                cmd,
                cwd=args.repo_root.resolve(),
                check=False,
                text=True,
                env=env,
                timeout=max(600, args.timeout_seconds),
            )
            returncode = int(proc.returncode)
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True
        compile_result_path = diag_dir / "compile_result.json"
        execute_result_path = diag_dir / "execute_result.json"
        compile_result = json.loads(compile_result_path.read_text(encoding="utf-8")) if compile_result_path.exists() else {}
        execute_result = json.loads(execute_result_path.read_text(encoding="utf-8")) if execute_result_path.exists() else {}
        rows.append(
            {
                "device_count": count,
                "visible_devices": visible,
                "returncode": returncode,
                "timed_out": timed_out,
                "compile_status": compile_result.get("status"),
                "execute_status": execute_result.get("status"),
                "artifact_dir": str(diag_dir),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifact_dir = (args.artifact_root / "local_125m_32k_swa_probe").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    faithful_gate = _run_faithful_gate(args, artifact_dir)
    secondary_rows: list[dict[str, object]] = []

    if args.dry_run:
        status = "dry_run"
        classification = "dry_run"
        exit_code = 0
    else:
        faithful_pass = str(faithful_gate.get("status", "")).strip() == "passed"
        if faithful_pass:
            status = "succeeded"
            classification = "faithful_gate_passed"
            exit_code = 0
        else:
            secondary_rows = _run_secondary_diagnosis(args, artifact_dir)
            smaller_pass = any(
                int(row.get("device_count", 0)) in {1, 2}
                and row.get("returncode") == 0
                and row.get("execute_status") == "ok"
                for row in secondary_rows
                if isinstance(row, dict)
            )
            status = "failed"
            if smaller_pass:
                classification = "faithful_gate_failed_smaller_topologies_passed"
                exit_code = 2
            else:
                classification = "faithful_gate_failed_all_secondary_topologies_failed"
                exit_code = 3

    summary = {
        "schema_version": "1.0",
        "status": status,
        "classification": classification,
        "paper_run_id": args.paper_run_id,
        "exp_folder": args.exp_folder,
        "created_at_utc": utc_now_iso(),
        "faithful_gate": faithful_gate,
        "secondary_rows": secondary_rows,
        "repo_root": str(repo_root),
    }
    _write_json(artifact_dir / "summary.json", summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
