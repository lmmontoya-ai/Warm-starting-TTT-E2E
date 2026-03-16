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

from ttt.research.cuda_preflight import prepare_cuda_runtime_env
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


def _parse_topologies(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        data_parallel, state_parallel = item.split(":")
        pairs.append((int(data_parallel), int(state_parallel)))
    if not pairs:
        raise ValueError("No topologies were provided.")
    return pairs


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
            "Run faithful local 125M S3 2-step probes across the 8-GPU topologies that "
            "matter for the current scaling blocker."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/s3_scaling_diag"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--exp-folder", default="protocol_r_125m_main_v1_s3_diag")
    parser.add_argument("--paper-run-id", default="protocol_r_125m_main_v1_s3_diag")
    parser.add_argument("--topologies", default="8:1,4:2,2:4")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--allow-incompatible-driver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifact_root = (args.artifact_root / "local_125m_s3_scaling_probe").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    runtime_env, cuda_preflight = prepare_cuda_runtime_env()
    topologies = _parse_topologies(args.topologies)
    visible_devices = _visible_devices()
    rows: list[dict[str, object]] = []

    for index, (n_data_parallel, n_state_parallel) in enumerate(topologies):
        total_devices = n_data_parallel * n_state_parallel
        label = f"dp{n_data_parallel}_sp{n_state_parallel}"
        faithful_topology = index == 0
        if total_devices > len(visible_devices):
            rows.append(
                {
                    "topology": label,
                    "n_data_parallel": n_data_parallel,
                    "n_state_parallel": n_state_parallel,
                    "faithful_topology": faithful_topology,
                    "status": "skipped",
                    "reason": f"only {len(visible_devices)} visible devices available",
                }
            )
            continue

        selected_devices = visible_devices[:total_devices]
        env = runtime_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_devices)
        artifact_dir = artifact_root / label
        run_id = f"pretrain-125m-e2e-{label}"
        stage_id = f"S3_PRETRAIN_E2E_125M_DIAG_{label.upper()}"
        log_path = artifact_dir / "run.log"
        checkpoint_dir = args.checkpoint_root.expanduser().resolve() / args.paper_run_id / run_id
        run_dir = args.exp_dir.expanduser().resolve() / args.paper_run_id / stage_id / run_id
        metrics_path = run_dir / "metrics.jsonl"
        cmd = [
            UV_EXECUTABLE,
            "run",
            "--exact",
            "train",
            "+deploy=interactive",
            "+experiment=125m/pretrain/pretrain-125m-e2e",
            "training.runtime_mode=jax_train",
            "training.log_wandb=false",
            "training.wandb_entity=none",
            "training.wandb_project=none",
            "training.wandb_key=env",
            f"training.exp_dir={args.exp_dir}",
            f"training.exp_folder={args.exp_folder}",
            f"training.exp_name={run_id}",
            f"training.paper_run_id={args.paper_run_id}",
            f"training.stage_id={stage_id}",
            f"training.run_id={run_id}",
            f"training.total_steps={args.steps}",
            f"training.save_milestone_freq={args.save_milestone_freq}",
            f"training.checkpoint_path={args.checkpoint_root}",
            f"training.global_batch_size={args.global_batch_size}",
            f"training.n_data_parallel={n_data_parallel}",
            f"training.n_state_parallel={n_state_parallel}",
            f"deploy_paths.checkpoint={args.checkpoint_root}",
            f"deploy_paths.data.dclm_filter_8k={args.dclm_root}",
        ]

        print("$ " + shlex.join(cmd), flush=True)
        if args.dry_run:
            rows.append(
                {
                    "topology": label,
                    "n_data_parallel": n_data_parallel,
                    "n_state_parallel": n_state_parallel,
                    "faithful_topology": faithful_topology,
                    "status": "dry_run",
                    "command": cmd,
                    "visible_devices": selected_devices,
                    "cuda_preflight": cuda_preflight.as_dict(),
                }
            )
            continue

        if cuda_preflight.status != "ok" and not args.allow_incompatible_driver:
            rows.append(
                {
                    "topology": label,
                    "n_data_parallel": n_data_parallel,
                    "n_state_parallel": n_state_parallel,
                    "faithful_topology": faithful_topology,
                    "visible_devices": selected_devices,
                    "returncode": 1,
                    "timed_out": False,
                    "checkpoint_written": False,
                    "first_metric_seen": False,
                    "first_metric_step": None,
                    "latest_step": -1,
                    "step0_train_step_seconds": None,
                    "latest_train_step_seconds": None,
                    "latest_loss_ce": None,
                    "gpu_before": _gpu_snapshot(),
                    "gpu_after": _gpu_snapshot(),
                    "peak_memory_mib_by_gpu": {},
                    "log_path": str(log_path),
                    "run_dir": str(run_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                    "status": "failed_preflight",
                    "reason": "CUDA runtime preflight failed",
                    "cuda_preflight": cuda_preflight.as_dict(),
                }
            )
            continue

        artifact_dir.mkdir(parents=True, exist_ok=True)
        before = _gpu_snapshot()
        peak_memory_by_gpu: dict[str, int] = {}
        timed_out = False
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
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
        after = _gpu_snapshot()
        step0_metrics, latest_metrics = _read_metrics(metrics_path)
        checkpoint_written = (checkpoint_dir / "latest.json").exists()
        first_metric_seen = bool(step0_metrics)
        first_metric_step = int(step0_metrics.get("step", -1)) if step0_metrics else None
        latest_step = int(latest_metrics.get("step", -1) or -1) if latest_metrics else -1
        passed = (
            returncode == 0
            and checkpoint_written
            and first_metric_seen
            and latest_step >= max(0, args.steps - 1)
        )
        row = {
            "topology": label,
            "n_data_parallel": n_data_parallel,
            "n_state_parallel": n_state_parallel,
            "faithful_topology": faithful_topology,
            "visible_devices": selected_devices,
            "returncode": returncode,
            "timed_out": timed_out,
            "checkpoint_written": checkpoint_written,
            "first_metric_seen": first_metric_seen,
            "first_metric_step": first_metric_step,
            "latest_step": latest_step,
            "step0_train_step_seconds": step0_metrics.get("train_step_seconds"),
            "latest_train_step_seconds": latest_metrics.get("train_step_seconds"),
            "latest_loss_ce": latest_metrics.get("loss_ce"),
            "gpu_before": before,
            "gpu_after": after,
            "peak_memory_mib_by_gpu": peak_memory_by_gpu,
            "log_path": str(log_path),
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "status": "passed" if passed else ("timeout" if timed_out else "failed"),
            "cuda_preflight": cuda_preflight.as_dict(),
        }
        rows.append(row)
        _write_json(artifact_dir / "result.json", row)

    if args.dry_run:
        classification = "dry_run"
        status = "dry_run"
        exit_code = 0
    else:
        current_label = f"dp{topologies[0][0]}_sp{topologies[0][1]}"
        current_row = next((row for row in rows if row.get("topology") == current_label), {})
        current_pass = str(current_row.get("status", "")) == "passed"
        any_exploratory_pass = any(
            str(row.get("status", "")) == "passed" and not bool(row.get("faithful_topology", False))
            for row in rows
        )
        if current_pass:
            classification = "faithful_topology_passed"
            status = "succeeded"
            exit_code = 0
        elif any_exploratory_pass:
            classification = "exploratory_topology_only"
            status = "failed"
            exit_code = 2
        else:
            classification = "all_topologies_failed"
            status = "failed"
            exit_code = 3

    summary = {
        "schema_version": "1.0",
        "status": status,
        "classification": classification,
        "paper_run_id": args.paper_run_id,
        "exp_folder": args.exp_folder,
        "created_at_utc": utc_now_iso(),
        "rows": rows,
        "repo_root": str(repo_root),
        "cuda_preflight": cuda_preflight.as_dict(),
    }
    _write_json(artifact_root / "summary.json", summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
