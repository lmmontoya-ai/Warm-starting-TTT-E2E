#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _gpu_snapshot() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except Exception as exc:
        return [{"error": repr(exc)}]
    rows: list[dict[str, str]] = []
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": parts[0],
                "memory_used_mib": parts[1],
                "memory_total_mib": parts[2],
                "utilization_gpu_pct": parts[3],
            }
        )
    return rows


def _visible_devices() -> list[str]:
    current = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if current:
        return [part.strip() for part in current.split(",") if part.strip()]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return ["0"]
    return [str(i) for i, _ in enumerate(proc.stdout.splitlines())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local 125M 32K SWA diagnostic at 1/2/8 visible devices, recording "
            "compile/execute artifacts and GPU memory snapshots for each topology."
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
    parser.add_argument("--exp-folder", default="oom_diagnosis")
    parser.add_argument("--device-counts", default="1,2,8")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=32768)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifact_root = (args.artifact_root / "local_125m_32k_swa_probe").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    all_devices = _visible_devices()
    requested = [int(part.strip()) for part in args.device_counts.split(",") if part.strip()]
    rows: list[dict[str, object]] = []

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
        artifact_dir = artifact_root / f"devices_{count}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
        before = _gpu_snapshot()
        cmd = [
            UV_EXECUTABLE,
            "run",
            "--exact",
            "python",
            "scripts/40_diagnose_125m_32k_fa_oom.py",
            "--artifact-dir",
            str(artifact_dir),
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
        print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
        if args.dry_run:
            rows.append({"device_count": count, "status": "dry_run", "visible_devices": visible, "command": cmd})
            continue
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                check=False,
                text=True,
                env=env,
                timeout=args.timeout_seconds,
            )
            returncode = int(proc.returncode)
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True
        after = _gpu_snapshot()
        compile_result_path = artifact_dir / "compile_result.json"
        execute_result_path = artifact_dir / "execute_result.json"
        compile_result = json.loads(compile_result_path.read_text()) if compile_result_path.exists() else {}
        execute_result = json.loads(execute_result_path.read_text()) if execute_result_path.exists() else {}
        rows.append(
            {
                "device_count": count,
                "visible_devices": visible,
                "returncode": returncode,
                "timed_out": timed_out,
                "compile_status": compile_result.get("status"),
                "execute_status": execute_result.get("status"),
                "gpu_before": before,
                "gpu_after": after,
                "artifact_dir": str(artifact_dir),
            }
        )

    if args.dry_run:
        classification = "dry_run"
        exit_code = 0
    else:
        passed_counts = [
            row["device_count"]
            for row in rows
            if row.get("returncode") == 0 and row.get("execute_status") == "ok"
        ]
        classification = "safe_to_run_full_s1"
        exit_code = 0
        if 8 not in passed_counts:
            smaller = {1, 2}.intersection(passed_counts)
            if smaller:
                classification = "local_bug_multi_device_swa"
                exit_code = 2
            else:
                classification = "local_swa_probe_failed_all_topologies"
                exit_code = 3

    summary = {
        "schema_version": "1.0",
        "classification": classification,
        "rows": rows,
        "repo_root": str(repo_root),
        "python": sys.executable,
    }
    _write_json(artifact_root / "summary.json", summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
