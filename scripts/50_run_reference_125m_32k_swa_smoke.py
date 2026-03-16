#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
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
REFERENCE_NO_WANDB_ENTRYPOINT = (
    Path(__file__).resolve().parent / "58_run_reference_entrypoint_no_wandb.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
            "Run a reference 125M 32K SWA smoke that mirrors the local S1 control: "
            "same SWA architecture, same FA seed restore, same 32K books3 train slice, "
            "but with prime disabled and suffix_len=0."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--resume-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--n-data-parallel", type=int, default=8)
    parser.add_argument("--n-state-parallel", type=int, default=1)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--exp-folder", default="reference_smokes")
    parser.add_argument("--exp-name", default="ext-125m-swa-32K-from-fa-ref-smoke")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-incompatible-driver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ref_root = repo_root / "ttte2e_reference" / "e2e"
    runtime_env, cuda_preflight = prepare_cuda_runtime_env()
    log_path = args.log_path or (
        repo_root
        / "artifacts"
        / "reference_smokes"
        / args.exp_name
        / "reference_125m_32k_swa_smoke.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        UV_EXECUTABLE,
        "run",
        "--python",
        args.python_version,
        "--exact",
        "python",
        str(REFERENCE_NO_WANDB_ENTRYPOINT),
        "--reference-root",
        str(ref_root),
        "--module",
        "ttt.train",
        "--",
        "+deploy=interactive",
        "+experiment=125m/extension/ext-125m-e2e-32K",
        f"training.checkpoint_path={args.checkpoint_root}",
        f"training.exp_dir={args.exp_dir}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_name={args.exp_name}",
        f"training.total_steps={args.steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        f"training.global_batch_size={args.global_batch_size}",
        f"training.n_data_parallel={args.n_data_parallel}",
        f"training.n_state_parallel={args.n_state_parallel}",
        "training.load_part=params",
        "training.log_wandb=false",
        r"training.wandb_entity=${oc.env:WANDB_ENTITY}",
        r"training.wandb_project=${oc.env:WANDB_PROJECT}",
        r"training.wandb_key=${oc.env:WANDB_API_KEY}",
        "training.train_mode=pretrain",
        "training.spec_inner=[]",
        "model.prime=false",
        "model.suffix_len=0",
        f"checkpoint.resume_checkpoint_dir={args.resume_checkpoint_dir}",
        f"deploy_paths.data.books3={args.books_root}",
        f"deploy_paths.checkpoint={args.checkpoint_root}",
    ]
    if args.resume_step is not None:
        cmd.append(f"training.resume_step={args.resume_step}")

    manifest = {
        "cwd": str(ref_root),
        "command": cmd,
        "log_path": str(log_path),
        "timeout_seconds": int(args.timeout_seconds),
        "cuda_preflight": cuda_preflight.as_dict(),
    }
    _write_json(log_path.with_suffix(".manifest.json"), manifest)

    print(
        "$ (cd "
        + shlex.quote(str(ref_root))
        + " && "
        + " ".join(shlex.quote(part) for part in cmd)
        + ")"
    )
    if args.dry_run:
        _write_json(
            log_path.with_suffix(".result.json"),
            {
                "status": "dry_run",
                "returncode": 0,
                "log_path": str(log_path),
                "checkpoint_written": False,
                "first_metric_seen": False,
                "completed_steps": 0,
                "cuda_preflight": cuda_preflight.as_dict(),
                "created_at_utc": utc_now_iso(),
            },
        )
        return 0

    if cuda_preflight.status != "ok" and not args.allow_incompatible_driver:
        _write_json(
            log_path.with_suffix(".result.json"),
            {
                "status": "failed_preflight",
                "returncode": 1,
                "log_path": str(log_path),
                "checkpoint_written": False,
                "first_metric_seen": False,
                "completed_steps": 0,
                "cuda_preflight": cuda_preflight.as_dict(),
                "created_at_utc": utc_now_iso(),
            },
        )
        return 1

    try:
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd,
                cwd=ref_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                timeout=args.timeout_seconds,
                env=runtime_env,
            )
        returncode = int(proc.returncode)
        status = "succeeded" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        returncode = 124
        status = "timeout"

    checkpoint_dir = args.checkpoint_root.expanduser().resolve() / args.exp_folder / args.exp_name
    metrics_path = args.exp_dir.expanduser().resolve() / args.exp_folder / args.exp_name / "metrics.jsonl"
    checkpoint_written = (checkpoint_dir / "latest.json").exists()
    first_metrics, latest_metrics = _read_metrics(metrics_path)
    first_metric_seen = bool(first_metrics)
    first_metric_step = int(first_metrics.get("step", -1)) if first_metrics else None
    latest_step = int(latest_metrics.get("step", -1)) if latest_metrics else None
    completed_steps = 0
    if latest_step is not None and latest_step >= 0:
        completed_steps = latest_step + 1

    _write_json(
        log_path.with_suffix(".result.json"),
        {
            "status": status,
            "returncode": returncode,
            "log_path": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_path": str(metrics_path),
            "checkpoint_written": checkpoint_written,
            "first_metric_seen": first_metric_seen,
            "first_metric_step": first_metric_step,
            "completed_steps": completed_steps,
            "latest_step": latest_step,
            "cuda_preflight": cuda_preflight.as_dict(),
            "created_at_utc": utc_now_iso(),
        },
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
