#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False, cwd=cwd).returncode


def _lock_path(repo_root: Path, paper_run_id: str) -> Path:
    return repo_root / "reports" / "paper" / paper_run_id / "protocol_r_local_gate.lock"


def _summary_path(repo_root: Path, paper_run_id: str) -> Path:
    return repo_root / "reports" / "paper" / paper_run_id / "protocol_r_local_gate_summary.json"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(repo_root: Path, paper_run_id: str) -> tuple[Path, bool]:
    summary_path = _summary_path(repo_root, paper_run_id)
    if summary_path.exists():
        return summary_path, False

    lock_path = _lock_path(repo_root, paper_run_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(lock_payload.get("pid", -1))
        except Exception:
            pid = -1
        if _pid_alive(pid):
            return lock_path, False
        lock_path.unlink(missing_ok=True)

    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "created_at": _utc_slug()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock_path, True


def _read_latest_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local-runtime Protocol R gate for 125M 32K FA: a 2-step smoke "
            "plus a 6-8 step benchmark, followed by parity jax_eval on the smoke run."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dclm-root", type=Path, default=None)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--resume-checkpoint-path", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-format", default="orbax")
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--bench-steps", type=int, default=8)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--paper-run-id", default=None)
    parser.add_argument("--exp-folder", default="protocol_r_local_gate")
    parser.add_argument("--wandb-key", default="env")
    parser.add_argument("--wandb-entity", default="none")
    parser.add_argument("--wandb-project", default="none")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _train_cmd(
    *,
    exp_name: str,
    paper_run_id: str,
    stage_id: str,
    total_steps: int,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--exact",
        "train",
        "+deploy=interactive",
        "+experiment=125m/extension/ext-125m-fa-32K",
        f"training.runtime_mode=jax_train",
        f"training.exp_dir={args.repo_root / 'experiments'}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_name={exp_name}",
        f"training.paper_run_id={paper_run_id}",
        f"training.stage_id={stage_id}",
        f"training.run_id={exp_name}",
        f"training.total_steps={total_steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        "training.load_part=params",
        f"training.resume_checkpoint_path={args.resume_checkpoint_path.expanduser().resolve()}",
        f"training.resume_checkpoint_format={args.resume_checkpoint_format}",
        f"training.checkpoint_path={args.checkpoint_root.expanduser().resolve()}",
        f"deploy_paths.checkpoint={args.checkpoint_root.expanduser().resolve()}",
        f"deploy_paths.data.books3={args.books_root.expanduser().resolve()}",
        f"training.global_batch_size={args.global_batch_size}",
        f"training.wandb_key={args.wandb_key}",
        f"training.wandb_entity={args.wandb_entity}",
        f"training.wandb_project={args.wandb_project}",
    ]
    if args.dclm_root is not None:
        cmd.append(f"deploy_paths.data.dclm_filter_8k={args.dclm_root.expanduser().resolve()}")
    return cmd


def _eval_cmd(*, paper_run_id: str, args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "--exact",
        "python",
        "scripts/34_eval_matrix_jax.py",
        "--paper-run-id",
        paper_run_id,
        "--exp-dir",
        str((args.repo_root / "experiments").resolve()),
        "--checkpoint-root",
        str(args.checkpoint_root.expanduser().resolve()),
        "--exp-folder",
        args.exp_folder,
        "--contexts",
        "32768",
        "--datasets",
        "books3",
        "--dclm-root",
        str(args.dclm_root.expanduser().resolve()) if args.dclm_root is not None else str(args.books_root.expanduser().resolve()),
        "--books-root",
        str(args.books_root.expanduser().resolve()),
        "--eval-split",
        "val",
        "--eval-batches",
        str(args.eval_batches),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    paper_run_id = args.paper_run_id or f"protocol_r_local_gate_b{args.global_batch_size}_{_utc_slug()}"
    lock_path, lock_acquired = _acquire_lock(repo_root, paper_run_id)
    if not lock_acquired:
        print(f"Protocol R gate already running or completed for {paper_run_id}: {lock_path}")
        return 0

    smoke_run = f"ext-125m-fa-32K-b{args.global_batch_size}-smoke{args.smoke_steps}"
    bench_run = f"ext-125m-fa-32K-b{args.global_batch_size}-bench{args.bench_steps}"

    rows: list[dict[str, object]] = []
    try:
        for step_id, cmd in [
            (
                "train_smoke",
                _train_cmd(
                    exp_name=smoke_run,
                    paper_run_id=paper_run_id,
                    stage_id="S0_125M_PROTOCOL_R_SMOKE",
                    total_steps=args.smoke_steps,
                    args=args,
                ),
            ),
            ("eval_smoke", _eval_cmd(paper_run_id=paper_run_id, args=args)),
            (
                "train_bench",
                _train_cmd(
                    exp_name=bench_run,
                    paper_run_id=paper_run_id,
                    stage_id="S0_125M_PROTOCOL_R_BENCH",
                    total_steps=args.bench_steps,
                    args=args,
                ),
            ),
        ]:
            rc = _run(cmd, cwd=repo_root, dry_run=args.dry_run)
            rows.append({"step_id": step_id, "command": cmd, "returncode": rc})
            if rc != 0:
                break

        smoke_dir = repo_root / "experiments" / paper_run_id / "S0_125M_PROTOCOL_R_SMOKE" / smoke_run
        bench_dir = repo_root / "experiments" / paper_run_id / "S0_125M_PROTOCOL_R_BENCH" / bench_run
        summary = {
            "schema_version": "1.0",
            "paper_run_id": paper_run_id,
            "global_batch_size": args.global_batch_size,
            "smoke_steps": args.smoke_steps,
            "bench_steps": args.bench_steps,
            "rows": rows,
            "smoke_metrics": _read_latest_metrics(smoke_dir),
            "bench_metrics": _read_latest_metrics(bench_dir),
            "smoke_eval_manifest": str(smoke_dir / "eval_manifest.json"),
        }
        summary_path = _summary_path(repo_root, paper_run_id)
        _write_json(summary_path, summary)
        print(f"Wrote local Protocol R gate summary: {summary_path}")
        if any(int(row.get("returncode", 1)) != 0 for row in rows):
            return 1
        return 0
    finally:
        if lock_acquired:
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
