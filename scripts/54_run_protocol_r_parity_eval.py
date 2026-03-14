#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ttt.research.author_checkpoints import load_env_file


CANONICAL_PAPER_RUN_ID = "protocol_r_125m_main_v1"


@dataclass(frozen=True)
class StageRef:
    stage_id: str
    run_id: str


@dataclass(frozen=True)
class EvalJob:
    name: str
    stage_id: str
    run_id: str
    datasets: str
    contexts: str
    canonical_for_stage: bool = False


RESTORE_TARGETS = {
    "S0_PRETRAIN_FA_125M": StageRef(stage_id="S0_PRETRAIN_FA_125M", run_id="pretrain-125m-fa"),
    "S0_125M": StageRef(stage_id="S0_125M", run_id="ext-125m-fa-32K"),
}


EVAL_PROFILES: dict[str, list[EvalJob]] = {
    "s0_pair": [
        EvalJob(
            name="pretrain_books3_8k_diag",
            stage_id="S0_PRETRAIN_FA_125M",
            run_id="pretrain-125m-fa",
            datasets="books3",
            contexts="8192",
        ),
        EvalJob(
            name="ext_books3_8k_diag",
            stage_id="S0_125M",
            run_id="ext-125m-fa-32K",
            datasets="books3",
            contexts="8192",
        ),
        EvalJob(
            name="pretrain_dclm_8k_canonical",
            stage_id="S0_PRETRAIN_FA_125M",
            run_id="pretrain-125m-fa",
            datasets="dclm_filter_8k",
            contexts="8192",
            canonical_for_stage=True,
        ),
        EvalJob(
            name="ext_books3_32k_canonical",
            stage_id="S0_125M",
            run_id="ext-125m-fa-32K",
            datasets="books3",
            contexts="32768",
            canonical_for_stage=True,
        ),
    ],
}


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
    raise FileNotFoundError("Could not locate `uv`.")


UV_EXECUTABLE = _resolve_uv_executable()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _redact_cmd(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    for part in cmd:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        redacted.append(part)
        if part == "--token":
            skip_next = True
    return redacted


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print("$ " + shlex.join(_redact_cmd(cmd)), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False, cwd=cwd).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore canonical Protocol R stages from HF, run parity jax_eval bundles, "
            "re-export canonical eval artifacts to HF, and write a comparison summary."
        )
    )
    parser.add_argument("--profile", default="s0_pair", choices=sorted(EVAL_PROFILES))
    parser.add_argument("--paper-run-id", default=CANONICAL_PAPER_RUN_ID)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--overwrite-restores", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _restore_stage(repo_root: Path, args: argparse.Namespace, ref: StageRef) -> int:
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/46_restore_stage_from_hf.py",
        "--repo-id",
        args.repo_id,
        "--token",
        args.token,
        "--source-paper-run-id",
        args.paper_run_id,
        "--source-stage-id",
        ref.stage_id,
        "--source-run-id",
        ref.run_id,
        "--target-paper-run-id",
        args.paper_run_id,
        "--target-stage-id",
        ref.stage_id,
        "--target-run-id",
        ref.run_id,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
    ]
    if args.overwrite_restores:
        cmd.append("--overwrite")
    return _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def _run_eval_job(repo_root: Path, args: argparse.Namespace, job: EvalJob, reports_root: Path) -> dict:
    job_root = reports_root / "jobs" / job.name
    summary_json = job_root / "summary.json"
    summary_csv = job_root / "summary.csv"
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/34_eval_matrix_jax.py",
        "--paper-run-id",
        args.paper_run_id,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--exp-folder",
        args.paper_run_id,
        "--stages",
        job.stage_id,
        "--runs",
        job.run_id,
        "--datasets",
        job.datasets,
        "--contexts",
        job.contexts,
        "--dclm-root",
        str(args.dclm_root),
        "--books-root",
        str(args.books_root),
        "--eval-split",
        args.eval_split,
        "--eval-batches",
        str(args.eval_batches),
        "--eval-id",
        job.name,
        "--summary-json",
        str(summary_json),
        "--summary-csv",
        str(summary_csv),
    ]
    if args.eval_batch_size > 0:
        cmd.extend(["--eval-batch-size", str(args.eval_batch_size)])
    rc = _run(cmd, cwd=repo_root, dry_run=args.dry_run)
    payload: dict = {
        "job_name": job.name,
        "stage_id": job.stage_id,
        "run_id": job.run_id,
        "datasets": job.datasets,
        "contexts": job.contexts,
        "canonical_for_stage": job.canonical_for_stage,
        "returncode": rc,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
    }
    if rc != 0 or args.dry_run:
        return payload

    run_root = args.exp_dir / args.paper_run_id / job.stage_id / job.run_id
    eval_manifest = run_root / "eval_manifest.json"
    raw_json = run_root / "eval_parity_raw.json"
    raw_csv = run_root / "eval_parity_raw.csv"
    jax_eval_root = run_root / "jax_eval"
    _copy_if_exists(eval_manifest, job_root / "eval_manifest_snapshot.json")
    _copy_if_exists(raw_json, job_root / "eval_parity_raw.json")
    _copy_if_exists(raw_csv, job_root / "eval_parity_raw.csv")
    _copy_if_exists(jax_eval_root, job_root / "jax_eval")
    if summary_json.exists():
        payload["summary"] = _load_json(summary_json)
    if eval_manifest.exists():
        payload["eval_manifest"] = _load_json(eval_manifest)
    return payload


def _export_stage(repo_root: Path, args: argparse.Namespace, ref: StageRef) -> int:
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/40_export_stage_to_hf.py",
        "--paper-run-id",
        args.paper_run_id,
        "--stage-id",
        ref.stage_id,
        "--run-id",
        ref.run_id,
        "--repo-id",
        args.repo_id,
        "--token",
        args.token,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
    ]
    return _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()
    args.exp_dir = args.exp_dir.expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.dclm_root = args.dclm_root.expanduser().resolve()
    args.books_root = args.books_root.expanduser().resolve()
    args.token = (
        args.token.strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )

    reports_root = repo_root / "reports" / "paper" / args.paper_run_id / "eval_bundles" / args.profile
    jobs = EVAL_PROFILES[args.profile]

    restore_stage_ids = sorted({job.stage_id for job in jobs})
    restore_rows: list[dict] = []
    for stage_id in restore_stage_ids:
        ref = RESTORE_TARGETS[stage_id]
        rc = _restore_stage(repo_root, args, ref)
        restore_rows.append(
            {
                "stage_id": ref.stage_id,
                "run_id": ref.run_id,
                "returncode": rc,
            }
        )
        if rc != 0:
            _write_json(
                reports_root / "bundle_summary.json",
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "profile": args.profile,
                    "status": "failed",
                    "restore_rows": restore_rows,
                    "job_rows": [],
                    "export_rows": [],
                },
            )
            return rc

    job_rows: list[dict] = []
    for job in jobs:
        payload = _run_eval_job(repo_root, args, job, reports_root)
        job_rows.append(payload)
        if payload["returncode"] != 0:
            _write_json(
                reports_root / "bundle_summary.json",
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "profile": args.profile,
                    "status": "failed",
                    "restore_rows": restore_rows,
                    "job_rows": job_rows,
                    "export_rows": [],
                },
            )
            return int(payload["returncode"])

    export_rows: list[dict] = []
    exported_stage_ids: set[str] = set()
    for job in jobs:
        if not job.canonical_for_stage or job.stage_id in exported_stage_ids:
            continue
        ref = RESTORE_TARGETS[job.stage_id]
        rc = _export_stage(repo_root, args, ref)
        export_rows.append(
            {
                "stage_id": ref.stage_id,
                "run_id": ref.run_id,
                "returncode": rc,
            }
        )
        if rc != 0:
            _write_json(
                reports_root / "bundle_summary.json",
                {
                    "schema_version": "1.0",
                    "paper_run_id": args.paper_run_id,
                    "profile": args.profile,
                    "status": "failed",
                    "restore_rows": restore_rows,
                    "job_rows": job_rows,
                    "export_rows": export_rows,
                },
            )
            return rc
        exported_stage_ids.add(job.stage_id)

    summary = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "profile": args.profile,
        "status": "succeeded" if not args.dry_run else "dry_run",
        "restore_rows": restore_rows,
        "job_rows": job_rows,
        "export_rows": export_rows,
    }
    _write_json(reports_root / "bundle_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
