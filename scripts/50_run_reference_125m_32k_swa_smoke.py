#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
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
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--exp-folder", default="reference_smokes")
    parser.add_argument("--exp-name", default="ext-125m-swa-32K-from-fa-ref-smoke")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ref_root = repo_root / "ttte2e_reference" / "e2e"
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
        "train",
        "+deploy=interactive",
        "+experiment=125m/extension/ext-125m-e2e-32K",
        f"training.checkpoint_path={args.checkpoint_root}",
        f"training.exp_dir={args.exp_dir}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_name={args.exp_name}",
        f"training.total_steps={args.steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        f"training.global_batch_size={args.global_batch_size}",
        "training.load_part=params",
        "training.log_wandb=false",
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
        return 0

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
            )
    except subprocess.TimeoutExpired:
        _write_json(
            log_path.with_suffix(".result.json"),
            {
                "status": "timeout",
                "timeout_seconds": int(args.timeout_seconds),
                "log_path": str(log_path),
            },
        )
        return 124

    _write_json(
        log_path.with_suffix(".result.json"),
        {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": int(proc.returncode),
            "log_path": str(log_path),
        },
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
