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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a faithful 2-step reference S3 125M pretrain smoke on the read-only "
            "ttte2e_reference snapshot."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--exp-folder", default="reference_s3_smokes")
    parser.add_argument("--exp-name", default="pretrain-125m-e2e-ref-smoke")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
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
        / "reference_125m_s3_pretrain_smoke.log"
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
        "+experiment=125m/pretrain/pretrain-125m-e2e",
        f"training.checkpoint_path={args.checkpoint_root}",
        f"training.exp_dir={args.exp_dir}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_name={args.exp_name}",
        f"training.total_steps={args.steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        f"training.global_batch_size={args.global_batch_size}",
        "training.log_wandb=false",
        "training.wandb_entity=none",
        "training.wandb_project=none",
        "training.wandb_key=env",
        f"deploy_paths.data.dclm_filter_8k={args.dclm_root}",
        f"deploy_paths.checkpoint={args.checkpoint_root}",
    ]

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
        _write_json(
            log_path.with_suffix(".result.json"),
            {
                "status": "dry_run",
                "log_path": str(log_path),
                "checkpoint_written": False,
            },
        )
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
        returncode = int(proc.returncode)
        status = "ok" if proc.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        returncode = 124
        status = "timeout"

    checkpoint_dir = args.checkpoint_root.expanduser().resolve() / args.exp_folder / args.exp_name
    checkpoint_written = (checkpoint_dir / "latest.json").exists()
    _write_json(
        log_path.with_suffix(".result.json"),
        {
            "status": status,
            "returncode": returncode,
            "log_path": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_written": checkpoint_written,
        },
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
