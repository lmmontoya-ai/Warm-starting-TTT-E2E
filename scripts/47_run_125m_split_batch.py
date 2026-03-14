#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from ttt.research.author_checkpoints import load_env_file
from ttt.research.registry import load_registry


CANONICAL_PAPER_RUN_ID = "protocol_r_125m_main_v1"
BOOTSTRAP_SOURCE_PAPER_RUN_ID = "protocol_r_125m_h200_20260312a"
BOOTSTRAP_STAGE_ID = "S0_PRETRAIN_FA_125M"
BOOTSTRAP_RUN_ID = "pretrain-125m-fa"
PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE = 8
PROTOCOL_R_BASE_EXT_GLOBAL_BATCH_SIZE = 32
PROTOCOL_R_EFFECTIVE_EXT_STEPS = 480
DEFAULT_REGISTRY = Path("./configs/research/warmstart_registry.yaml")

BATCH_CONFIG = {
    "bootstrap_s0": {
        "parents": [],
        "gates": [],
        "stages": [],
    },
    "h200_s0": {
        "parents": [BOOTSTRAP_STAGE_ID],
        "gates": [],
        "stages": ["S0_125M"],
    },
    "h200_s1_diag": {
        "parents": [BOOTSTRAP_STAGE_ID],
        "gates": [],
        "stages": [],
    },
    "h100_b": {
        "parents": [BOOTSTRAP_STAGE_ID],
        "gates": ["S2_ADAPT_125M", "S3_PRETRAIN_E2E_125M"],
        "stages": ["S2_ADAPT_125M", "S3_PRETRAIN_E2E_125M"],
    },
    "h200_c": {
        "parents": ["S2_ADAPT_125M", "S3_PRETRAIN_E2E_125M"],
        "gates": ["S2_125M", "S3_125M"],
        "stages": ["S2_125M", "S3_125M"],
    },
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
    raise FileNotFoundError(
        "Could not locate the `uv` executable. Install uv or add it to PATH before running split batches."
    )


UV_EXECUTABLE = _resolve_uv_executable()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _env_default(name: str, fallback: str) -> str:
    value = str(os.environ.get(name, "")).strip()
    return value or fallback


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_dir(checkpoint_root: Path, paper_run_id: str, run_id: str) -> Path:
    return checkpoint_root / paper_run_id / run_id


def _experiment_dir(exp_dir: Path, paper_run_id: str, stage_id: str, run_id: str) -> Path:
    return exp_dir / paper_run_id / stage_id / run_id


def _checkpoint_exists(checkpoint_root: Path, paper_run_id: str, run_id: str) -> bool:
    return (_checkpoint_dir(checkpoint_root, paper_run_id, run_id) / "latest.json").exists()


def _stage_done(exp_dir: Path, checkpoint_root: Path, paper_run_id: str, stage_id: str, run_id: str) -> bool:
    result_path = _experiment_dir(exp_dir, paper_run_id, stage_id, run_id) / "run_result.json"
    if not result_path.exists():
        return False
    if not _checkpoint_exists(checkpoint_root, paper_run_id, run_id):
        return False
    try:
        payload = _load_json(result_path)
    except Exception:
        return False
    return str(payload.get("status", "")) in {"succeeded", "dry_run"}


def _protocol_r_stage(stage_id: str) -> bool:
    return stage_id in {"S0_125M", "S1_125M", "S2_125M", "S3_125M"}


def _latest_step(checkpoint_dir: Path) -> int:
    return int(_load_json(checkpoint_dir / "latest.json")["step"])


def _prune_checkpoint_history(checkpoint_dir: Path, *, keep_steps: int = 2) -> None:
    if not checkpoint_dir.exists():
        return
    step_dirs = []
    for child in checkpoint_dir.iterdir():
        if child.is_dir():
            try:
                step_dirs.append((int(child.name), child))
            except ValueError:
                continue
    if len(step_dirs) <= keep_steps:
        return
    step_dirs.sort()
    keep = {step for step, _ in step_dirs[-keep_steps:]}
    for step, step_dir in step_dirs:
        if step not in keep:
            shutil.rmtree(step_dir, ignore_errors=True)
    for metadata in checkpoint_dir.glob("step_metadata_*.json"):
        try:
            step = int(metadata.stem.split("_")[-1])
        except Exception:
            continue
        if step not in keep:
            metadata.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 125M Protocol R ladder as split hardware batches with HF-backed "
            "parent checkpoint restore and per-stage export."
        )
    )
    parser.add_argument("--batch", required=True, choices=sorted(BATCH_CONFIG))
    parser.add_argument("--paper-run-id", default=CANONICAL_PAPER_RUN_ID)
    parser.add_argument("--exp-folder", default="")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--dclm-root", type=Path, required=True)
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--wandb-entity", default=_env_default("WANDB_ENTITY", "none"))
    parser.add_argument("--wandb-project", default=_env_default("WANDB_PROJECT", "none"))
    parser.add_argument("--wandb-key", default="env")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain-steps", type=int, default=4800)
    parser.add_argument("--adapt-steps", type=int, default=480)
    parser.add_argument("--ext-steps", type=int, default=120)
    parser.add_argument("--save-milestone-freq", type=int, default=120)
    parser.add_argument("--gate-steps", type=int, default=2)
    parser.add_argument("--gate-save-milestone-freq", type=int, default=999)
    parser.add_argument("--allow-missing-fingerprints", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--stop-after-gates", action="store_true")
    parser.add_argument("--s1-reference-timeout-seconds", type=int, default=900)
    parser.add_argument("--s1-local-probe-timeout-seconds", type=int, default=600)
    parser.add_argument("--s1-local-probe-device-counts", default="1,2,8")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _restore_stage(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    source_paper_run_id: str,
    source_stage_id: str,
    source_run_id: str,
    target_stage_id: str,
    target_run_id: str,
) -> int:
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
        source_paper_run_id,
        "--source-stage-id",
        source_stage_id,
        "--source-run-id",
        source_run_id,
        "--target-paper-run-id",
        args.paper_run_id,
        "--target-stage-id",
        target_stage_id,
        "--target-run-id",
        target_run_id,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--overwrite",
    ]
    return _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def _restore_stage_if_exported(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    stage_id: str,
    run_id: str,
) -> int:
    return _restore_stage(
        repo_root=repo_root,
        args=args,
        source_paper_run_id=args.paper_run_id,
        source_stage_id=stage_id,
        source_run_id=run_id,
        target_stage_id=stage_id,
        target_run_id=run_id,
    )


def _export_stage(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    stage_id: str,
    run_id: str,
) -> int:
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/40_export_stage_to_hf.py",
        "--paper-run-id",
        args.paper_run_id,
        "--stage-id",
        stage_id,
        "--run-id",
        run_id,
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


def _train_gate_cmd(
    *,
    stage,
    stage_map,
    args: argparse.Namespace,
    exp_folder: str,
    gate_run_id: str,
    gate_stage_id: str,
) -> list[str]:
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "train",
        "+deploy=interactive",
        f"+experiment={stage.experiment}",
        "training.runtime_mode=jax_train",
        f"training.exp_dir={args.exp_dir.resolve()}",
        f"training.exp_folder={exp_folder}",
        f"training.exp_name={gate_run_id}",
        f"training.paper_run_id={args.paper_run_id}",
        f"training.stage_id={gate_stage_id}",
        f"training.run_id={gate_run_id}",
        f"training.total_steps={args.gate_steps}",
        f"training.save_milestone_freq={args.gate_save_milestone_freq}",
        f"training.checkpoint_path={args.checkpoint_root.resolve()}",
        f"deploy_paths.checkpoint={args.checkpoint_root.resolve()}",
        f"deploy_paths.data.dclm_filter_8k={args.dclm_root.resolve()}",
        f"deploy_paths.data.books3={args.books_root.resolve()}",
        f"training.wandb_key={args.wandb_key}",
        f"training.wandb_entity={args.wandb_entity}",
        f"training.wandb_project={args.wandb_project}",
    ]
    if _protocol_r_stage(stage.stage_id):
        cmd.append(f"training.global_batch_size={PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE}")
    parent_ids = list(stage.required_parent_checkpoint_ids)
    if parent_ids:
        parent_stage = stage_map[parent_ids[0]]
        parent_checkpoint = _checkpoint_dir(args.checkpoint_root, args.paper_run_id, parent_stage.exp_name)
        cmd.extend(
            [
                "training.load_part=params",
                f"training.resume_checkpoint_path={parent_checkpoint.resolve()}",
                "training.resume_checkpoint_format=orbax",
            ]
        )
    return cmd


def _run_gate(
    *,
    repo_root: Path,
    stage,
    stage_map,
    args: argparse.Namespace,
) -> int:
    gate_run_id = f"{stage.exp_name}-gate{args.gate_steps}"
    gate_stage_id = f"{stage.stage_id}_GATE"
    exp_folder = f"{args.exp_folder}_gates"
    cmd = _train_gate_cmd(
        stage=stage,
        stage_map=stage_map,
        args=args,
        exp_folder=exp_folder,
        gate_run_id=gate_run_id,
        gate_stage_id=gate_stage_id,
    )
    return _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def _run_stage(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    stage_id: str,
) -> int:
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/23_warmstart_registry.py",
        "--paper-run-id",
        args.paper_run_id,
        "--registry",
        str(args.registry),
        "--stage-ids",
        stage_id,
        "--runtime-mode",
        "jax_train",
        "--exp-folder",
        args.exp_folder,
        "--exp-dir",
        str(args.exp_dir),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--profile-root",
        str(args.profile_root),
        "--dclm-root",
        str(args.dclm_root),
        "--books-root",
        str(args.books_root),
        "--pretrain-steps",
        str(args.pretrain_steps),
        "--adapt-steps",
        str(args.adapt_steps),
        "--ext-steps",
        str(PROTOCOL_R_EFFECTIVE_EXT_STEPS if _protocol_r_stage(stage_id) else args.ext_steps),
        "--seed",
        str(args.seed),
        "--save-milestone-freq",
        str(args.save_milestone_freq),
        "--wandb-entity",
        args.wandb_entity,
        "--wandb-project",
        args.wandb_project,
        "--wandb-key",
        args.wandb_key,
    ]
    if _protocol_r_stage(stage_id):
        cmd.extend(["--ext-global-batch-size", str(PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE)])
    if args.allow_missing_fingerprints:
        cmd.append("--allow-missing-fingerprints")
    if args.skip_existing:
        cmd.append("--skip-existing")
    return _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def _summary_out(repo_root: Path, paper_run_id: str, batch: str) -> Path:
    return repo_root / "reports" / "paper" / paper_run_id / "split_batches" / f"{batch}.json"


def _run_reference_s1_gate(*, repo_root: Path, args: argparse.Namespace) -> int:
    parent_checkpoint = _checkpoint_dir(args.checkpoint_root, args.paper_run_id, BOOTSTRAP_RUN_ID)
    resume_step = _latest_step(parent_checkpoint) if (parent_checkpoint / "latest.json").exists() else 4799
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/50_run_reference_125m_32k_swa_smoke.py",
        "--repo-root",
        str(repo_root),
        "--books-root",
        str(args.books_root),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--resume-checkpoint-dir",
        str(parent_checkpoint),
        "--resume-step",
        str(resume_step),
        "--steps",
        str(args.gate_steps),
        "--save-milestone-freq",
        str(args.gate_save_milestone_freq),
        "--global-batch-size",
        str(PROTOCOL_R_EXT_GLOBAL_BATCH_SIZE),
        "--exp-dir",
        str(args.exp_dir),
        "--exp-folder",
        f"{args.exp_folder}_s1_refdiag",
        "--exp-name",
        "ext-125m-swa-32K-from-fa-ref-gate2",
    ]
    cmd.extend(["--timeout-seconds", str(args.s1_reference_timeout_seconds)])
    return _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def _run_local_s1_probe(*, repo_root: Path, args: argparse.Namespace) -> int:
    parent_checkpoint = _checkpoint_dir(args.checkpoint_root, args.paper_run_id, BOOTSTRAP_RUN_ID)
    cmd = [
        UV_EXECUTABLE,
        "run",
        "--exact",
        "python",
        "scripts/51_probe_local_125m_32k_swa.py",
        "--repo-root",
        str(repo_root),
        "--books-root",
        str(args.books_root),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--resume-checkpoint-path",
        str(parent_checkpoint),
        "--resume-checkpoint-format",
        "orbax",
        "--exp-dir",
        str(args.exp_dir),
        "--exp-folder",
        f"{args.exp_folder}_s1_localdiag",
        "--device-counts",
        args.s1_local_probe_device_counts,
        "--timeout-seconds",
        str(args.s1_local_probe_timeout_seconds),
    ]
    if str(args.dclm_root):
        cmd.extend(["--dclm-root", str(args.dclm_root)])
    return _run(cmd, cwd=repo_root, dry_run=args.dry_run)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    args = parse_args()
    args.exp_folder = args.exp_folder.strip() or args.paper_run_id
    args.exp_dir = args.exp_dir.expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.profile_root = args.profile_root.expanduser().resolve()
    args.dclm_root = args.dclm_root.expanduser().resolve()
    args.books_root = args.books_root.expanduser().resolve()
    args.registry = args.registry.expanduser().resolve()
    args.token = args.token.strip() or _env_default("HF_TOKEN", "")

    registry = load_registry(args.registry)
    stage_map = registry.stage_map()
    summary_rows: list[dict[str, object]] = []

    def record(step_id: str, rc: int, **extra: object) -> int:
        payload = {"step_id": step_id, "returncode": rc}
        payload.update(extra)
        summary_rows.append(payload)
        return rc

    config = BATCH_CONFIG[args.batch]

    if args.batch == "bootstrap_s0":
        source_run_id = BOOTSTRAP_RUN_ID
        target_run_id = BOOTSTRAP_RUN_ID
        if not _stage_done(args.exp_dir, args.checkpoint_root, args.paper_run_id, BOOTSTRAP_STAGE_ID, target_run_id):
            rc = _restore_stage(
                repo_root=repo_root,
                args=args,
                source_paper_run_id=BOOTSTRAP_SOURCE_PAPER_RUN_ID,
                source_stage_id=BOOTSTRAP_STAGE_ID,
                source_run_id=source_run_id,
                target_stage_id=BOOTSTRAP_STAGE_ID,
                target_run_id=target_run_id,
            )
            if record("restore_s0_seed", rc, source_paper_run_id=BOOTSTRAP_SOURCE_PAPER_RUN_ID) != 0:
                _write_json(_summary_out(repo_root, args.paper_run_id, args.batch), {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows})
                return rc
        rc = _export_stage(repo_root=repo_root, args=args, stage_id=BOOTSTRAP_STAGE_ID, run_id=target_run_id)
        record("export_s0_seed", rc, stage_id=BOOTSTRAP_STAGE_ID, run_id=target_run_id)
        _write_json(
            _summary_out(repo_root, args.paper_run_id, args.batch),
            {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
        )
        return 0 if rc == 0 else rc

    if args.batch == "h200_s1_diag":
        parent_stage = stage_map[BOOTSTRAP_STAGE_ID]
        if not _checkpoint_exists(args.checkpoint_root, args.paper_run_id, parent_stage.exp_name):
            rc = _restore_stage(
                repo_root=repo_root,
                args=args,
                source_paper_run_id=args.paper_run_id,
                source_stage_id=BOOTSTRAP_STAGE_ID,
                source_run_id=parent_stage.exp_name,
                target_stage_id=BOOTSTRAP_STAGE_ID,
                target_run_id=parent_stage.exp_name,
            )
            if record("restore_parent", rc, stage_id=BOOTSTRAP_STAGE_ID, run_id=parent_stage.exp_name) != 0:
                _write_json(
                    _summary_out(repo_root, args.paper_run_id, args.batch),
                    {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
                )
                return rc
        rc = _run_reference_s1_gate(repo_root=repo_root, args=args)
        if record("reference_s1_gate", rc, stage_id="S1_125M", gate_steps=args.gate_steps) != 0:
            _write_json(
                _summary_out(repo_root, args.paper_run_id, args.batch),
                {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
            )
            return rc
        rc = _run_local_s1_probe(repo_root=repo_root, args=args)
        if record("local_s1_probe", rc, stage_id="S1_125M") != 0:
            _write_json(
                _summary_out(repo_root, args.paper_run_id, args.batch),
                {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
            )
            return rc
        stage_id = "S1_125M"
        stage = stage_map[stage_id]
        run_id = stage.exp_name
        if not (args.skip_existing and _stage_done(args.exp_dir, args.checkpoint_root, args.paper_run_id, stage_id, run_id)):
            rc = _run_stage(repo_root=repo_root, args=args, stage_id=stage_id)
            if record("run_stage", rc, stage_id=stage_id, run_id=run_id) != 0:
                _write_json(
                    _summary_out(repo_root, args.paper_run_id, args.batch),
                    {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
                )
                return rc
            rc = _export_stage(repo_root=repo_root, args=args, stage_id=stage_id, run_id=run_id)
            if record("export_stage", rc, stage_id=stage_id, run_id=run_id) != 0:
                _write_json(
                    _summary_out(repo_root, args.paper_run_id, args.batch),
                    {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
                )
                return rc
            if not args.dry_run:
                checkpoint_dir = _checkpoint_dir(args.checkpoint_root, args.paper_run_id, run_id)
                _prune_checkpoint_history(checkpoint_dir, keep_steps=2)
                record("prune_checkpoint_history", 0, stage_id=stage_id, run_id=run_id, latest_step=_latest_step(checkpoint_dir))
        else:
            record("skip_existing_stage", 0, stage_id=stage_id, run_id=run_id)
        _write_json(
            _summary_out(repo_root, args.paper_run_id, args.batch),
            {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
        )
        return 0

    for parent_stage_id in config["parents"]:
        parent_stage = stage_map[parent_stage_id]
        if _checkpoint_exists(args.checkpoint_root, args.paper_run_id, parent_stage.exp_name):
            record("parent_present", 0, stage_id=parent_stage_id, run_id=parent_stage.exp_name)
            continue
        rc = _restore_stage(
            repo_root=repo_root,
            args=args,
            source_paper_run_id=args.paper_run_id,
            source_stage_id=parent_stage_id,
            source_run_id=parent_stage.exp_name,
            target_stage_id=parent_stage_id,
            target_run_id=parent_stage.exp_name,
        )
        if record("restore_parent", rc, stage_id=parent_stage_id, run_id=parent_stage.exp_name) != 0:
            _write_json(
                _summary_out(repo_root, args.paper_run_id, args.batch),
                {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
            )
            return rc

    if not args.skip_gates:
        for gate_stage_id in config["gates"]:
            gate_stage = stage_map[gate_stage_id]
            rc = _run_gate(repo_root=repo_root, stage=gate_stage, stage_map=stage_map, args=args)
            if record("gate", rc, stage_id=gate_stage_id, gate_steps=args.gate_steps) != 0:
                _write_json(
                    _summary_out(repo_root, args.paper_run_id, args.batch),
                    {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
                )
                return rc

    if args.stop_after_gates:
        _write_json(
            _summary_out(repo_root, args.paper_run_id, args.batch),
            {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
        )
        return 0

    for stage_id in config["stages"]:
        stage = stage_map[stage_id]
        run_id = stage.exp_name
        if args.skip_existing:
            if not _stage_done(args.exp_dir, args.checkpoint_root, args.paper_run_id, stage_id, run_id):
                rc = _restore_stage_if_exported(
                    repo_root=repo_root,
                    args=args,
                    stage_id=stage_id,
                    run_id=run_id,
                )
                record("restore_existing_stage", rc, stage_id=stage_id, run_id=run_id)
            if _stage_done(args.exp_dir, args.checkpoint_root, args.paper_run_id, stage_id, run_id):
                record("skip_existing_stage", 0, stage_id=stage_id, run_id=run_id)
                continue
        rc = _run_stage(repo_root=repo_root, args=args, stage_id=stage_id)
        if record("run_stage", rc, stage_id=stage_id, run_id=run_id) != 0:
            _write_json(
                _summary_out(repo_root, args.paper_run_id, args.batch),
                {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
            )
            return rc
        rc = _export_stage(repo_root=repo_root, args=args, stage_id=stage_id, run_id=run_id)
        if record("export_stage", rc, stage_id=stage_id, run_id=run_id) != 0:
            _write_json(
                _summary_out(repo_root, args.paper_run_id, args.batch),
                {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
            )
            return rc
        if not args.dry_run:
            checkpoint_dir = _checkpoint_dir(args.checkpoint_root, args.paper_run_id, run_id)
            _prune_checkpoint_history(checkpoint_dir, keep_steps=2)
            record("prune_checkpoint_history", 0, stage_id=stage_id, run_id=run_id, latest_step=_latest_step(checkpoint_dir))

    _write_json(
        _summary_out(repo_root, args.paper_run_id, args.batch),
        {"schema_version": "1.0", "paper_run_id": args.paper_run_id, "batch": args.batch, "rows": summary_rows},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
