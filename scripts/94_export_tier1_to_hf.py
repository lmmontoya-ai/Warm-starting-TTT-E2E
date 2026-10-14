#!/usr/bin/env python3
"""Upload completed Tier-1 reports and manifests to Hugging Face repos you name.

Plans the upload (writes --manifest-out) by default; --execute uploads. The
125M files go to --repo-id-125m (or HF_125M_RESULTS_REPO); with --include-760m
the 760M files go to --repo-id-760m (or HF_760M_RESULTS_REPO).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ttt.research.author_checkpoints import load_env_file
from ttt.research.tier1 import (
    load_tier1_config,
    read_json,
    sha256_file,
    sha256_payload,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export successful Tier-1 artifacts to HF."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/tier1"))
    parser.add_argument(
        "--tier1-root",
        type=Path,
        default=Path("reports/revision_v2/tier1/revision_v2_t1_eval_v1"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="T1b figure directory (default: <tier1-root>/aggregate/figures, where 92 writes it).",
    )
    parser.add_argument("--include-optional-t1b", action="store_true")
    parser.add_argument("--include-760m", action="store_true")
    parser.add_argument(
        "--repo-id-125m",
        default=os.environ.get("HF_125M_RESULTS_REPO", ""),
        help="Hugging Face repo for 125M artifacts (default: $HF_125M_RESULTS_REPO).",
    )
    parser.add_argument(
        "--repo-id-760m",
        default=os.environ.get("HF_760M_RESULTS_REPO", ""),
        help="Hugging Face repo for 760M artifacts (default: $HF_760M_RESULTS_REPO).",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("reports/revision_v2/tier1/tier1_hf_export_manifest.json"),
    )
    return parser.parse_args()


def _successful(path: Path, experiment: str) -> None:
    payload = read_json(path)
    if payload.get("experiment") != experiment or payload.get("status") != "succeeded":
        raise RuntimeError(f"Not a successful {experiment} manifest: {path}")


def _files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def main() -> int:
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()
    repo_by_scale = {
        "125m": args.repo_id_125m.strip(),
        "760m": args.repo_id_760m.strip(),
    }
    if not repo_by_scale["125m"]:
        raise ValueError("Name the 125M target repo with --repo-id-125m or HF_125M_RESULTS_REPO.")
    if args.include_760m and not repo_by_scale["760m"]:
        raise ValueError("--include-760m needs --repo-id-760m or HF_760M_RESULTS_REPO.")
    config = load_tier1_config(args.config)
    artifact_root = args.artifact_root.expanduser().resolve()
    tier1_root = args.tier1_root.expanduser().resolve()
    figures_dir = (
        args.figures_dir.expanduser().resolve()
        if args.figures_dir is not None
        else tier1_root / "aggregate" / "figures"
    )
    aggregate = read_json(tier1_root / "aggregate" / "aggregate_manifest.json")
    if aggregate.get("status") != "succeeded":
        raise RuntimeError("Tier-1 aggregate report is not complete")

    by_scale: dict[str, list[Path]] = {"125m": [], "760m": []}
    common = [
        artifact_root / "t1a_manifest.json",
        artifact_root / "t1b_manifest.json",
        artifact_root / "t1c_harness_validation.json",
        artifact_root / "t1c_ruler" / "manifest.json",
        artifact_root / "t1c_ruler" / "prompt_tokens.npy",
        figures_dir / "fast_weight_update_magnitude.pdf",
        figures_dir / "fast_weight_update_magnitude.png",
        *_files_under(tier1_root / "aggregate"),
    ]
    for path in common:
        if not path.is_file():
            raise FileNotFoundError(path)
    by_scale["125m"].extend(common)
    if args.include_760m:
        by_scale["760m"].extend(common)

    experiment_conditions = {
        "t1a": [
            "FA_SEED_125M",
            "S2_125M_SEED001",
            "S2_MINUS_125M_SEED001",
            "S3_125M",
            "S2_BRIDGE_40PCT_125M",
        ],
        "t1b": ["S2_125M_SEED001", "S2_MINUS_125M_SEED001"],
        "t1c": ["S2_125M_SEED001", "S2_MINUS_125M_SEED001", "S3_125M"],
    }
    for experiment, conditions in experiment_conditions.items():
        for condition in conditions:
            run_dir = tier1_root / experiment / condition
            _successful(run_dir / "run_manifest.json", f"T1{experiment[-1]}")
            by_scale["125m"].extend(_files_under(run_dir))
    if args.include_optional_t1b:
        for condition in ("S2_BRIDGE_5PCT_125M", "S2_BRIDGE_40PCT_125M"):
            run_dir = tier1_root / "t1b" / condition
            _successful(run_dir / "run_manifest.json", "T1b")
            by_scale["125m"].extend(_files_under(run_dir))
    if args.include_760m:
        for condition in ("S2_760M", "S3_760M"):
            run_dir = tier1_root / "t1c" / condition
            _successful(run_dir / "run_manifest.json", "T1c")
            by_scale["760m"].extend(_files_under(run_dir))

    export_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for scale, files in by_scale.items():
        if scale == "760m" and not args.include_760m:
            continue
        for path in files:
            key = (scale, str(path))
            if key in seen:
                continue
            seen.add(key)
            try:
                relative = path.relative_to(REPO_ROOT)
            except ValueError:
                relative = Path(path.name)
            export_rows.append(
                {
                    "scale": scale,
                    "repo_id": repo_by_scale[scale],
                    "local_path": str(path),
                    "path_in_repo": f"{config['paper_run_id']}/tier1/{relative.as_posix()}",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "include_optional_t1b": bool(args.include_optional_t1b),
        "include_760m": bool(args.include_760m),
        "files": export_rows,
        "status": "planned" if not args.execute else "uploading",
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    write_json(args.manifest_out, manifest)
    if not args.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required for Tier-1 export") from exc
    token = (
        args.token.strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        or None
    )
    api = HfApi(token=token)
    for row in export_rows:
        api.upload_file(
            repo_id=row["repo_id"],
            repo_type="model",
            path_or_fileobj=row["local_path"],
            path_in_repo=row["path_in_repo"],
            commit_message=f"Add {config['paper_run_id']} Tier-1 artifacts",
        )
    manifest["status"] = "succeeded"
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_payload(manifest)
    write_json(args.manifest_out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
