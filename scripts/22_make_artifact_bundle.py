#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from ttt.research.lineage import file_sha256
from ttt.research.types import utc_now_iso


def _collect_existing(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.exists() and path.is_file():
            out.append(path)
    return out


def _collect_glob(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for pattern in patterns:
        out.extend([p for p in root.rglob(pattern) if p.is_file()])
    return out


def _manifest_rows(files: list[Path], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(set(files), key=lambda p: str(p)):
        rel = path.relative_to(repo_root)
        rows.append(
            {
                "path": str(rel),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible paper artifact bundle and manifest from configs, "
            "run manifests, tables, and figures."
        )
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--reports-root", type=Path, default=Path("./reports/paper"))
    parser.add_argument("--bundle-out", type=Path, default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    exp_root = args.exp_dir.expanduser().resolve() / args.paper_run_id
    paper_root = args.reports_root.expanduser().resolve() / args.paper_run_id

    fixed_files = _collect_existing(
        [
            repo_root / "configs/research/warmstart_registry.yaml",
            repo_root / "research_protocol/warmstart_preregistered_plan.yaml",
        ]
    )

    report_files = _collect_glob(paper_root, ["*.csv", "*.png", "*.json"])
    experiment_files = _collect_glob(
        exp_root,
        [
            "run_manifest.json",
            "stage_manifest.json",
            "checkpoint_manifest.json",
            "eval_manifest.json",
            "budget_manifest.json",
            "environment_manifest.json",
            "run_result.json",
            "resolved_config.yaml",
            "unresolved_config.yaml",
            "metrics.jsonl",
            "events.jsonl",
            "command.sh",
            "latest.json",
            "step_metadata_*.json",
            "eval_parity_raw.json",
            "eval_parity_raw.csv",
            "per_position_nll.npy",
        ],
    )

    files = sorted(set([*fixed_files, *report_files, *experiment_files]), key=lambda p: str(p))
    rows = _manifest_rows(files=files, repo_root=repo_root)

    if args.manifest_out is None:
        manifest_out = paper_root / "artifact_manifest.json"
    else:
        manifest_out = args.manifest_out.expanduser().resolve()
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "paper_run_id": args.paper_run_id,
        "repo_root": str(repo_root),
        "n_files": len(rows),
        "files": rows,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.bundle_out is None:
        bundle_out = paper_root / "artifact_bundle.tar.gz"
    else:
        bundle_out = args.bundle_out.expanduser().resolve()
    bundle_out.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle_out, "w:gz") as tar:
        for row in rows:
            src = repo_root / row["path"]
            tar.add(src, arcname=row["path"])

    print(f"Wrote artifact manifest: {manifest_out}")
    print(f"Wrote artifact bundle:   {bundle_out}")
    print(f"Included files: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
