#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ttt.research.eval_runner import run_eval_command


@dataclass(frozen=True)
class RunRef:
    stage_id: str
    run_id: str
    exp_name: str
    run_dir: Path


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _discover_run_refs(
    *,
    exp_dir: Path,
    paper_run_id: str,
    stage_filter: set[str],
    run_filter: set[str],
) -> list[RunRef]:
    root = exp_dir / paper_run_id
    if not root.exists():
        raise FileNotFoundError(f"Missing paper run root: {root}")

    refs: list[RunRef] = []
    for manifest_path in root.rglob("run_manifest.json"):
        run_dir = manifest_path.parent
        payload = _load_json(manifest_path)

        stage_id = str(payload.get("stage_id", "")).strip()
        run_id = str(payload.get("run_id", run_dir.name)).strip() or run_dir.name
        exp_name = str(payload.get("exp_name", run_id)).strip() or run_id

        if not stage_id:
            # Infer from path in case legacy records omit stage_id.
            try:
                rel = run_dir.relative_to(root)
                if len(rel.parts) >= 2:
                    stage_id = rel.parts[0]
            except ValueError:
                pass
        if not stage_id:
            continue

        if stage_filter and stage_id not in stage_filter:
            continue
        if run_filter and run_id not in run_filter and exp_name not in run_filter:
            continue

        refs.append(RunRef(stage_id=stage_id, run_id=run_id, exp_name=exp_name, run_dir=run_dir))

    refs.sort(key=lambda r: (r.stage_id, r.run_id))
    return refs


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fields = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-aligned evaluation matrix for all runs under "
            "experiments/<paper_run_id>/<stage_id>/<run_id>."
        )
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", required=True)

    parser.add_argument("--stages", default="", help="Comma-separated stage ids to evaluate")
    parser.add_argument("--runs", default="", help="Comma-separated run ids/exp_names to evaluate")

    parser.add_argument("--eval-id", default="default_longctx")
    parser.add_argument("--contexts", default="8192,32768,65536,131072")
    parser.add_argument("--datasets", default="books3")
    parser.add_argument("--dclm-root", type=Path, default=Path("/tmp/phase1_token_data_dclm"))
    parser.add_argument("--books-root", type=Path, default=Path("/tmp/phase1_token_data_books"))
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)

    parser.add_argument("--niah-examples", type=int, default=64)
    parser.add_argument("--niah-candidates", type=int, default=16)
    parser.add_argument("--niah-positions", default="0.1,0.5,0.9")

    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--decode-prompts", type=int, default=8)

    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    dclm_root = args.dclm_root.expanduser().resolve()
    books_root = args.books_root.expanduser().resolve()

    stage_filter = set(_parse_csv(args.stages))
    run_filter = set(_parse_csv(args.runs))

    run_refs = _discover_run_refs(
        exp_dir=exp_dir,
        paper_run_id=args.paper_run_id,
        stage_filter=stage_filter,
        run_filter=run_filter,
    )
    if not run_refs:
        raise FileNotFoundError(
            "No runs found for evaluation under "
            f"{exp_dir / args.paper_run_id} with the requested filters."
        )

    if args.summary_json is None:
        summary_json = (
            Path("./reports/paper") / args.paper_run_id / "eval" / "eval_matrix_summary.json"
        ).resolve()
    else:
        summary_json = args.summary_json.expanduser().resolve()

    if args.summary_csv is None:
        summary_csv = (
            Path("./reports/paper") / args.paper_run_id / "eval" / "eval_matrix_summary.csv"
        ).resolve()
    else:
        summary_csv = args.summary_csv.expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[1]

    rows: list[dict[str, Any]] = []
    failed = 0
    for ref in run_refs:
        raw_json = ref.run_dir / "eval_raw.json"
        raw_csv = ref.run_dir / "eval_raw.csv"

        cmd = [
            "uv",
            "run",
            "--exact",
            "python",
            "scripts/11_external_eval.py",
            "--exp-dir",
            str(exp_dir),
            "--checkpoint-root",
            str(checkpoint_root),
            "--exp-folder",
            args.exp_folder,
            "--paper-run-id",
            args.paper_run_id,
            "--runs",
            ref.exp_name,
            "--contexts",
            args.contexts,
            "--datasets",
            args.datasets,
            "--dclm-root",
            str(dclm_root),
            "--books-root",
            str(books_root),
            "--eval-split",
            args.eval_split,
            "--eval-batches",
            str(args.eval_batches),
            "--eval-seed",
            str(args.eval_seed),
            "--niah-examples",
            str(args.niah_examples),
            "--niah-candidates",
            str(args.niah_candidates),
            "--niah-positions",
            args.niah_positions,
            "--decode-steps",
            str(args.decode_steps),
            "--decode-prompts",
            str(args.decode_prompts),
            "--out-json",
            str(raw_json),
            "--out-csv",
            str(raw_csv),
        ]
        if args.eval_batch_size > 0:
            cmd.extend(["--eval-batch-size", str(args.eval_batch_size)])
        if args.strict:
            cmd.append("--strict")

        print("$ " + " ".join(shlex.quote(x) for x in cmd))

        result = run_eval_command(
            eval_id=args.eval_id,
            paper_run_id=args.paper_run_id,
            stage_id=ref.stage_id,
            run_id=ref.run_id,
            exp_dir=exp_dir,
            command=cmd,
            raw_json_path=raw_json,
            raw_csv_path=raw_csv,
            repo_root=repo_root,
            dry_run=args.dry_run,
        )

        row = {
            "paper_run_id": args.paper_run_id,
            "stage_id": ref.stage_id,
            "run_id": ref.run_id,
            "exp_name": ref.exp_name,
            "status": result.status,
            "eval_id": args.eval_id,
            "loss_mean": result.metrics.get("loss_mean"),
            "niah_accuracy_mean": result.metrics.get("niah_accuracy_mean"),
            "tokens_per_second_mean": result.metrics.get("tokens_per_second_mean"),
            "eval_wall_seconds": result.metrics.get("eval_wall_seconds"),
            "raw_csv_path": str(raw_csv),
            "raw_json_path": str(raw_json),
            "error_message": result.error_message,
        }
        rows.append(row)

        if result.status == "failed":
            failed += 1
            if args.strict:
                break

    summary = {
        "schema_version": "1.0",
        "paper_run_id": args.paper_run_id,
        "exp_folder": args.exp_folder,
        "eval_id": args.eval_id,
        "contexts": args.contexts,
        "datasets": args.datasets,
        "rows": rows,
        "n_runs": len(rows),
        "n_failed": failed,
    }
    _write_json(summary_json, summary)
    _write_csv(summary_csv, rows)

    print(f"Wrote eval matrix summary JSON: {summary_json}")
    print(f"Wrote eval matrix summary CSV:  {summary_csv}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
