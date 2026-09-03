#!/usr/bin/env python3
"""Static check: every params-restoring registry stage keeps its parent's tensor shapes.

Composes the Hydra config of each stage that warm-starts from a parent and of
the parent itself, then compares the model fields that determine parameter
shapes (vocab, hidden size, FFN width, depth, heads, tied embeddings). Any
difference is an error, because the loader would silently leave the affected
tensors at fresh initialization. Behaviour fields such as rope_theta produce
warnings. Needs no checkpoints or GPUs, so run it before renting anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt.research.registry import load_registry
from ttt.research.warmstart_validity import check_registry_warmstart_shapes, render_shape_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("./configs/research/warmstart_registry.yaml"))
    parser.add_argument("--stage-ids", default="", help="Comma-separated subset; default checks every stage.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(args.registry.expanduser().resolve())
    stage_ids = [s.strip() for s in args.stage_ids.split(",") if s.strip()] or None
    report = check_registry_warmstart_shapes(registry=registry, repo_root=repo_root, stage_ids=stage_ids)
    print(render_shape_report(report))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")
    has_warning = any(f.severity == "warning" for f in report.findings)
    if not report.ok or (args.fail_on_warning and has_warning):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
