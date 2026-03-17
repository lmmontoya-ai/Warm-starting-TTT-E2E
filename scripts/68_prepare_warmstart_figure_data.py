#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt.research.paper_figures import load_figure_set_spec, prepare_warmstart_figure_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized plot-data tables for the warm-start paper figures."
        )
    )
    parser.add_argument("--figure-set-id", default="warmstart_paper_v1")
    parser.add_argument(
        "--figure-set-config",
        type=Path,
        default=Path("./configs/research/paper_figure_sets.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--plot-data-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    exp_dir = args.exp_dir.expanduser().resolve()
    spec_path = args.figure_set_config.expanduser().resolve()
    spec = load_figure_set_spec(spec_path, figure_set_id=args.figure_set_id)
    if args.plot_data_dir is None:
        plot_data_dir = (
            repo_root / "reports" / "paper" / spec.output_report_id / "plot_data"
        ).resolve()
    else:
        plot_data_dir = args.plot_data_dir.expanduser().resolve()

    manifest = prepare_warmstart_figure_data(
        repo_root=repo_root,
        exp_dir=exp_dir,
        spec=spec,
        plot_data_dir=plot_data_dir,
        strict=bool(args.strict),
        spec_path=spec_path,
    )
    print(f"Wrote plot-data manifest: {plot_data_dir / 'plot_data_manifest.json'}")
    for figure_id, meta in manifest.get("figures", {}).items():
        print(f"{figure_id}: {meta.get('csv_path', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
