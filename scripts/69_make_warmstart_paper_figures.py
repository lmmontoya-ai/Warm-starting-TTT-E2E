#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt.research.paper_figures import render_warmstart_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render publication-quality warm-start paper figures from plot-data CSVs."
    )
    parser.add_argument("--figure-set-id", default="warmstart_paper_v1")
    parser.add_argument("--plot-data-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_root = Path("./reports/paper") / args.figure_set_id
    plot_data_dir = (
        report_root / "plot_data" if args.plot_data_dir is None else args.plot_data_dir
    ).expanduser().resolve()
    figures_dir = (
        report_root / "figures" if args.figures_dir is None else args.figures_dir
    ).expanduser().resolve()
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("At least one output format is required.")

    manifest = render_warmstart_figures(
        figure_set_id=args.figure_set_id,
        plot_data_dir=plot_data_dir,
        figures_dir=figures_dir,
        formats=formats,
        strict=bool(args.strict),
    )
    print(f"Wrote figure manifest: {figures_dir / 'figure_render_manifest.json'}")
    for figure_id, meta in manifest.get("figures", {}).items():
        print(f"{figure_id}: {', '.join(meta.get('output_files', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
