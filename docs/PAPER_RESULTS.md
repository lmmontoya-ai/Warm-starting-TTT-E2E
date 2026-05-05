# Paper Results

The manuscript's figures and tables are generated from checkpoint-based
evaluation summaries and curated plot-data CSVs.

## Final Figures

The final LaTeX-ready figures are in `paper/plots/figures/`:

- `main_comparison_bar.pdf`
- `cost_quality_pareto.pdf`
- `continuation_trajectories.pdf`
- `extension_training_curves.pdf`
- `per_position_nll.pdf`
- `dclm8k_comparison.pdf`

Regenerate the working plot bundle with:

```bash
uv run --exact python scripts/75_make_paper_plots.py
```

## Curated Inputs

The most important tracked result roots are:

- `reports/paper/protocol_r_125m_main_v1/`
- `reports/paper/protocol_r_125m_ablations_v1/`
- `reports/paper/protocol_r_760m_author_seed_v1/`
- `reports/paper/protocol_r_760m_eta_live_v1*/`
- `reports/paper/warmstart_paper_v1/`

Raw eval snapshots, W&B logs, checkpoint trees, and local gate reports were
removed from the public tree. They are not needed to inspect the final paper
claims.

## Paper Claim Mapping

| Paper item | Primary source |
| --- | --- |
| 125M S0-S3 main comparison | `reports/paper/protocol_r_125m_main_v1/tables/stage_summary_loss_mean.csv` |
| 760M S2-S3 comparison | `reports/paper/protocol_r_760m_author_seed_v1/tables/stage_summary_loss_mean.csv` |
| 125M continuation analysis | `reports/paper/protocol_r_125m_ablations_v1/iso_total_tokens_summary.csv` |
| Cost-quality figure | `reports/paper/warmstart_paper_v1/plot_data/figure1_cost_quality_points.csv` |
| Extension curves | `reports/paper/warmstart_paper_v1/plot_data/figure4_extension_training_curves.csv` |
| Per-position NLL | `reports/paper/protocol_r_125m_main_v1/eval/per_position_nll_summary.json` and the 760M equivalent |
| Retrieval probe | `reports/paper/protocol_r_*/*/niah_jax_*` summaries |
