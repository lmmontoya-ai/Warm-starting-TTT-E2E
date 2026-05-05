# Reproducibility

This repository supports two reproducibility levels:

1. Local smoke runs using generated toy token streams.
2. Full JAX training/evaluation runs using external datasets and checkpoints.

The canonical experiment ladder is defined in
`configs/research/warmstart_registry.yaml`. All reproduction implementation is
in this repository; reference snapshots such as `og_repo/`,
`ttte2e_reference/`, and `swaa_reference/` are not required for public runs and
must remain outside the committed tree.

## Runtime Modes

- `training.runtime_mode=simulate` for orchestration dry runs.
- `training.runtime_mode=token_stats` for token-driven pilot runs.
- `training.runtime_mode=jax_train` for native in-repo JAX training.
- `training.runtime_mode=jax_eval` for native in-repo JAX evaluation.

## Toy Smoke Run

```bash
uv run --exact python scripts/04_make_token_data.py \
  --out /tmp/warmstart_tokens \
  --train-tokens 70000 \
  --val-tokens 70000
```

```bash
uv run --exact python scripts/23_warmstart_registry.py \
  --paper-run-id warmstart_smoke \
  --exp-folder warmstart_smoke \
  --dclm-root /tmp/warmstart_tokens \
  --books-root /tmp/warmstart_tokens \
  --runtime-mode token_stats
```

## Full Ladder

Use separate token roots for the two data surfaces:

```bash
uv run --exact python scripts/23_warmstart_registry.py \
  --paper-run-id warmstart_125m \
  --exp-folder warmstart_125m \
  --dclm-root /path/to/dclm_filter_8k \
  --books-root /path/to/books3 \
  --runtime-mode jax_train
```

For checkpoint-based evaluation:

```bash
uv run --exact python scripts/18_eval_matrix.py \
  --paper-run-id warmstart_125m \
  --exp-folder warmstart_125m \
  --dclm-root /path/to/dclm_filter_8k \
  --books-root /path/to/books3
```

Then regenerate tables and figures:

```bash
uv run --exact python scripts/20_make_paper_tables.py --paper-run-id warmstart_125m
uv run --exact python scripts/21_make_paper_figures.py --paper-run-id warmstart_125m
```

## Stage Mapping

| Stage | Description |
| --- | --- |
| S0 | full-attention seed directly extended to 32K |
| S1 | full-attention seed converted to sliding-window attention, then extended |
| S2 | full-attention seed adapted through the 8K TTT-E2E bridge, then extended |
| S3 | TTT-E2E trained from scratch at 8K, then extended |

The paper's 760M comparison uses author-provided 8K seeds as fixed upstream
starting points. Those checkpoints are not redistributed in this repository.
