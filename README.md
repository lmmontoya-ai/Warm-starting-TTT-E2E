# Warm-starting TTT-E2E for Long-Context Language Modeling

This repository contains the in-repo JAX runtime, experiment registry, analysis
scripts, and curated paper artifacts for:

**Efficient Test-Time Training for Long Context via Warm-Starting from Pretrained
Transformers**

The study asks whether a long-context TTT-E2E model should be trained from
scratch or warm-started from a pretrained short-context full-attention
Transformer. The public release is organized around the finished paper rather
than around the earlier scratch reproduction workspace.

## Main Result

The experiments compare four long-context paths:

| ID | Path | Role |
| --- | --- | --- |
| S0 | full-attention seed -> direct 32K extension | long-context continuation control |
| S1 | full-attention seed -> naive SWA conversion -> 32K extension | mechanism-swap control |
| S2 | full-attention seed -> 8K TTT-E2E bridge -> 32K extension | warm-started TTT-E2E |
| S3 | scratch 8K TTT-E2E pretraining -> 32K extension | strongest in-family reference |

At 125M, warm-started TTT-E2E reduces Books32K validation loss by about 40%
relative to naive SWA conversion and is about 7.4x cheaper than the full scratch
TTT-E2E path when a full-attention seed already exists. Scratch TTT-E2E remains
the best final-quality path at both 125M and 760M. The S2-S3 gap approximately
halves from 125M to 760M.

## Repository Layout

- `ttt/` - local runtime, JAX training/eval code, checkpointing, reporting, and
  research utilities.
- `configs/` - Hydra model, training, experiment, backend, and research
  configurations.
- `configs/research/warmstart_registry.yaml` - canonical S0-S3 stage registry.
- `scripts/` - supported reproduction, evaluation, table, figure, and artifact
  helpers; older operational cloud/debug helpers are archival and unsupported.
- `tests/` - CPU-oriented unit and smoke tests for the local runtime and
  research utilities.
- `paper/plots/figures/` - final manuscript figures.
- `reports/paper/` - small curated result summaries needed to regenerate paper
  plots. Generated outputs are ignored by default.

Large generated artifacts, raw datasets, checkpoints, W&B runs, private
reference snapshots, and cloud logs are intentionally not part of the public
tree.

## Installation

This project uses Python 3.11 and `uv`.

```bash
uv sync
```

On Linux GPU hosts, the pinned CUDA/JAX dependencies in `pyproject.toml` are
used. On non-Linux systems, CPU JAX is installed for local tests and dry runs.

## Quick Smoke Test

Generate tiny local token data:

```bash
uv run --exact python scripts/04_make_token_data.py --out /tmp/warmstart_tokens
```

For registry smoke runs that include 32K stages, use a longer toy stream:

```bash
uv run --exact python scripts/04_make_token_data.py \
  --out /tmp/warmstart_tokens \
  --train-tokens 70000 \
  --val-tokens 70000
```

Run a token-stats pilot:

```bash
uv run --exact python scripts/06_phase1_pilot.py \
  --bootstrap-token-data \
  --skip-existing
```

Run the registry ladder in dry-run/token-stats mode:

```bash
uv run --exact python scripts/23_warmstart_registry.py \
  --paper-run-id warmstart_smoke \
  --exp-folder warmstart_smoke \
  --dclm-root /tmp/warmstart_tokens \
  --books-root /tmp/warmstart_tokens \
  --runtime-mode token_stats
```

## Supported Public Workflows

Dataset reproducibility:

```bash
uv run --exact python scripts/13_dataset_fingerprint.py \
  --dataset-id dclm_filter_8k \
  --path /path/to/dclm_filter_8k \
  --split train
```

```bash
uv run --exact python scripts/14_dataset_card.py \
  --fingerprints /path/to/train.fingerprint.json \
  --json-out ./reports/paper/demo/dataset_card.json \
  --csv-out ./reports/paper/demo/dataset_card.csv
```

Warm-start import and compatibility checks:

```bash
uv run --exact python scripts/15_import_hf_checkpoint.py \
  --model-key qwen2_5_0_5b \
  --exp-folder external_phase1_research
```

```bash
uv run --exact python scripts/16_audit_checkpoint_compat.py \
  --model-key qwen2_5_0_5b \
  --experiment external/qwen2_5_0_5b/pretrain-fa-import-8K \
  --exp-folder external_phase1_research \
  --exp-name import-qwen05-fa-base \
  --on-unresolved error
```

Evaluation and paper artifacts:

```bash
uv run --exact python scripts/18_eval_matrix.py \
  --paper-run-id warmstart_smoke \
  --exp-folder warmstart_smoke \
  --dclm-root /path/to/dclm_filter_8k \
  --books-root /path/to/books3
```

```bash
uv run --exact python scripts/20_make_paper_tables.py --paper-run-id warmstart_smoke
uv run --exact python scripts/21_make_paper_figures.py --paper-run-id warmstart_smoke
uv run --exact python scripts/22_make_artifact_bundle.py --paper-run-id warmstart_smoke
```

Final manuscript plots:

```bash
uv run --exact python scripts/75_make_paper_plots.py
```

## Documentation

- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Datasets](docs/DATASETS.md)
- [Artifacts](docs/ARTIFACTS.md)
- [Paper results](docs/PAPER_RESULTS.md)
- [760M revised protocol](docs/760M_PROTOCOL.md)

## Artifact Policy

The repository tracks only code, configs, tests, documentation, and curated
small paper summaries. Do not commit:

- tokenized datasets or raw dataset shards
- model checkpoints or author-provided checkpoint snapshots
- W&B run directories
- cloud provider logs or machine-local manifests
- `og_repo/`, `ttte2e_reference/`, or `swaa_reference/`

Use external storage for large artifacts and document their expected local
layout in `docs/ARTIFACTS.md`.

## Citation

See [CITATION.cff](CITATION.cff).
