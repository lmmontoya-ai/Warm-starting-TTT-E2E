# 125M Ladder Runbook

This runbook is the authoritative launch path for the first real `125M` ladder on the rebased parity `jax_runtime`.

## Environment

Use the repo-managed environment:

```bash
cd /Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E
uv venv
source .venv/bin/activate
uv sync --exact
```

Recommended W&B environment:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export WANDB_PROJECT=ttt-e2e-warmstart
```

If `WANDB_ENTITY` or `WANDB_PROJECT` is unset, the 125M wrapper defaults them to `none`, which disables GUI logging while preserving filesystem artifacts.

Dataset roots must include split fingerprint sidecars.
If the local `125M` compact package has already been deleted after upload, restore it from Backblaze B2 before launch, or intentionally point the run at the local `760M` compact root instead.

## Dry Run

Validate registry resolution, manifests, stage ordering, and command generation without starting training:

```bash
uv run --exact python scripts/35_run_125m_ladder.py \
  --paper-run-id warmstart_125m_dryrun \
  --exp-folder warmstart_125m_dryrun \
  --dclm-root /path/to/paper_budget_125m_val-full/dclm_filter_8k \
  --books-root /path/to/paper_budget_125m_val-full/books3 \
  --dry-run \
  --skip-figures \
  --skip-bundle
```

## Real Launch

Exploratory single-seed launch:

```bash
uv run --exact python scripts/35_run_125m_ladder.py \
  --paper-run-id warmstart_125m_seed0 \
  --exp-folder warmstart_125m_seed0 \
  --dclm-root /path/to/paper_budget_125m_val-full/dclm_filter_8k \
  --books-root /path/to/paper_budget_125m_val-full/books3 \
  --pretrain-steps 4800 \
  --adapt-steps 480 \
  --ext-steps 120 \
  --seed 0 \
  --save-milestone-freq 120
```

## Monitoring

Primary live monitor:
- W&B group = `paper_run_id`

Primary source of truth:
- `experiments/<paper_run_id>/<stage>/<run>/metrics.jsonl`
- `experiments/<paper_run_id>/<stage>/<run>/events.jsonl`
- `checkpoints/<exp_folder>/<exp_name>/latest.json`
- `experiments/<paper_run_id>/<stage>/<run>/eval_manifest.json`
- `experiments/<paper_run_id>/<stage>/<run>/jax_eval/<dataset>/ctx_<context>/per_position_nll.npy`

Useful commands:

```bash
tail -f experiments/<paper_run_id>/<stage>/<run>/metrics.jsonl
cat checkpoints/<exp_folder>/<exp_name>/latest.json
cat experiments/<paper_run_id>/<stage>/<run>/eval_manifest.json
```

## Failure Checks

Stop and inspect if any of these occur:
- `NaN` loss in `metrics.jsonl`
- missing `latest.json` after a checkpoint milestone should have been reached
- missing `eval_manifest.json` after parity eval
- missing `per_position_nll.npy` under the eval directory
- restore failure on warm-start stages `S1_125M` or `S2_ADAPT_125M`

## Outputs

Expected paper-facing outputs:
- `reports/paper/<paper_run_id>/eval/eval_parity_summary.json`
- `reports/paper/<paper_run_id>/eval/eval_parity_summary.csv`
- `reports/paper/<paper_run_id>/tables/warmstart_core_deltas.csv`
- `reports/paper/<paper_run_id>/tables/run_inventory.csv`
- `reports/paper/<paper_run_id>/figures/*.png`
- `reports/paper/<paper_run_id>/artifact_manifest.json`
- `reports/paper/<paper_run_id>/artifact_bundle.tar.gz`
