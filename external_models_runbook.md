# External Models Runbook (Qwen2.5-0.5B + SmolLM2-360M)

This runbook describes everything needed to launch scratch and adapter training
paths for the external-pretrained follow-up track.

## Scope
- Primary: `Qwen/Qwen2.5-0.5B`
- Secondary: `HuggingFaceTB/SmolLM2-360M`
- Workflows covered:
  - scratch FA path,
  - scratch TTT-E2E path,
  - imported FA controls,
  - SWA bridge,
  - TTT bridge and direct 32K extension.

## 1) Prepare external model profiles

```bash
uv run --exact python scripts/07_prepare_external_models.py --model all
```

Outputs:
- `./artifacts/external_models/qwen2_5_0_5b/hf_config.json`
- `./artifacts/external_models/qwen2_5_0_5b/model_profile.json`
- `./artifacts/external_models/smollm2_360m/hf_config.json`
- `./artifacts/external_models/smollm2_360m/model_profile.json`

Notes:
- Config fetches are pinned to fixed Hugging Face revisions and SHA-256 validated.
- `model_profile.json` stores `source_revision` and `source_config_sha256`.

## 2) Prepare imported initialization checkpoints (adapter path prerequisite)

Adapter-path FA runs now require warm-start checkpoints to avoid metadata-only
"imported" runs.

```bash
uv run --exact python scripts/10_seed_external_import_checkpoints.py \
  --model all \
  --exp-folder external_phase1_pilot
```

Expected checkpoint names in your experiment folder:
- `import-qwen05-fa-base`
- `import-smol360-fa-base`

Each must contain `latest.json` under:
- `./checkpoints/<exp_folder>/<resume_exp_name>/latest.json`

Optional (better prior than synthetic Zipf): seed from token calibration data.

```bash
uv run --exact python scripts/10_seed_external_import_checkpoints.py \
  --model all \
  --exp-folder external_phase1_pilot \
  --dataset-root /tmp/phase1_token_data_dclm \
  --max-tokens 200000 \
  --force
```

## 3) Inspect launch matrix

```bash
uv run --exact python scripts/08_external_matrix.py \
  --model all \
  --path all \
  --runtime-mode token_stats
```

## 4) Run short-budget pilot

```bash
uv run --exact python scripts/09_external_pilot.py \
  --model all \
  --path all \
  --runtime-mode token_stats \
  --bootstrap-token-data \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books \
  --exp-folder external_phase1_pilot \
  --pretrain-steps 8 \
  --adapt-steps 4 \
  --ext-steps 4 \
  --skip-existing
```

Outputs:
- manifest JSON in `./experiments/*external_manifest.json`
- CSV/JSON report in `./experiments/*external_report.{csv,json}`

Stage data mapping:
- `pretrain` and `adapt` stages use `--dclm-root`.
- `ext` stages use `--books-root`.

## 5) Run scratch-only or adapter-only

Scratch-only:

```bash
uv run --exact python scripts/09_external_pilot.py \
  --model qwen2_5_0_5b \
  --path scratch \
  --bootstrap-token-data \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books
```

Adapter-only:

```bash
uv run --exact python scripts/09_external_pilot.py \
  --model smollm2_360m \
  --path adapter \
  --bootstrap-token-data \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books
```

## 6) Config map

### Qwen configs
- Model: `configs/model/qwen2_5_0_5b.yaml`
- Training: `configs/training/qwen2_5_0_5b/*.yaml`
- Experiments: `configs/experiment/external/qwen2_5_0_5b/*.yaml`

### Smol configs
- Model: `configs/model/smollm2_360m.yaml`
- Training: `configs/training/smollm2_360m/*.yaml`
- Experiments: `configs/experiment/external/smollm2_360m/*.yaml`

## 7) Evaluate runs (paper-style proxy suite)

```bash
uv run --exact python scripts/11_external_eval.py \
  --exp-folder external_phase1_pilot \
  --exp-dir ./experiments \
  --checkpoint-root ./checkpoints \
  --contexts 8192,32768,65536,131072 \
  --datasets books3 \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books
```

Outputs:
- `./experiments/<exp_folder>_external_eval.json`
- `./experiments/<exp_folder>_external_eval.csv`

Metrics produced:
- Loss vs context length (8K -> 128K).
- Efficiency: tokens/sec, latency proxy, FLOPs/sec proxy.
- NIAH-style retrieval proxy.
- Long-sequence decode-trend proxy (`decode_nll_slope`).

## 8) Notes for comparability
- Use same stage split (8K stage + 32K extension).
- Use `k=8K` sliding-window defaults in non-ablation runs.
- Keep E2E parameter-count matching via reduced `intermediate_size` in TTT-E2E configs.
- Keep FA imported controls when evaluating adapter variants.
- Include scratch TTT-E2E controls (not only scratch FA).
- Report both token-matched and wall-clock-matched comparisons.
- Use `scripts/05_phase1_report.py` outputs as the primary run ledger.

## 9) Exact experiment table (paper-extension track)

Conventions:
- Full budgets come from `configs/training/*` defaults.
- Pilot budgets are `scripts/09_external_pilot.py` defaults (`pretrain=8`, `adapt=4`, `ext=4`).
- All warm-start stages are fail-fast on missing source checkpoint/profile.
- Data roots:
  - `pretrain`/`adapt` -> `--dclm-root`
  - `ext` -> `--books-root`

### Qwen2.5-0.5B

| Run ID | Path | Config | Input checkpoint | Output checkpoint (`exp_name`) | Full steps | Pilot steps | Core metrics | Stop criteria |
|---|---|---|---|---|---:|---:|---|---|
| Q-A1 | adapter | `external/qwen2_5_0_5b/pretrain-fa-import-8K` | `import-qwen05-fa-base` | `pretrain-qwen05-fa-import-8K` | 12000 | 8 | train loss, tokens/sec, restore_status | stop at `total_steps`; fail on missing import/profile |
| Q-A2 | adapter | `external/qwen2_5_0_5b/ext-fa-32K-from-import` | `pretrain-qwen05-fa-import-8K` | `ext-qwen05-fa-32K-from-import` | 300 | 4 | 32K loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| Q-A3 | adapter | `external/qwen2_5_0_5b/adapt-swa-8K-from-import` | `pretrain-qwen05-fa-import-8K` | `adapt-qwen05-swa-8K-from-import` | 12000 | 4 | 8K stabilization loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| Q-A4 | adapter | `external/qwen2_5_0_5b/adapt-e2e-8K-from-import` | `adapt-qwen05-swa-8K-from-import` | `adapt-qwen05-e2e-8K-from-import` | 12000 | 4 | meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| Q-A5 | adapter | `external/qwen2_5_0_5b/ext-e2e-32K-from-import-bridge` | `adapt-qwen05-e2e-8K-from-import` | `ext-qwen05-e2e-32K-from-import-bridge` | 300 | 4 | 32K meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| Q-A6 | adapter | `external/qwen2_5_0_5b/ext-e2e-32K-from-import-direct` | `pretrain-qwen05-fa-import-8K` | `ext-qwen05-e2e-32K-from-import-direct` | 300 | 4 | 32K meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| Q-S1 | scratch | `external/qwen2_5_0_5b/pretrain-fa-scratch-8K` | fresh | `pretrain-qwen05-fa-scratch-8K` | 12000 | 8 | train loss, tokens/sec | stop at `total_steps` |
| Q-S2 | scratch | `external/qwen2_5_0_5b/ext-fa-32K-from-scratch` | `pretrain-qwen05-fa-scratch-8K` | `ext-qwen05-fa-32K-from-scratch` | 300 | 4 | 32K loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| Q-S3 | scratch | `external/qwen2_5_0_5b/pretrain-e2e-scratch-8K` | fresh | `pretrain-qwen05-e2e-scratch-8K` | 12000 | 8 | 8K meta loss, tokens/sec | stop at `total_steps` |
| Q-S4 | scratch | `external/qwen2_5_0_5b/ext-e2e-32K-from-scratch` | `pretrain-qwen05-e2e-scratch-8K` | `ext-qwen05-e2e-32K-from-scratch` | 300 | 4 | 32K meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |

### SmolLM2-360M

| Run ID | Path | Config | Input checkpoint | Output checkpoint (`exp_name`) | Full steps | Pilot steps | Core metrics | Stop criteria |
|---|---|---|---|---|---:|---:|---|---|
| S-A1 | adapter | `external/smollm2_360m/pretrain-fa-import-8K` | `import-smol360-fa-base` | `pretrain-smol360-fa-import-8K` | 9600 | 8 | train loss, tokens/sec, restore_status | stop at `total_steps`; fail on missing import/profile |
| S-A2 | adapter | `external/smollm2_360m/ext-fa-32K-from-import` | `pretrain-smol360-fa-import-8K` | `ext-smol360-fa-32K-from-import` | 240 | 4 | 32K loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| S-A3 | adapter | `external/smollm2_360m/adapt-swa-8K-from-import` | `pretrain-smol360-fa-import-8K` | `adapt-smol360-swa-8K-from-import` | 9600 | 4 | 8K stabilization loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| S-A4 | adapter | `external/smollm2_360m/adapt-e2e-8K-from-import` | `adapt-smol360-swa-8K-from-import` | `adapt-smol360-e2e-8K-from-import` | 9600 | 4 | meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| S-A5 | adapter | `external/smollm2_360m/ext-e2e-32K-from-import-bridge` | `adapt-smol360-e2e-8K-from-import` | `ext-smol360-e2e-32K-from-import-bridge` | 240 | 4 | 32K meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| S-A6 | adapter | `external/smollm2_360m/ext-e2e-32K-from-import-direct` | `pretrain-smol360-fa-import-8K` | `ext-smol360-e2e-32K-from-import-direct` | 240 | 4 | 32K meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| S-S1 | scratch | `external/smollm2_360m/pretrain-fa-scratch-8K` | fresh | `pretrain-smol360-fa-scratch-8K` | 9600 | 8 | train loss, tokens/sec | stop at `total_steps` |
| S-S2 | scratch | `external/smollm2_360m/ext-fa-32K-from-scratch` | `pretrain-smol360-fa-scratch-8K` | `ext-smol360-fa-32K-from-scratch` | 240 | 4 | 32K loss, tokens/sec | stop at `total_steps`; fail on missing resume |
| S-S3 | scratch | `external/smollm2_360m/pretrain-e2e-scratch-8K` | fresh | `pretrain-smol360-e2e-scratch-8K` | 9600 | 8 | 8K meta loss, tokens/sec | stop at `total_steps` |
| S-S4 | scratch | `external/smollm2_360m/ext-e2e-32K-from-scratch` | `pretrain-smol360-e2e-scratch-8K` | `ext-smol360-e2e-32K-from-scratch` | 240 | 4 | 32K meta loss, tokens/sec | stop at `total_steps`; fail on missing resume |

### Evaluation bundle (for paper-style comparability)

Run this on final FA and TTT checkpoints for each model/path branch:
- Context sweep: `{8K, 32K, 64K, 128K}`.
- Efficiency: tokens/sec, latency (ms/token), and FLOPs estimate at each context.
- Retrieval: RULER/NIAH accuracy vs context and needle position.
- Decoding trend: long-sequence quality metric vs generated length/context.

Suggested stop criteria for each eval job:
- Complete fixed eval token budget or fixed prompt count per context length.
- Mark run failed if any context bucket is missing or produces NaN/Inf metrics.

## 10) One-command end-to-end flow

```bash
uv run --exact python scripts/12_external_e2e_research.py \
  --model all \
  --path all \
  --budget pilot \
  --exp-folder external_phase1_research \
  --bootstrap-token-data \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books
```

For full-budget runs (model-specific step budgets from training defaults):

```bash
uv run --exact python scripts/12_external_e2e_research.py \
  --model all \
  --path all \
  --budget full \
  --exp-folder external_phase1_full \
  --dclm-root /path/to/dclm \
  --books-root /path/to/books3
```
