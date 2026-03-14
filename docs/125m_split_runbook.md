# 125M Split Protocol R Runbook

This runbook is the authoritative execution path for the split `125M` Protocol R ladder.

Canonical lineage:
- `paper_run_id = protocol_r_125m_main_v1`
- `exp_folder = protocol_r_125m_main_v1`
- HF repo is the durable checkpoint store between hardware batches

## Protocol Rules

Only the `32K` extension stages use Protocol R overrides:
- `S0_125M`
- `S1_125M`
- `S2_125M`
- `S3_125M`

Frozen extension settings:
- `ext_global_batch_size = 8`
- `effective_ext_steps = 480`
- preserve extension token budget relative to the original `32`

The `8K` stages remain faithful:
- `S0_PRETRAIN_FA_125M`
- `S2_ADAPT_125M`
- `S3_PRETRAIN_E2E_125M`

## Stage Lineage

- `S0_125M <- S0_PRETRAIN_FA_125M`
- `S1_125M <- S0_PRETRAIN_FA_125M`
- `S2_ADAPT_125M <- S0_PRETRAIN_FA_125M`
- `S2_125M <- S2_ADAPT_125M`
- `S3_PRETRAIN_E2E_125M <- scratch`
- `S3_125M <- S3_PRETRAIN_E2E_125M`

Important:
- `S1_125M` must not resume from `S0_125M`

## Hardware Split

1. `H200` batch `h200_s0`
   - `S0_125M`

2. `H100/RTX96GB` batch `h100_b`
   - `S2_ADAPT_125M`
   - `S3_PRETRAIN_E2E_125M`

3. `H200` batch `h200_s1_diag`
   - reference `S1` SWA gate
   - local `S1` 1/2/8-device compile probe
   - full `S1_125M` only if both diagnostics clear

4. `H200` batch `h200_c`
   - `S2_125M`
   - `S3_125M`

## Bootstrap the Canonical Seed

Rehydrate the already-completed `S0_PRETRAIN_FA_125M` into the canonical lineage and re-export it to HF:

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch bootstrap_s0 \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /path/to/dclm_filter_8k \
  --books-root /path/to/books3 \
  --exp-dir /tmp/protocol_r_125m_main_v1/experiments \
  --checkpoint-root /tmp/protocol_r_125m_main_v1/checkpoints
```

## Batch `h200_s0`

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch h200_s0 \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
```

## Batch `h100_b`

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch h100_b \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
```

For a cheap gate-only validation on 8x96GB hardware:

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch h100_b \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3 \
  --stop-after-gates
```

## Batch `h200_s1_diag`

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch h200_s1_diag \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
```

This batch runs:
- `scripts/50_run_reference_125m_32k_swa_smoke.py`
- `scripts/51_probe_local_125m_32k_swa.py`

and only launches the full `S1_125M` stage if both diagnostics pass.

## Batch `h200_c`

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch h200_c \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
```

## Durability Rules

- restore parent stages from HF before each batch
- export every successful stage to HF immediately
- keep only the latest complete checkpoint plus one fallback on the pod
- do not mirror heavy checkpoints back to this machine
- later batches should be able to start from HF-restored parents only
