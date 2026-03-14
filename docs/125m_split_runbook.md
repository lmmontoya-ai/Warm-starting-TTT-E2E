# 125M Split Protocol R Runbook

This runbook is the authoritative execution path for the split `125M` Protocol R ladder.

Canonical lineage:
- `paper_run_id = protocol_r_125m_main_v1`
- `exp_folder = protocol_r_125m_main_v1`
- HF repo is the durable checkpoint store between hardware batches

Canonical means:
- the stage ran to completion
- the final checkpoint and experiment files exist
- parity eval succeeded for the canonical stage run
- the stage was exported to HF under `protocol_r_125m_main_v1/stages/...`
- a verified `hf_export_manifest.json` exists for that canonical export
- the stage can be restored without relying on local leftovers

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

## Canonical Completed Stages

Already complete under the canonical lineage:
- `S0_PRETRAIN_FA_125M`
- `S0_125M`
- `S1_125M`
- `S2_ADAPT_125M`
- `S2_125M`

## Pending Subladders

1. `S3` diagnostic gate `s3_diag`
   - reference `S3_PRETRAIN_E2E_125M` 8-GPU smoke
   - local faithful `S3_PRETRAIN_E2E_125M` 8-GPU gate
   - local 8-GPU topology characterization

2. `S3` subladder `s3_ladder`
   - `S3_PRETRAIN_E2E_125M`
   - `S3_125M`

Historical batch names remain accepted by the split runner as compatibility aliases, but they are not the documented paper path anymore.

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

## H200 Subladder `h200_a` (Completed, Retained for Reruns)

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch h200_a \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
```

This subladder did all of the following in one durable unit:
- restore the canonical FA seed from HF
- run reference and local `S1` diagnostics first
- record the `S1` classification before any full `S1` attempt
- run/export/eval `S1_125M` only if the diagnostics say it is safe
- run/export/eval `S2_ADAPT_125M`
- restore the exported `S2_ADAPT_125M` parent from HF
- run/export/eval `S2_125M`

If `S1_125M` is classified as a feasibility failure, the runner still continues with `S2_ADAPT_125M` and `S2_125M`.

Current note:
- `S2_ADAPT_125M` and `S2_125M` are canonically complete from this batch
- `S1_125M` was later completed canonically via the dedicated faithful recovery path after the local faithful gate passed and the reference gate exposed a restore mismatch

## S3 Diagnostic Gate `s3_diag`

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch s3_diag \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
```

This gate runs:
- `scripts/56_run_reference_125m_s3_pretrain_smoke.py`
- `scripts/57_probe_local_125m_s3_scaling.py`

Faithful production target:
- `8x H200`
- `global_batch_size = 64`
- `n_data_parallel = 8`
- `n_state_parallel = 1`

Interpretation rules:
- classify the faithful pure data-parallel `8:1` path first
- treat `4:2` and `2:4` as exploratory topology characterization only
- if `8:1` passes and exploratory state-parallel topologies fail, `S3` is still considered production-viable

and classifies the outcome exactly as:
- `reference_pass_local_pass`
- `reference_pass_local_fail`
- `reference_fail_local_fail`

`s3_ladder` must not start unless `s3_diag` recorded `reference_pass_local_pass`.

Current saved result:
- faithful `8x H200`, pure data-parallel `8:1` passed on both the reference and local runtime
- saved classification:
  - `reports/paper/protocol_r_125m_main_v1/split_batches/s3_diag.json`

Important:
- `s3_diag` dry-runs or partial artifacts do not unlock `s3_ladder`
- `s3_ladder` only unlocks from a saved `s3_diag` summary with:
  - `status = succeeded`
  - `classification = reference_pass_local_pass`

## S3 Subladder `s3_ladder`

```bash
uv run --exact python scripts/47_run_125m_split_batch.py \
  --batch s3_ladder \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
```

This subladder:
- refuses to start unless `s3_diag` has passed
- runs/exports/evals `S3_PRETRAIN_E2E_125M`
- rehydrates the exported canonical parent from HF
- runs/exports/evals `S3_125M`

## Durability Rules

- restore parent stages from HF before each batch
- run parity `jax_eval` for every canonical stage before export
- export every successful stage to HF immediately after eval
- treat a stage as canonically complete only after:
  - succeeded training output
  - valid checkpoint
  - successful `eval_manifest.json`
  - verified `hf_export_manifest.json`
- keep only the latest complete checkpoint plus one fallback on the pod
- do not mirror heavy checkpoints back to this machine
- later batches should be able to start from HF-restored parents only
