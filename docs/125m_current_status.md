# 125M Current Status

This note freezes the paper-grade `125M` ladder after the reporting repair pass on March 16, 2026.

What is now complete:
- all seven canonical `125M` stages are done under `protocol_r_125m_main_v1`
- canonical `run_result.json` metadata has nonzero observed `tokens_seen`
- canonical evals were rerun from saved/exported checkpoints only
- no `125M` training rerun was required

## Canonical Result

| Stage | Run ID | Dataset / Context | Eval `loss_ce_mean` | Tokens Seen | GPU Hours |
|---|---|---|---:|---:|---:|
| `S0_PRETRAIN_FA_125M` | `pretrain-125m-fa` | `dclm_filter_8k / 8192` | `3.524169921875` | `2516582400` | `19.142564091152614` |
| `S0_125M` | `ext-125m-fa-32K` | `books3 / 32768` | `6.583984375` | `125829120` | `0.5128987514957165` |
| `S1_125M` | `ext-125m-swa-32K-from-fa` | `books3 / 32768` | `6.5418243408203125` | `125829120` | `1.453785880711447` |
| `S2_ADAPT_125M` | `adapt-125m-e2e-8K-from-fa` | `dclm_filter_8k / 8192` | `4.32745361328125` | `251658240` | `2.10838213028231` |
| `S2_125M` | `ext-125m-e2e-32K-from-fa-bridge` | `books3 / 32768` | `3.917266845703125` | `125829120` | `1.3572525721488313` |
| `S3_PRETRAIN_E2E_125M` | `pretrain-125m-e2e` | `dclm_filter_8k / 8192` | `3.481170654296875` | `2516582400` | `24.161505875049365` |
| `S3_125M` | `ext-125m-e2e-32K` | `books3 / 32768` | `3.2729225158691406` | `125829120` | `1.5457706773829543` |

Primary artifacts:
- aggregate summary: `reports/paper/protocol_r_125m_main_v1/eval/aggregate_summary.json`
- run inventory: `reports/paper/protocol_r_125m_main_v1/tables/run_inventory.csv`
- Books `32K` rerun summary: `reports/paper/protocol_r_125m_main_v1/eval/books_32k_eval64_summary.json`
- DCLM `8K` rerun summary: `reports/paper/protocol_r_125m_main_v1/eval/dclm_8k_eval64_summary.json`

## Paper Interpretation

The repaired `125M` results support the following claims:
- Protocol F remains a real feasibility result for faithful `32K` FA extension.
- Protocol R is sufficient to finish the full ladder without retraining the already-completed stages.
- Warm-started TTT-E2E (`S2_125M`) is a large quality improvement over FA continuation (`S0_125M`) and naive SWA (`S1_125M`).
- From-scratch TTT-E2E (`S3_125M`) is the best final-quality `125M` model in the ladder.

Books `32K` comparison:
- `S0_125M`: `6.583984375`
- `S1_125M`: `6.5418243408203125`
- `S2_125M`: `3.917266845703125`
- `S3_125M`: `3.2729225158691406`

So the `125M` answer to the main research question is:
- warm-start is a strong practicality win
- warm-start is not the final-quality winner at `125M`
- the final-quality winner is scratch TTT-E2E (`S3_125M`)

Cost framing from the repaired run inventory:
- warm-start marginal cost:
  - `S2_ADAPT_125M + S2_125M = 3.4656347024311414` GPU-hours
- scratch TTT-E2E cost:
  - `S3_PRETRAIN_E2E_125M + S3_125M = 25.70727655243232` GPU-hours
- end-to-end FA-seed warm-start path:
  - `S0_PRETRAIN_FA_125M + S2_ADAPT_125M + S2_125M = 22.608198793583753` GPU-hours

At `125M`, the warm-start branch is cheaper than the scratch TTT-E2E branch, but it also lands at a worse final `32K` Books loss.

## Reporting Repair Notes

No new `125M` training was run for this repair pass.

What changed:
- final eval scalar aggregation now promotes metric arrays to `float32`
- `34_eval_matrix_jax.py` now records `loss_ce` from `eval_loss_ce`
- canonical metadata backfill now repairs observed `tokens_seen`, `wall_seconds`, and `gpu_hours`
- canonical evals were rerun in place at `64` eval batches

Important operational note:
- the H200 eval rerun used `TTT_ATTENTION_IMPLEMENTATION=xla` to avoid cuDNN attention plan failures during eval-only runs on the remote surface
- this was an eval-runtime stabilization step, not a training-method change

## Ladder Status

| NAME | IS_DONE | RESULT | INHERITS MODEL FROM |
|---|---|---|---|
| `S0_PRETRAIN_FA_125M` | `yes` | canonical complete; DCLM `8K` eval `3.524169921875` | `scratch` |
| `S0_125M` | `yes` | canonical complete under Protocol R; Books `32K` eval `6.583984375` | `S0_PRETRAIN_FA_125M` |
| `S1_125M` | `yes` | canonical complete under Protocol R; Books `32K` eval `6.5418243408203125` | `S0_PRETRAIN_FA_125M` |
| `S2_ADAPT_125M` | `yes` | canonical complete; DCLM `8K` eval `4.32745361328125` | `S0_PRETRAIN_FA_125M` |
| `S2_125M` | `yes` | canonical complete under Protocol R; Books `32K` eval `3.917266845703125` | `S2_ADAPT_125M` |
| `S3_PRETRAIN_E2E_125M` | `yes` | canonical complete; DCLM `8K` eval `3.481170654296875` | `scratch` |
| `S3_125M` | `yes` | canonical complete under Protocol R; Books `32K` eval `3.2729225158691406` | `S3_PRETRAIN_E2E_125M` |

## Next Step

The `125M` ladder no longer needs training work. The next paper step is the `760M` ladder seeded from the author-provided FA and TTT-E2E `8K` checkpoints, with the same evaluation and cost accounting discipline used here.
