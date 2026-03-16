# 125M Paper Draft Results

This note consolidates the final `125M` results for the paper draft after the March 16, 2026 reporting repair pass.

Scope:
- all canonical `125M` stages under `protocol_r_125m_main_v1`
- corrected eval-only reruns at `64` batches
- repaired observed `tokens_seen`, `wall_seconds`, and `gpu_hours`
- no `125M` training reruns

Primary source artifacts:
- [aggregate_summary.json](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/eval/aggregate_summary.json)
- [run_inventory.csv](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/tables/run_inventory.csv)
- [books_32k_eval64_summary.json](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/eval/books_32k_eval64_summary.json)
- [dclm_8k_eval64_summary.json](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/eval/dclm_8k_eval64_summary.json)
- [warmstart_core_deltas.csv](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/tables/warmstart_core_deltas.csv)
- [s2_s3_warmstart_tax.csv](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/tables/s2_s3_warmstart_tax.csv)

## Executive Summary

At `125M`, the ladder is fully complete and the central result is clear:
- warm-started TTT-E2E (`S2_125M`) is a strong practicality win over FA continuation (`S0_125M`) and naive SWA (`S1_125M`)
- from-scratch TTT-E2E (`S3_125M`) is the best final-quality model
- therefore, at `125M`, warm-start is not the final-quality winner

The repaired `125M` evidence supports two paper claims:
1. faithful `32K` FA extension is a real feasibility problem, which motivates Protocol R
2. under Protocol R, warm-start is much cheaper than scratch TTT-E2E, but scratch still wins on final `32K` Books quality

## Canonical Stage Results

### Pretraining / Adaptation Stages

| Stage | Dataset | Context | Eval `loss_ce_mean` | Tokens Seen | GPU Hours | Wall Seconds |
|---|---|---:|---:|---:|---:|---:|
| `S0_PRETRAIN_FA_125M` | `dclm_filter_8k` | `8192` | `3.524169921875` | `2516582400` | `19.142564091152614` | `8614.153841018677` |
| `S2_ADAPT_125M` | `dclm_filter_8k` | `8192` | `4.32745361328125` | `251658240` | `2.10838213028231` | `948.7719586270396` |
| `S3_PRETRAIN_E2E_125M` | `dclm_filter_8k` | `8192` | `3.481170654296875` | `2516582400` | `24.161505875049365` | `10872.677643772215` |

### Long-Context `32K` Stages

| Stage | Initialization | Dataset | Context | Eval `loss_ce_mean` | Tokens Seen | GPU Hours | Wall Seconds |
|---|---|---|---:|---:|---:|---:|---:|
| `S0_125M` | FA `8K` seed | `books3` | `32768` | `6.583984375` | `125829120` | `0.5128987514957165` | `230.8044381730724` |
| `S1_125M` | FA `8K` seed + SWA conversion | `books3` | `32768` | `6.5418243408203125` | `125829120` | `1.453785880711447` | `654.2036463201512` |
| `S2_125M` | FA `8K` seed -> E2E adapt | `books3` | `32768` | `3.917266845703125` | `125829120` | `1.3572525721488313` | `610.7636574669741` |
| `S3_125M` | scratch E2E `8K` pretrain | `books3` | `32768` | `3.2729225158691406` | `125829120` | `1.5457706773829543` | `695.5968048223294` |

## Core Comparisons

### `S1` vs `S0`: naive SWA conversion

From [s1_s0_degradation.csv](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/tables/s1_s0_degradation.csv):
- loss delta: `-0.0421600341796875`
- throughput delta: `+225768.84429435022` tokens/s

Interpretation:
- naive SWA does not collapse quality at `125M`
- but it also does not solve the long-context quality problem in a meaningful way

### `S2` vs `S1`: warm-started recovery

From [s2_s1_recovery.csv](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/tables/s2_s1_recovery.csv):
- loss delta: `-2.6245574951171875`
- throughput delta: `-231288.23770121724` tokens/s

Interpretation:
- warm-started TTT-E2E produces the main qualitative jump of the ladder
- relative to `S1_125M`, this is about a `40.12%` reduction in `loss_ce_mean`

### `S2` vs `S3`: warm-start tax

From [s2_s3_warmstart_tax.csv](/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/reports/paper/protocol_r_125m_main_v1/tables/s2_s3_warmstart_tax.csv):
- loss delta (`S2 - S3`): `+0.6443443298339844`
- throughput delta (`S2 - S3`): `-6097.960537728766` tokens/s

Interpretation:
- scratch TTT-E2E is the final-quality winner at `125M`
- `S3_125M` improves on `S2_125M` by about `16.45%` relative to `S2`’s loss
- there was no near-tie after the repaired `64`-batch eval rerun, so no `256`-batch escalation was needed

## Cost Analysis

### Marginal cost

Warm-start marginal path:
- `S2_ADAPT_125M + S2_125M = 3.4656347024311414` GPU-hours

Scratch TTT-E2E path:
- `S3_PRETRAIN_E2E_125M + S3_125M = 25.70727655243232` GPU-hours

This means the warm-start marginal path uses about `13.48%` of the scratch path GPU-hours, or about `7.42x` less GPU time.

### End-to-end cost

If we include the FA seed pretraining cost in the warm-start branch:
- `S0_PRETRAIN_FA_125M + S2_ADAPT_125M + S2_125M = 22.608198793583753` GPU-hours

Relative to scratch TTT-E2E:
- warm-start end-to-end cost is about `87.94%` of the scratch TTT-E2E cost

Interpretation:
- if an FA seed already exists, warm-start is a major practicality win
- if the FA seed must also be trained from scratch, the cost advantage narrows sharply at `125M`
- even in that end-to-end view, warm-start remains somewhat cheaper, but it still loses on final quality

## Paper-Ready Claims Supported by `125M`

### Claim A: faithful feasibility boundary

The paper can still state that faithful `32K` FA extension is a real feasibility problem and that Protocol R was required to complete the full `125M` ladder.

### Claim B: warm-start meaningfully improves over simple baselines

At `32K` on Books:
- `S0_125M`: `6.583984375`
- `S1_125M`: `6.5418243408203125`
- `S2_125M`: `3.917266845703125`

This supports the statement that warm-started TTT-E2E is materially better than FA continuation and naive SWA conversion under the matched revised protocol.

### Claim C: scratch wins final quality at `125M`

At `32K` on Books:
- `S2_125M`: `3.917266845703125`
- `S3_125M`: `3.2729225158691406`

This supports the statement that, at `125M`, the warm-start branch is not the final-quality winner.

## Recommended Draft Text

Suggested paper-safe wording:

> At `125M`, warm-started TTT-E2E substantially improves over both FA continuation and a naive SWA conversion baseline under the revised matched protocol, but from-scratch TTT-E2E remains the best final-quality model. The warm-start path is therefore best interpreted as a practicality win at this scale rather than a final-quality win.

Suggested cost wording:

> When an FA seed is already available, the warm-start path (`S2_ADAPT + S2`) is about `7.4x` cheaper in GPU-hours than the scratch TTT-E2E branch (`S3_PRETRAIN_E2E + S3`). However, the scratch branch still attains the best final `32K` Books loss.

## Reporting / Method Notes

This draft uses the repaired reporting layer, not the earlier low-precision summaries.

Changes included in the repaired pass:
- float32 final aggregation for eval metrics
- corrected `loss_ce` extraction
- observed-token and observed-runtime backfill
- eval-only reruns from canonical exported checkpoints

Operational note:
- remote H200 eval-only reruns used `TTT_ATTENTION_IMPLEMENTATION=xla` to avoid cuDNN attention plan failures
- this changed the eval runtime surface only, not the training method or the saved checkpoints

## Next Step

The `125M` section is now ready to support the paper draft. The next experimental step is the `760M` ladder using the author-provided FA and TTT-E2E `8K` checkpoints as fixed seeds.
