# Revision V3 Preregistration: Width-Preserving Warm Start

Status: active, 2026-09-02. Supersedes the revision-v2 warm-start runs listed
below; it does not change any revision-v2 decision rule.

## Why A Revision V3 Exists

An audit of the revision-v2 artifacts (`scripts/84_audit_warmstart_restore.py`
against the canonical FA seed `protocol_r_125m_main_v1/S0_PRETRAIN_FA_125M`)
found that every "warm-started" 125M run inherited only the embeddings and the
attention projections. The converted configs set `model.intermediate_size`
to `1664` (the parameter-matched TTT-E2E width) while the seed was trained at
`2048`, and the params restore silently left every shape-mismatched tensor at
fresh initialization:

| Target config (revision v2) | Inherited | Fresh init |
| --- | --- | --- |
| S1 sliding-window conversion | embeddings `98.5M`, attention `28.3M` | all 12 FFN blocks `46.0M` |
| S2_ADAPT bridge, S2-minus direct | embeddings `98.5M`, attention `28.3M` | all 12 FFN blocks `46.0M`, prime MLPs `11.5M` |

The training logs agree: the bridge started at loss `11.62`, S2-minus at
`11.06`, S1 at `11.94`, and the S0 full-attention extension at `11.94`, where
`11.94` is the loss of a random init and `ln(128256) = 11.76`. The 760M
conversions have the same defect (`4096 -> 3328`), already visible in
`artifacts/author_checkpoints/760m_local_runtime_audit.json`.

Consequences for the revision-v2 evidence:

- The E1 "bridge effect" (`2.0767`) compares 250M bridge tokens plus 126M
  extension tokens against 126M extension tokens, both starting from a mostly
  random model. It does not isolate a bridge for a pretrained checkpoint.
- The S0/S1 baselines (`6.5879`, `6.5423`) are 480-step near-scratch runs, not
  extensions of the seed.
- E3 is a pretraining learning curve, not a bridge-budget knob.
- The scratch path (S3) and the eval pipeline are unaffected.

## What Changes

1. Width-preserving conversion. Every converted config keeps the seed's FFN
   width (`model.intermediate_size: 2048` at 125M, `4096` at 760M). The
   fast-weight MLPs keep the published TTT-E2E width through the new
   `model.prime_intermediate_size` field (`1664` / `3328`). The converted 125M
   model has `194.98M` parameters (`183.47M` inherited, `11.51M` new prime).
2. The S0 full-attention extension uses the seed's RoPE base
   (`rope_theta: 500000` instead of `2000000`).
3. Trainer gates (`ttt/jax_runtime/warmstart_guard.py`), on by default for
   `training.load_part=params`:
   - any shape-mismatched inherited tensor fails the run unless
     `training.warmstart_allow_shape_mismatch=true`;
   - more than `warmstart_max_new_param_fraction` (`0.25`) of parameters with
     no checkpoint counterpart fails the run;
   - a first logged loss above `warmstart_max_initial_loss` (`7.0`) fails the run.
   Every warm-started run writes `restore_report.json` and records the
   coverage in `events.jsonl` under `run_started.restore.coverage`.
4. Static registry check (`scripts/85_check_registry_warmstart_shapes.py`,
   also `tests/test_warmstart_validity.py`): every params-restoring stage must
   keep its parent's shape-determining model fields.

## Runs

| Stage | Run id | Steps | Global batch | Context | Seeds |
| --- | --- | ---: | ---: | ---: | --- |
| S0_125M | `ext-125m-fa-32K` | 480 | 8 | 32K | 0 |
| S1_125M | `ext-125m-swa-32K-from-fa` | 480 | 8 | 32K | 0 |
| S2_ADAPT_125M | `adapt-125m-e2e-8K-from-fa` | 480 | 64 | 8K | 0 |
| S2_125M | `ext-125m-e2e-32K-from-fa-bridge-seedNNN` | 480 | 8 | 32K | 1-5 |
| S2_MINUS_125M | `ext-125m-e2e-32K-from-fa-direct-seedNNN` | 480 | 8 | 32K | 1-5 |

Paper run id: `revision_v3_widthpreserving_v1`. Deploy profile:
`revision_v3_vast_h100_8x` (8x H100 80GB, `data=8`, no accumulation).
Evaluation: the same checkpoint-restore float32 Books32K surface as revision
v2 (`64` eval batches, batch size `8`, context `32768`).

## Decision Rules (unchanged from revision v2)

The E1 statistic, seed pairing, `0.10` margin, bootstrap interval, and category
labels are those in `PREREGISTRATION_REVISION_V2.md`. New sanity expectations,
fixed before the runs:

- Step-0 training loss of S1, S2_ADAPT, and S2-minus must be below `7.0`
  (enforced by the gate). If the bridge starts near `4.5` or lower, the
  restore worked.
- S0 (full attention, 32K) must reach a Books32K loss comparable to or better
  than S3 (`3.2722`); full attention is the strongest baseline in Tandon et al.
  If it does not, the S0 config is still wrong and the paper must say so.
- The revision-v2 numbers are reported only as the superseded, defective
  conversion in an appendix, labelled as such.

## Interpretation Guard

If the width-preserving bridge effect is below the `0.10` margin, the paper's
bridge claim is withdrawn. If it remains positive, the characterization is
re-stated on the new numbers only. Either outcome is reportable.
