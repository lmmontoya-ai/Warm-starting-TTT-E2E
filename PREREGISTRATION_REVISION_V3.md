# Revision V3 Preregistration: Width-Preserving Warm Start

Status: active, 2026-09-02. Supersedes the revision-v2 warm-start runs listed
below; it does not change any revision-v2 decision rule.

> **Read Amendment 1 (2026-09-28) at the end of this file first.** It corrects
> the diagnosis below (revision v2 inherited nothing, not "embeddings and
> attention"; S0's failure was not the RoPE base), records the extension and
> bridge learning-rate schedule defects, and replaces the run design. No
> revision-v3 run had been launched when the amendment was written.

## Why A Revision V3 Exists

An audit of the revision-v2 artifacts (`scripts/84_audit_warmstart_restore.py`
against the canonical FA seed `protocol_r_125m_main_v1/S0_PRETRAIN_FA_125M`)
found that every "warm-started" 125M run inherited only the embeddings and the
attention projections (*Amendment 1: in fact nothing was inherited*). The converted configs set `model.intermediate_size`
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
- The scratch path (S3) and the eval pipeline are unaffected. *Amendment 1: S3's
  weights are unaffected, but its extension used the truncated LR schedule.*

## What Changes

1. Width-preserving conversion. Every converted config keeps the seed's FFN
   width (`model.intermediate_size: 2048` at 125M, `4096` at 760M). The
   fast-weight MLPs keep the published TTT-E2E width through the new
   `model.prime_intermediate_size` field (`1664` / `3328`). The converted 125M
   model has `194.98M` parameters (`183.47M` inherited, `11.51M` new prime).
2. The S0 full-attention extension uses the seed's RoPE base
   (`rope_theta: 500000` instead of `2000000`). *Superseded by Amendment 1:
   reverted to the published `2000000`; the base was not the cause.*
3. Trainer gates (`ttt/jax_runtime/warmstart_guard.py`), on by default for
   `training.load_part=params`:
   - any shape-mismatched inherited tensor fails the run unless
     `training.warmstart_allow_shape_mismatch=true`;
   - more than `warmstart_max_new_param_fraction` (`0.25`) of parameters with
     no checkpoint counterpart fails the run;
   - a first logged loss above `warmstart_max_initial_loss` (`7.0`) fails the run.
   Every warm-started run writes `restore_report.json` and records the
   coverage in `events.jsonl` under `run_started.restore.coverage`.
4. Static registry check (`scripts/85_check_registry_warmstart_shapes.py`, now
   `scripts/95_check_registry_warmstart_shapes.py`,
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

## Amendment 1 (2026-09-28): Corrected Diagnosis, Code Fixes, Revised Run Design

Written before any revision-v3 run. Nothing in this amendment was chosen after
seeing a revision-v3 result, because none exists.

### Corrections to the diagnosis above

1. **Revision v2 inherited nothing.** The shape-tolerant ("fallback") params
   restore returned a tree already shaped like the model. The trainer's apply
   step then looked up dict keys in that tree, matched none, and left every
   tensor at its fresh initialization. The coverage report was built from key
   lists, not from the model's values, so it reported embeddings and attention
   as inherited when they were not; `scripts/84_audit_warmstart_restore.py` as
   first written had the same blind spot. Reproduced on 2026-09-28 with a tiny
   model: the fallback reported 33,376 parameters restored while 20 of 20
   tensors kept their fresh values. Consequence: S1, the bridge (S2_ADAPT),
   S2-minus, every E3 arm and the 760M S2 were trained from random
   initialization. The E1 "bridge effect" (`2.0767`) compares about 250M tokens
   of TTT-E2E training from scratch against none.
2. **S0's random start was not caused by the RoPE base.** S0 ran on
   2026-03-13 at 23:55 UTC, before the exact restore path existed (commit
   `5d68399`, 2026-03-14 02:51 UTC), so it went through the no-op path too.
   This is inferred from timestamps because the S0 run manifest has an empty
   `git_commit`. With current code the seed restores bit-exactly into the S0
   config at `rope_theta: 2000000` and scores `3.910` against the seed's `3.891`
   on the probe below. The v3 change to `500000` is reverted to the published
   `2000000` (Tandon et al. p.28) and declared in the config through
   `training.warmstart_intended_changes`.
3. **Extension schedules never matched the run length.** Every protocol-R and
   revision-v2 125M extension (S0, S1, S2, S2-minus, S3, E3 arms) ran 480 steps
   at batch 8 with the 120-step schedule (warmup 12, cosine to `1e-5` at step
   120), so the LR sat at `1e-5` for steps 120-479. The logged
   `outer_learning_rate` on HF confirms it. The 760M extensions have the same
   pattern (decay over 725 or 2,900 steps against runs of up to 11,600).
   Consequence: the historical S3 numbers (`3.2722` at 125M, `2.6752` at 760M)
   are valid checkpoints of a truncated recipe and are not comparable with
   runs that use the fixed schedule. S3 is re-extended below.
4. **The bridge never annealed.** The 480-step bridge inherited the 4,800-step
   pretraining schedule, so it was one linear warmup ending at `2.975e-3`, and
   its inner-LR warmup spanned the whole bridge. E3 arms ended at different LRs.
5. **E2a is not a surviving null.** It compares the defective S2 with S3, and
   its 16-candidate scoring puts chance at 6.25%, where every arm landed
   (6.0% to 8.4%). E2a is retired; no retrieval claim is made from it.

### Code changes made with this amendment

- `ttt/jax_runtime/checkpoint.py`: both restore modes return the
  checkpoint-keyed dict; `apply_restored_weights` is the only function that
  writes checkpoint values into a model; the exact path falls back only when a
  shape differs or the checkpoint holds tensors the target lacks, and any other
  restore error propagates. `verify_restored_values` compares every same-shape
  tensor with the checkpoint after the restore.
- `ttt/jax_runtime/train.py`, `eval.py`: every restore is value-checked and
  fails if any same-shape tensor differs from the checkpoint. Eval also fails
  when any tensor stays at fresh initialization
  (`training.eval_allow_partial_restore` overrides it for deliberate probes).
  An optimizer-state resume that matches no leaf fails.
- `ttt/jax_runtime/schedule_guard.py`: the trainer refuses a run whose outer
  LR schedule does not fit `total_steps` (floor reached early, run ends in
  warmup, cosine cut off, inner-LR warmup longer than the run).
  `training.lr_schedule_check=warn` records a deliberate exception.
- `ttt/research/orchestrator.py`: `schedule_policy="scaled"` sets warmup to 10%
  of each stage's steps and cosine decay to its last step (inner-LR warmup
  also 10% for meta pretrain and adapt stages); `ext_accum_steps` applies
  gradient accumulation to extension stages only. Legacy plans are unchanged.
- `ttt/jax_runtime/model/attention.py`: the RoPE table covers
  `window + mini_batch` positions; before, SWA and TTT-E2E models scored below
  about 4.6K tokens returned NaN.
- `model.prime_post_norm_init` (default `1.0`, the published init): `0.0`
  makes a converted model compute the seed's function at step 0.
- Parameter-matched scratch control: `pretrain-125m-e2e-wide` and
  `ext-125m-e2e-wide-32K` (FFN 2048 + prime 1664, 194.98M parameters, the same
  as S2), registry stages `S3W_PRETRAIN_E2E_125M` and `S3W_125M`.
- `scripts/96_run_revision_v3_ladder.py` replaces `86_*`: one independent
  bridge per seed, `--variant` tags for pilot arms, the published extension
  recipe by default, optional chained evaluation with
  `scripts/34_eval_matrix_jax.py`. `scripts/85_*` is renumbered `95_*` so the
  Tier-1 scripts 85-94 on `main` merge without collisions.
- `scripts/84_audit_warmstart_restore.py`: value check plus an optional loss
  probe that scores the converted model and its parent on the same
  validation sequences.

### Pre-launch evidence (local CPU, 2026-09-28)

Canonical FA seed `protocol_r_125m_main_v1/pretrain-125m-fa` step 4799, probe
on 4 Books3 validation sequences of 2,048 tokens, inner-LR step 0. JSON in
`reports/revision_v3/preflight_local_20260928/` (untracked).

| Target | Restore | Restored | Value check | Probe loss | vs seed (3.8906) |
| --- | --- | ---: | --- | ---: | ---: |
| S0_125M (FA, RoPE 2e6) | exact | 183.47M | 15/15 | 3.9102 | +0.0195 |
| S1_125M (SWA) | exact | 183.47M | 15/15 | 3.8867 | -0.0039 |
| S2_ADAPT_125M | exact | 183.47M + 11.51M new | 15/15 | 4.2070 | +0.3164 |
| S2_MINUS_125M | exact | 183.47M + 11.51M new | 15/15 | 4.2500 | +0.3594 |
| S2_ADAPT_125M, prime post-norm 0 | exact | 183.47M + 11.51M new | 15/15 | 3.8867 | -0.0039 |

The `+0.32` to `+0.36` step-0 cost of the converted models comes from the
fresh prime MLPs, whose post-norm adds a unit-RMS branch to the residual
stream; with the gain at 0 the converted model matches the seed.

### Revised run design (replaces the Runs table above)

- Extension, every stage: 120 steps at global batch 32 (4-way accumulation on
  8 GPUs), 32K context, 126M tokens (the same token count as the 480 x 8 runs),
  10% warmup, cosine to `1e-5` at the last step. This is the published recipe.
- Bridge: 480 steps at batch 64 and 8K (10% of the 2.52B pretraining tokens),
  10% warmup, cosine to `1e-5` over the bridge, inner LR `0.1 -> 1.0` over the
  first 48 steps.
- One bridge per seed. Revision v2's "five paired seeds" shared one bridge.
- S3 is re-extended from `pretrain-125m-e2e` under the same extension recipe
  and re-evaluated with the same pipeline. Historical S3 values are not
  compared against revision-v3 runs.
- Evaluation: checkpoint restore, float32, Books3 validation at 32K,
  64 batches of 8 (`scripts/34_eval_matrix_jax.py`); bridges also on DCLM-8K
  at 8K.

### Pilot (Phase 1): PROPOSED, NOT YET PREREGISTERED

Proposed by the 2026-09-28 audit. Luis confirms or edits this section, and
removes this notice, before the pilot launches. Until then it is a plan, not a
preregistration.

- Runs, seed 1: S0, S1, S3 (re-extension), S2-minus, and bridge plus S2 at
  bridge peak LR `4e-4`, `1e-3` and `3e-3` (variants `lr4e-4`, `lr1e-3`,
  `lr3e-3`).
- Stop-and-diagnose gates before Phase 2:
  - every run passes the trainer gates (restore coverage, value check,
    initial loss below `7.0`, LR schedule);
  - S0 Books32K loss within `0.2` of the re-extended S3 (Tandon et al. Fig. 5b
    put 125M full attention within about `0.06` of TTT-E2E at 32K). If not,
    report it and diagnose before spending on Phase 2.
- Bridge peak-LR selection rule: pick the LR whose S2 has the lowest Books32K
  64-batch loss; if two are within `0.02`, pick the lower LR. The bridge's
  DCLM-8K loss against the seed's `3.5242` is recorded as a retention
  diagnostic, not a selection criterion. Phase 2 uses only the selected LR,
  and the pilot seed-1 runs count as Phase 2 seed 1 only if their configs are
  identical.
- Parameter-matched scratch (`S3W`) runs in Phase 2 only if the selected S2 is
  within `0.05` of the re-extended S3 or better, because only then can the
  10.6M-parameter difference change the conclusion.

### Decision rules

The E1 statistic, margin (`0.10`) and paired seed bootstrap from
`PREREGISTRATION_REVISION_V2.md` still apply, now over independent
bridge-plus-extension chains (at least three seeds). The interpretation guard
above is unchanged: if the bridge effect is below the margin, the bridge claim
is withdrawn.
