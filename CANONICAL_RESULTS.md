# Canonical Results Ledger

Status: 2026-06-08.

This ledger defines the authoritative historical result surfaces for the current
paper/revision work. It resolves conflicts among tracked paper reports, plot
data, and draft artifacts. The repository tables listed here are the source of
truth for historical statements; manuscripts are not.

This ledger does not replace the revision-v2 preregistration. New revision
claims, especially the S2-minus bridge-isolation gate, must re-evaluate S1, S2,
S3, and S2-minus with the same current checkpoint-based float32 evaluation
pipeline. Do not compare a fresh S2-minus run against only a historical ledger
value.

## Global Rules

- Existing curated paper reports are read-only historical artifacts.
- New revision runs must use new `paper_run_id` values and new report roots.
- Reference snapshots are read-only if present: `og_repo/`,
  `ttte2e_reference/`, and `swaa_reference/`.
- Every new reported scalar must come from checkpoint restore plus float32
  evaluation.
- Future comparison groups must enforce matching dataset fingerprints. The
  current tracked reports name datasets and eval surfaces, but do not provide
  complete tracked fingerprint sidecars inside `reports/paper`.
- If reconciliation changes a headline number beyond the current eval-noise
  floor, pause and escalate before writing or running follow-on compute.

Headline numbers for the escalation rule are:

- 125M Books32K warm-start tax, `S2_125M - S3_125M`.
- 760M Books32K warm-start tax, `S2 - S3`.
- 125M continuation endpoints.
- 125M marginal/full/scratch GPU-hour accounting.
- Retrieval proxy direction, recorded only as preliminary context because the
  current proxy result is underpowered.

## Revision V2 Current-Pipeline 64-Batch Reconciliation

Status: 2026-06-10. The revision-v2 E1 gate initially used an 8-batch
current-pipeline eval for speed. Those 8-batch values moved the historical
S1/S2/S3 losses by roughly `0.09` to `0.20`, triggering the escalation rule
above. The follow-up reran the same checkpoint-based JAX/float32 pipeline at
the historical 64-batch Books32K surface:

- context length: `32768`
- dataset: `books3` validation
- eval batches: `64`
- eval batch size: `8`
- eval tokens per row: `16,777,216`
- paper run id for report artifacts: `revision_v2_current_pipeline64_v1`
- checkpoint folder: `revision_v2_e1_paired_v1`

S0 initially OOMed on the 2xH100 Prime topology during XLA autotuning at this
surface. The rerun succeeded at the same 64-batch / batch-size-8 surface with
`XLA_FLAGS=--xla_gpu_autotune_level=3`; S1/S2/S3 and E1 eval64 used the same
setting.

Local report artifacts, mirrored to
`Luxel/ttt-e2e-125m-results/revision_v2_e1_paired_v1/reports/revision_v2/`:

- `reports/revision_v2/current_eval64/s0_s1_s2_s3_books32k_jax_eval64_combined.json`
- `reports/revision_v2/current_eval64/s0_s1_s2_s3_books32k_jax_eval64_combined.csv`
- `reports/revision_v2/current_eval64/s0_s1_s2_s3_books32k_jax_eval64_combined.md`
- `reports/revision_v2/e1_pairs_eval64/e1_bridge_effect_analysis_eval64.json`
- `reports/revision_v2/e1_pairs_eval64/e1_bridge_effect_paired_losses_eval64.csv`
- `reports/revision_v2/e1_pairs_eval64/e1_bridge_effect_analysis_eval64.md`

Current-pipeline 64-batch Books32K losses:

| Stage | Loss |
| --- | ---: |
| S0_125M | 6.587890625 |
| S1_125M | 6.5423126220703125 |
| S2_125M | 3.9168930053710938 |
| S3_125M | 3.272228240966797 |

Current-pipeline 64-batch deltas:

| Quantity | Value |
| --- | ---: |
| S1_125M - S0_125M | -0.0455780029296875 |
| S2_125M - S1_125M | -2.6254196166992188 |
| S2_125M - S3_125M | 0.6446647644042969 |

Revision-v2 E1 bridge-isolation result at the same 64-batch surface:

| Quantity | Value |
| --- | ---: |
| Mean S2_125M loss over five paired seeds | 3.9208450317382812 |
| Mean S2_MINUS_125M loss over five paired seeds | 5.997589111328125 |
| Mean bridge effect, `S2_MINUS - S2` | 2.0767440795898438 |
| 95% paired seed-level bootstrap CI | [2.050018310546875, 2.0969802856445314] |
| Preregistered margin | 0.10 |
| Decision | helps |

This closes the escalation: the 64-batch current-pipeline values return to the
historical eval64 neighborhood, so the earlier drift is attributed to the
smaller 8-batch gate eval rather than a systematic pipeline bias. For revision
manuscript tables, use these current-pipeline 64-batch values. The
preregistered 8-batch E1 gate decision is not reopened; the 64-batch rerun
keeps the same decision with a larger bridge-effect estimate.

## Revision V2 E3 Bridge-Budget Frontier

Status: 2026-06-12. E3 was run after the E1 bridge-isolation gate passed. It
characterizes the 125M bridge-budget frontier at the same Books32K 64-batch
checkpoint-evaluation surface used for revision-v2 manuscript claims.

Artifacts:

- HF repo: `Luxel/ttt-e2e-125m-results`
- E3 paper run id: `revision_v2_e3_frontier_v1`
- S2-minus continuation paper run id: `revision_v2_s2minus_cont_v1`
- Local copied report bundle:
  `reports/revision_v2/e3_frontier/remote_bundle/`
- Remote summaries preserved locally inside that bundle:
  - `reports/revision_v2/e3_frontier/vast_h200_run_summary.json`
  - `reports/revision_v2/e3_frontier/e3_eval_summary.json`
  - `reports/revision_v2/e3_frontier/s2minus_cont_eval_summary.json`

Execution notes:

- Hardware: 8x H200 on Vast.ai, pure data parallelism.
- Bridge and extension ran with `accum_steps=1`.
- All seven training stages succeeded and exported to HF.
- Total observed training GPU-hours across the E3 arms and S2-minus
  continuation: `25.394241899416234`.
- The Vast instance was destroyed after HF export verification and local
  artifact copy.

Canonical E3 frontier losses:

| Bridge budget | Source | Loss | Recovery vs E1 bridge effect |
| --- | --- | ---: | ---: |
| 0% | E1 `S2_MINUS_125M` five-seed mean | 5.997589111328125 | 0.0000 |
| 5% | `S2_BRIDGE_5PCT_125M`, seed001 | 4.539649963378906 | 0.7020 |
| 10% | E1 `S2_125M` five-seed mean | 3.9208450317382812 | 1.0000 |
| 20% | `S2_BRIDGE_20PCT_125M`, seed001 | 3.645965576171875 | 1.1324 |
| 40% | `S2_BRIDGE_40PCT_125M`, seed001 | 3.4461746215820312 | 1.2286 |

Preregistered E3 shape read:

- The 5% bridge arm narrowly meets the preregistered `threshold-like` boundary:
  it recovers `70.20%` of the E1 bridge effect, just above the `70%` threshold.
- The frontier is also budget-responsive beyond the 10% anchor: 20% improves
  over the 10% E1 mean by `0.2748794555664062`, and 40% improves over the 10%
  E1 mean by `0.47467041015625`.
- The `saturating` label is not earned because 20% and 40% are not within
  `0.10` of the 10% anchor.
- The `budget-hungry` label is not earned because 5% recovers more than 30% of
  the E1 bridge effect.

Interpretation caveat:

- E3 changes bridge token budget, so it also changes upstream short-context
  training tokens. It characterizes the practical bridge-budget frontier; it is
  not a pure structural ablation isolating bridge mechanism from extra upstream
  tokens.
- E3 points are single-seed except for the 0% and 10% E1 anchors. Treat the
  5% threshold label as arithmetic under the preregistered rule, not as a broad
  variance claim.

S2-minus continuation:

| Stage | Loss | Improvement vs E1 S2-minus mean | Gap vs E1 S2 mean |
| --- | ---: | ---: | ---: |
| `S2_MINUS_CONT_125M` (+1440 extension steps) | 5.487419128417969 | 0.510169982910156 | 1.5665740966796878 |

The continuation improvement is `>= 0.25`, so the preregistered continuation
read is `plateau claim weakened`. Extra extension compute helps S2-minus, but
the continued no-bridge model remains far worse than the bridged 10%, 20%, and
40% paths. Do not claim the no-bridge plateau is persistent without this
caveat.

## Authoritative Historical Inputs

### 125M Main Protocol R

Use these files for current 125M historical main-result claims:

- `reports/paper/protocol_r_125m_main_v1/tables/stage_summary_loss_mean.csv`
- `reports/paper/protocol_r_125m_main_v1/tables/run_inventory.csv`
- `reports/paper/protocol_r_125m_main_v1/tables/warmstart_core_deltas.csv`
- `reports/paper/protocol_r_125m_main_v1/tables/s2_s3_warmstart_tax.csv`
- `reports/paper/protocol_r_125m_main_v1/eval/books_32k_eval64_summary.csv`
- `reports/paper/protocol_r_125m_main_v1/eval/dclm_8k_eval64_summary.csv`
- `reports/paper/protocol_r_125m_main_v1/eval/dclm_8k_s2_s3_eval64_summary.csv`
- `reports/paper/protocol_r_125m_main_v1/eval/per_position_nll_summary.json`
- `reports/paper/protocol_r_125m_main_v1/eval/niah_jax_s2_s3_summary.csv`
- `reports/paper/protocol_r_125m_main_v1/books3_compact_audit.json`

Canonical Books32K losses:

| Stage | Loss |
| --- | ---: |
| S0_125M | 6.583984375 |
| S1_125M | 6.5418243408203125 |
| S2_125M | 3.917266845703125 |
| S3_125M | 3.2729225158691406 |

Canonical DCLM-8K upstream/eval losses:

| Stage | Loss |
| --- | ---: |
| S0_PRETRAIN_FA_125M | 3.524169921875 |
| S2_ADAPT_125M | 4.32745361328125 |
| S3_PRETRAIN_E2E_125M | 3.481170654296875 |

Core 125M deltas:

| Quantity | Value |
| --- | ---: |
| S1_125M - S0_125M | -0.0421600341796875 |
| S2_125M - S1_125M | -2.6245574951171875 |
| S2_125M - S3_125M | 0.6443443298339844 |

Canonical 125M GPU-hour accounting:

| Quantity | GPU-hours |
| --- | ---: |
| Warm-start marginal, S2_ADAPT_125M + S2_125M | 3.4656347024311414 |
| Warm-start full, S0_PRETRAIN_FA_125M + S2_ADAPT_125M + S2_125M | 22.608198793583757 |
| Scratch, S3_PRETRAIN_E2E_125M + S3_125M | 25.70727655243232 |
| Scratch / warm-start marginal | 7.417768680120463 |
| Scratch / warm-start full | 1.137077605657293 |

125M retrieval/NIAH proxy context is preliminary and underpowered:

| Stage | Mean accuracy | 8K | 32K |
| --- | ---: | ---: | ---: |
| S2_125M | 0.11458333333333333 | 0.08333333333333333 | 0.14583333333333334 |
| S3_125M | 0.07291666666666667 | 0.08333333333333333 | 0.0625 |

The Books3 compact audit reports that the compact train package is a strict
prefix and is less representative than the larger validation surface. Future
revision runs should prefer canonical token roots with explicit fingerprint
sidecars and should not rely on contiguous prefix exports for new claims.

### 125M Continuation

Use this normalized continuation curve for current historical continuation
claims and final figures:

- `reports/paper/warmstart_paper_v1/plot_data/figure3_continuation_frontier.csv`

It is the authoritative continuation table because it combines the ablation
snapshots with recovered/evaluated endpoint losses. It supersedes stale summary
files in `reports/paper/protocol_r_125m_ablations_v1`.

Canonical continuation endpoints:

| Mode | Extra steps | Checkpoint step | Loss |
| --- | ---: | ---: | ---: |
| S2 iso-quality continuation | 1440 | 1919 | 3.8657073974609375 |
| S3 token-equalized continuation | 960 | 1439 | 3.2623252868652344 |

Historical interpretation:

- S2 improves from 3.917266845703125 to 3.8657073974609375 over 1440
  additional steps and does not reach the S3 target.
- S3 improves from 3.2729225158691406 to 3.2623252868652344 under the
  token-equalized continuation budget.
- Only 125M continuation data exists; 760M continuation was not run.

### 760M Author-Seed Protocol R

Use these files for current 760M historical quality claims:

- `reports/paper/protocol_r_760m_author_seed_v1/tables/stage_summary_loss_mean.csv`
- `reports/paper/protocol_r_760m_author_seed_v1/tables/run_inventory.csv`
- `reports/paper/protocol_r_760m_author_seed_v1/tables/s2_s3_warmstart_tax.csv`
- `reports/paper/protocol_r_760m_author_seed_v1/eval/books_32k_eval64_summary.csv`
- `reports/paper/protocol_r_760m_author_seed_v1/eval/dclm_8k_s2_s3_eval64_summary.csv`
- `reports/paper/protocol_r_760m_author_seed_v1/eval/per_position_nll_summary.json`
- `reports/paper/protocol_r_760m_author_seed_v1/eval/niah_jax_s2_s3_summary.csv`

Canonical Books32K losses:

| Stage | Loss |
| --- | ---: |
| S2 | 2.9939842224121094 |
| S3 | 2.675201416015625 |

Canonical 760M Books32K warm-start tax:

| Quantity | Value |
| --- | ---: |
| S2 - S3 | 0.3187828063964844 |

Canonical 760M DCLM-8K S2/S3 losses:

| Stage | Loss |
| --- | ---: |
| S2 | 3.343231201171875 |
| S3 | 2.982208251953125 |

760M retrieval/NIAH proxy context is preliminary and underpowered:

| Stage | Mean accuracy | 8K | 32K |
| --- | ---: | ---: | ---: |
| S2 | 0.09375 | 0.0625 | 0.125 |
| S3 | 0.07291666666666667 | 0.020833333333333332 | 0.125 |

760M cost caveat:

- The 760M S2/S3 quality numbers are checkpoint-based historical results.
- Homogeneous 760M cost accounting is incomplete. The S2/S3 extension rows in
  `run_inventory.csv` are observed local extension costs, but the upstream
  author-provided seed costs are external and not measured here.
- ETA-style 760M cost estimates belong only in appendices or limitations, not
  in main result tables.

### Final Manuscript Plots

Use these final plot artifacts for the current paper figure set:

- `scripts/75_make_paper_plots.py`
- `paper/plots/plot_manifest.json`
- `paper/plots/figures/main_comparison_bar.pdf`
- `paper/plots/figures/cost_quality_pareto.pdf`
- `paper/plots/figures/continuation_trajectories.pdf`
- `paper/plots/figures/extension_training_curves.pdf`
- `paper/plots/figures/per_position_nll.pdf`
- `paper/plots/figures/dclm8k_comparison.pdf`

The final plot manifest records the current caveats:

- Continuation trajectories are 125M only.
- 125M cost-quality points are exact branch and marginal costs from canonical
  run-inventory rows.
- 760M warm-start marginal cost in the Pareto plot combines ETA-derived
  S2_ADAPT cost with observed resumed S2 cost.
- 760M full-branch cost is intentionally omitted because author-provided seed
  GPU-hours are external.
- DCLM-8K comparison plots only S2/S3 at 760M because 760M S0/S1 controls are
  not available.

## Non-Authoritative Or Stale Surfaces

Do not use these for new headline claims when they conflict with the
authoritative sources above:

- `reports/paper/draft_v1.md`: useful historical draft, but stale relative to
  the later 760M reports and final plot path.
- `reports/paper/protocol_r_125m_ablations_v1/iso_quality_summary.csv`
- `reports/paper/protocol_r_125m_ablations_v1/iso_quality_summary.json`
- `reports/paper/protocol_r_125m_ablations_v1/iso_quality_summary.md`
- `reports/paper/protocol_r_125m_ablations_v1/iso_total_tokens_summary.csv`
- `reports/paper/protocol_r_125m_ablations_v1/frontier.csv`: raw ablation
  frontier with missing loss values for most continuation rows. Use
  `reports/paper/warmstart_paper_v1/plot_data/figure3_continuation_frontier.csv`
  for normalized continuation loss curves.
- `reports/paper/warmstart_paper_v1/plot_data/plot_data_manifest.json`: older
  intermediate plot-data manifest that marks 760M figure1/figure2 rows as
  omitted. The final manuscript plot path is `paper/plots`, generated by
  `scripts/75_make_paper_plots.py`.
- `reports/paper/protocol_r_760m_author_seed_v1/launch/launcher_summary.json`:
  dry-run launcher summary, not evidence that the tracked 760M S2/S3 curated
  results were produced by the local launcher in this checkout.
- `reports/paper/protocol_r_760m_eta_live_v1/eta_summary_combined.json` and
  related `protocol_r_760m_eta_live_v1*` reports: appendix/limitations context
  only, not main-table homogeneous cost evidence.
- `reports/paper/warmstart_125m_dryrun_exec` and
  `reports/paper/warmstart_125m_dryrun_ci`: smoke/dry-run artifacts only.
- `reports/paper/protocol_r_dryrun` and local-gate reports: orchestration
  diagnostics only.

## A0 Reconciliation Outcome

No headline historical number changed during this A0 pass. The main correction
for future work is procedural: revision-v2 gates must use current-pipeline
re-evaluation and paired seed-level analysis rather than ledger-only historical
comparisons.
