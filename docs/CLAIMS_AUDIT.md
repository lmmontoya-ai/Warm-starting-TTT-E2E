# Revision V2 Claims-Evidence Audit

Status: in progress, 2026-06-11.

This audit is the manuscript guardrail for the revised warm-start TTT-E2E
paper. Every abstract-level or contributions-level claim must appear here with
an evidence artifact, a strength label, and an explicit reviewer objection. If
the objection cannot be answered at the stated strength, the claim must be
weakened or deleted before submission.

Strength labels:

- `preregistered-confirmed`: committed before the run and confirmed by the
  matching analysis.
- `characterized`: measured on the canonical surface, but not a hypothesis
  test.
- `observed`: supported by the current evidence, but too narrow for a broad
  trend claim.
- `hypothesis-future-work`: discussion-only; must not appear as a contribution.

## Claims Table

| Claim for manuscript | Evidence artifact / run id | Strength | Section | Reviewer objection | Paper answer |
| --- | --- | --- | --- | --- | --- |
| Warm-starting a pretrained full-attention 125M seed into TTT-E2E via the bridge reaches Books32K loss `3.9169` on the current 64-batch manuscript surface. | `CANONICAL_RESULTS.md` revision-v2 64-batch reconciliation; `reports/revision_v2/current_eval64/s0_s1_s2_s3_books32k_jax_eval64_combined.json`; run id `revision_v2_current_pipeline64_v1`. | characterized | Results: main 125M comparison | The manuscript numbers moved between drafts. | The ledger explains the 8-batch drift as eval sampling noise and supersedes manuscript-era values with one float32 64-batch pipeline. |
| From-scratch TTT-E2E remains better than warm-start on LM loss at 125M: `S2 - S3 = 0.6447`. | `CANONICAL_RESULTS.md`; `reports/revision_v2/current_eval64/s0_s1_s2_s3_books32k_jax_eval64_combined.json`. | characterized | Results: quality trade-off | Does the revised framing hide that scratch is better? | No. The warm-start tax is reported in the main result and motivates the reuse/barrier-to-entry framing rather than an absolute quality-win framing. |
| The bridge contribution is isolated: no-bridge S2-minus is much worse than S2 under matched 32K extension, with mean effect `S2_MINUS - S2 = 2.0767`, 95% paired seed-bootstrap CI `[2.0500, 2.0970]`. | `PREREGISTRATION_REVISION_V2.md`; `CANONICAL_RESULTS.md`; `reports/revision_v2/e1_pairs_eval64/e1_bridge_effect_analysis_eval64.json`; paper run `revision_v2_e1_paired_v1`. | preregistered-confirmed | Results: bridge isolation | The original S2 vs S1 comparison only showed TTT-E2E beats SWA, not that the bridge matters. | E1 adds the missing control: FA seed -> TTT-E2E/SWA conversion -> 32K extension with no bridge. The only intended difference is the 8K bridge. |
| S2-minus trains smoothly but plateaus near loss `6.0`; the bridge is not merely rescuing a crashed run. | E1 training curves under `revision_v2_e1_paired_v1`; E1 advisor memo; `CANONICAL_RESULTS.md` E1 result. | observed | Mechanism / ablation discussion | Maybe S2-minus failed due to an implementation or optimization crash. | The training curves descend smoothly from high loss to a stable plateau, and five paired seeds show tight variance. |
| The S2-minus versus S2 gap is broadly uniform across the 32K window rather than only a tail-position failure. | E1 64-batch per-position analysis, `reports/revision_v2/e1_pairs_eval64/` artifacts; E1 advisor memo. | observed | Mechanism / per-position analysis | Maybe the bridge only fixes late-context positions. | Per-position deciles show a gap across the whole context window; the mechanism claim is global adaptation, not tail-only repair. |
| At 760M, the warm-start tax narrows across the two tested scales: `0.3188` at 760M vs `0.6447` at 125M. | `CANONICAL_RESULTS.md`; `reports/paper/protocol_r_760m_author_seed_v1/tables/s2_s3_warmstart_tax.csv`; 125M current 64-batch reconciliation. | observed | Results: second-scale check | Two scales do not establish a scaling law. | The text says "narrows across the two tested scales" and explicitly reserves sustained trend claims for future work with at least three scales. |
| The 760M result is quality-only; no homogeneous 760M warm-start cost claim is made in the main text. | `CANONICAL_RESULTS.md` 760M section; writing rule in `PREREGISTRATION_REVISION_V2.md`; manuscript limitations. | characterized | Limitations / cost accounting | The 760M warm-start path used mixed hardware and estimated costs. | The main tables keep only checkpoint-based quality numbers; mixed/ETA-style cost information is appendix or limitations-only. |
| The powered E2a retrieval proxy does not support a warm-start retrieval-complementarity headline. | `reports/revision_v2/e2a_analysis.md`; E2a manifests under paper run `revision_v2_e2a_proxy_v1`; `PREREGISTRATION_REVISION_V2.md` E2a hierarchy. | preregistered-confirmed | Results: capability proxy | The old n=16 probe hinted S2 might beat S3 on recall. | At n=500, the preregistered primary 125M S2-vs-S3 test is not significant and is directionally scratch-favoring (`0.060` vs `0.084`, p=`0.943`). The complementarity branch is deleted. |
| Absolute retrieval-proxy accuracy is low at these scales, and both warm-start and scratch largely sit near the retrieval floor. | `reports/revision_v2/e2a_analysis.md`. | characterized | Capability discussion | Low absolute accuracy might make the proxy uninformative. | The paper separates absolute capability from paired differences: the proxy is adequate to reject a large complementarity effect, but not used to claim strong retrieval capability. |
| The revised contribution is a reuse/entry-path characterization, not an "efficient beats scratch" claim. | `CANONICAL_RESULTS.md`; E1/E2a results; revised framing plan; cost tables. | characterized | Introduction / contributions | The original "efficient" framing contradicted the data. | The new claim treats the FA seed as a dual-use/sunk asset and reports the quality tax directly; scratch remains the quality reference. |
| At 125M, marginal warm-start cost is lower than scratch if the FA seed already exists; end-to-end savings are modest. | `CANONICAL_RESULTS.md` 125M GPU-hour accounting. | characterized | Cost-quality analysis | The cost advantage depends on favorable accounting. | The paper separates marginal reuse cost from full end-to-end cost and does not hide the `22.61` vs `25.71` GPU-hour end-to-end comparison. |
| The bridge-budget frontier is threshold-like at entry but budget-responsive afterward: 5% recovers `70.20%` of the E1 bridge effect, 20% reaches loss `3.6460`, and 40% reaches loss `3.4462`. | `CANONICAL_RESULTS.md` revision-v2 E3 section; `PREREGISTRATION_REVISION_V2.md` E3 section; paper run `revision_v2_e3_frontier_v1`; HF repo `Luxel/ttt-e2e-125m-results`. | characterized | Results: bridge-budget frontier | More bridge budget is confounded with more short-context training tokens, and E3 non-anchor points are single-seed. | The paper states E3 as a practical budget frontier, not a pure structural ablation. The 5% threshold-like label is reported as preregistered arithmetic and described as narrow because it lands just above the 70% boundary. |
| S2-minus improves under +1,440 continuation steps but remains far behind bridged paths: loss improves by `0.5102` to `5.4874`, still `1.5666` worse than the E1 S2 mean. | `CANONICAL_RESULTS.md` revision-v2 E3 section; `PREREGISTRATION_REVISION_V2.md` continuation rule; paper run `revision_v2_s2minus_cont_v1`; HF repo `Luxel/ttt-e2e-125m-results`. | characterized | Results: continuation | Maybe S2-minus only needed more long-context training. | The preregistered continuation read is `plateau claim weakened`, not closed. Extra long-context training helps direct conversion, but it does not erase the large bridge advantage. |
| The revision-v2 experimental artifacts are reproducible from checkpoint restores, fingerprinted datasets, protected reference snapshots, and committed run ids. | `docs/CHECKPOINT_DOWNLOADS.md`; `docs/REVISION_V2_PRIME_RUNBOOK.md`; `CANONICAL_RESULTS.md`; preflight manifests; HF repo `Luxel/ttt-e2e-125m-results`. | characterized | Reproducibility | The prior manuscripts mixed stale reports and inconsistent result surfaces. | A0 canonicalization, current-pipeline eval64, fingerprint sidecars, and run-id namespacing make the revision artifacts auditable. |

## Rejection-Critique Mapping

| Desk-reject critique | Evidence rows that answer it | Status |
| --- | --- | --- |
| The paper did not establish the contribution of the bridge. | Bridge-isolation E1 row; S2-minus training-curve row; per-position gap row; S2-minus continuation row. | Core complaint answered; continuation adds the required nuance that extra direct-extension training helps but does not erase the bridge advantage. |
| Ablations were limited. | E1 no-bridge control; E3 bridge-budget frontier; S2-minus continuation. | Complete for revision-v2 scope. |
| Evaluation scope was narrow and retrieval evidence underpowered. | E2a no-complementarity row; low-absolute-retrieval row. | Powered proxy complete; no complementarity headline. |
| Cost-quality trade-off and generality were overstated. | Reuse framing row; 125M cost-accounting row; 760M quality-only row; two-scale narrowing row. | Main text must keep cost/generality claims at these strengths. |

## Current Manuscript Spine

The revised paper should be a characterization study with one decisive mechanism
result:

1. A pretrained FA seed can be adapted into long-context TTT-E2E.
2. The 8K bridge stage is not incidental; E1 isolates a large bridge effect.
3. Warm-starting remains worse than scratch on LM loss, so the honest value
   proposition is reuse/lower barrier to entry, not absolute quality dominance.
4. Powered retrieval-proxy evaluation does not support a complementarity
   headline.
5. E3 shows that 5% bridge budget narrowly crosses the preregistered
   threshold-like boundary, while 20% and 40% continue improving quality.
6. S2-minus continuation weakens any hard plateau claim: extra long-context
   training helps direct conversion, but the result remains far behind bridged
   paths.

## Pending Updates

- Add final manuscript section pointers once the revised draft exists.
- Add plotted figure/table artifact paths after the paper tables and figures are
  regenerated from the revision-v2 ledger.
