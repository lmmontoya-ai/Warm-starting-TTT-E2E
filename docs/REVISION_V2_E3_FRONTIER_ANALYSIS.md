# Revision V2 E3 Frontier Analysis

Status: complete, 2026-06-12.

E3 was the final revision-v2 characterization run after the load-bearing E1
bridge-isolation result and the powered E2a retrieval proxy. It answers a
practical question: how much bridge budget is needed once the bridge itself is
known to matter?

## Inputs

- E1 no-bridge anchor, 0% bridge: `5.997589111328125`
- E1 warm-start anchor, 10% bridge: `3.9208450317382812`
- E1 bridge effect, `G = loss(0%) - loss(10%)`: `2.0767440795898438`
- New E3 seed: `seed001`
- Eval surface: Books32K, context `32768`, `64` eval batches, float32
  checkpoint restore pipeline.
- Hardware: Vast.ai 8x H200, pure data parallelism, `accum_steps=1`.
- HF artifact repo: `Luxel/ttt-e2e-125m-results`.

## Results

| Bridge budget | Stage | Loss | Recovery vs `G` |
| --- | --- | ---: | ---: |
| 0% | E1 `S2_MINUS_125M` five-seed mean | 5.997589111328125 | 0.0000 |
| 5% | `S2_BRIDGE_5PCT_125M`, seed001 | 4.539649963378906 | 0.7020 |
| 10% | E1 `S2_125M` five-seed mean | 3.9208450317382812 | 1.0000 |
| 20% | `S2_BRIDGE_20PCT_125M`, seed001 | 3.645965576171875 | 1.1324 |
| 40% | `S2_BRIDGE_40PCT_125M`, seed001 | 3.4461746215820312 | 1.2286 |

The 5% bridge arm narrowly meets the preregistered `threshold-like` label by
recovering `70.20%` of the E1 bridge effect. This is just above the `70%`
boundary, so the manuscript should say "narrowly meets" rather than imply a
wide margin.

The frontier is not saturated at 10%:

- 20% improves over the 10% E1 mean by `0.2748794555664062`.
- 40% improves over the 10% E1 mean by `0.47467041015625`.
- 40% remains worse than S3 by `0.1739463806152342`.

Preregistered labels:

- `threshold-like`: earned, narrowly.
- `smooth`: not earned, because the 5% recovery is just above 70%.
- `budget-hungry`: not earned, because 5% recovers more than 30%.
- `saturating`: not earned, because 20% and 40% are not within `0.10` of the
  10% anchor.

## S2-Minus Continuation

| Stage | Loss | Improvement vs E1 S2-minus mean | Gap vs E1 S2 mean |
| --- | ---: | ---: | ---: |
| `S2_MINUS_CONT_125M` | 5.487419128417969 | 0.510169982910156 | 1.5665740966796878 |

The preregistered continuation read is `plateau claim weakened`, because the
loss improves by more than `0.25`. The honest mechanism language is therefore:
direct FA-to-TTT-E2E conversion can keep improving with extra long-context
extension, but it remains far behind bridged warm-starting under this budget.

Do not claim that S2-minus is permanently stuck at the original E1 endpoint.
Do claim that the matched-budget bridge effect is large, replicated, and not
erased by the continuation run.

## Interpretation

The revised paper now has a clean characterization spine:

1. E1 isolates the bridge and shows that it is necessary under the matched
   standard extension budget.
2. E2a removes the old retrieval-complementarity branch; the powered proxy
   finds no warm-start retrieval advantage over scratch.
3. E3 shows that a small 5% bridge is already enough to recover most of the E1
   bridge effect, while larger bridge budgets continue to improve LM loss.

The practical message is not "warm-start beats scratch." Scratch remains the
quality reference. The message is that the bridge is a real adaptation stage
for reusing a pretrained FA seed, and bridge budget is a tunable cost-quality
knob.

## Caveats

- E3 changes bridge token budget and therefore changes upstream short-context
  training tokens. It is a practical budget frontier, not a pure structural
  ablation.
- The 5%, 20%, and 40% E3 points are single-seed. The 0% and 10% anchors are
  five-seed E1 means.
- The 5% threshold-like classification is close to the preregistered boundary.
- The continuation result weakens any strong plateau language for S2-minus.

## Cost And Artifacts

The seven E3/continuation training stages consumed
`25.394241899416234` observed aggregate GPU-hours. The Vast instance ran for
about `8.04` wall-clock hours at roughly `$28.97/hr`, for an approximate
session cost of `$233` before teardown.

The instance was destroyed after:

- HF export verification for all four final/eval stages.
- Local copy of the report and eval bundle into
  `reports/revision_v2/e3_frontier/remote_bundle/`.

The raw report bundle is intentionally treated as generated output; the tracked
source-of-truth summary is `CANONICAL_RESULTS.md`.
