# Research Plan: Efficient Test-Time Training for Long Context via Warm-Starting from Pretrained Transformers

**Author:** Luis Montoya
**Course:** CS8903, Georgia Tech
**Advisor:** Prof. Vijay Madisetti
**Date:** March 2026

---

## 1. Research Question

> When TTT-E2E is warm-started from a pretrained full-attention Transformer,
> what quality-compute trade-off does it achieve relative to from-scratch
> TTT-E2E, and how much additional continuation budget is required to close
> any remaining quality gap?

This question reframes the project around the most useful scientific distinction
for long-context training:

- **Practicality:** can warm-start deliver a strong long-context model at much
  lower marginal cost than scratch TTT-E2E?
- **Final quality:** does warm-start match or exceed the final quality of a
  model trained as TTT-E2E from scratch?

The paper will answer both parts directly rather than treating them as the same
claim.

### Why This Matters

- **TTT-E2E training is expensive at short context.** The original paper
  reports about `3.4x` training slowdown at `8K` relative to full attention.
  If warm-start preserves most of the quality while avoiding most of that
  compute, it becomes a practical route to long-context adaptation.
- **The critical comparison is causal, not anecdotal.** A good study has to
  separate plain full-attention continuation, naive architecture conversion,
  warm-started TTT-E2E, and from-scratch TTT-E2E. Otherwise it is impossible to
  say what the bridge stage actually buys.
- **Negative results are scientifically useful.** If warm-start improves over
  simple baselines but still fails to reach scratch TTT-E2E, that is not a null
  outcome. It establishes that practicality and final quality diverge for this
  architecture and budget regime.

---

## 2. Related Work

| Paper | Main idea | Warm-start? | TTT-E2E? | Controlled warm-start ladder? |
|:------|:----------|:-----------:|:--------:|:-----------------------------:|
| **TTT-E2E** (Tandon et al., 2025) | End-to-end meta-learning with inner SGD on prime MLPs | No | Yes | No |
| **AllMem** (Wang et al., 2026) | Warm-start into a Titans-style memory architecture | Yes | No | No |
| **In-Place TTT** (Feng et al., 2026) | NTP-aligned fast-weight updates in a different architecture | Yes | No | Partial |
| **TPTT** (Furfaro, 2025) | LoRA + linearized attention + memory gate | Yes | No | No |
| **SWAA** (2025) | Training-free FA→SWA conversion strategies | Yes | No | Strategy comparison only |
| **This work** | Warm-starting specifically into TTT-E2E with a causal ladder | Yes | Yes | Yes |

The claim of novelty is therefore narrow and precise: this project studies
warm-starting **specifically into TTT-E2E**, using a controlled causal ladder
and continuation ablations rather than a single endpoint.

---

## 3. Background: TTT-E2E in Brief

TTT-E2E replaces full attention in part of the model with an online adaptation
mechanism. Instead of using attention alone to preserve all useful information
from the past, it updates a restricted subset of parameters during sequence
processing using the ordinary next-token loss. The resulting system has three
important properties:

1. **Local sequence modeling remains strong.** A sliding-window backbone keeps
   strong local context behavior.
2. **Memory is partly moved into weight updates.** A suffix of the network
   contains prime MLP parameters that are updated through inner-loop gradient
   steps.
3. **Training is end to end.** The outer optimizer differentiates through the
   inner updates, so the model is trained to be a good initialization for its
   own test-time adaptation.

This makes TTT-E2E attractive for long context, but also creates the central
cost problem of the paper: end-to-end meta-learning is substantially more
expensive than ordinary pretraining at short context.

---

## 4. Experimental Design: The Causal Ladder

The core design is a four-stage comparison ladder. Each stage answers a
different question about long-context adaptation.

| Stage | System | Role |
|:------|:-------|:-----|
| **S0** | FA seed → direct `32K` extension | Full-attention long-context control |
| **S1** | FA seed → naive SWA conversion → `32K` extension | Negative control for bare architecture swap |
| **S2** | FA seed → `8K` E2E bridge → `32K` TTT-E2E extension | Warm-started TTT-E2E |
| **S3** | Scratch TTT-E2E `8K` pretrain → `32K` extension | From-scratch TTT-E2E gold standard |

The adjacent comparisons are the main scientific outputs:

| Comparison | Interpretation |
|:-----------|:---------------|
| `S1` vs `S0` | What is lost by a naive mechanism swap without a learned bridge? |
| `S2` vs `S1` | What does the warm-start bridge buy beyond bare SWA conversion? |
| `S2` vs `S3` | What is the warm-start tax, if any, at the final `32K` endpoint? |
| `S2` cost vs `S3` cost | What is the practical compute savings of warm-start? |

### Continuation Ablations

The ladder alone will not fully settle the fairness question, so the plan
includes two continuation ablations:

1. **Iso-quality continuation**
   - Continue the final `S2` checkpoint without changing optimizer state,
     schedule, data, batch size, or context length.
   - Stop when either:
     - `S2` matches the final `S3` held-out `32K` loss, or
     - the run hits a hard continuation cap.

2. **Iso-total-tokens continuation**
   - Continue the final `S3` checkpoint by exactly the extra upstream token
     budget consumed by the `S2` bridge stage.
   - Compare the resulting endpoint against `S2` at equal total branch tokens.

### Pre-Registered Outcome Logic

The paper will interpret outcomes under the following decision framework:

- **Warm-start practicality success**
  - `S2` beats both `S0` and `S1` on final `32K` quality, and
  - the marginal warm-start path is materially cheaper than the scratch path.
- **Warm-start final-quality success**
  - `S2` matches or beats `S3`, or
  - plain continuation of `S2` reaches `S3` with modest additional compute.
- **Warm-start practicality-only result**
  - `S2` clearly improves over `S0` and `S1`,
  - but `S3` remains the quality winner, even after continuation or fairer
    token-budget equalization.

This last case is still a publishable and useful outcome.

---

## 5. Checkpoint Provenance: Author-Shared 760M Weights

The `760M` study will use author-shared short-context checkpoints as fixed
seeds:

- **760M FA checkpoint**
  - full-attention pretraining at `8K`
  - seed for `S0`, `S1`, and `S2_ADAPT`
- **760M E2E checkpoint**
  - scratch TTT-E2E pretraining at `8K`
  - seed for `S3`

This changes the role of the `760M` study. It is no longer a full reproduction
of the expensive `8K` pretraining stages. Instead, it becomes a scale-up test of
the warm-start hypothesis under fixed imported seeds. That is acceptable for the
paper because the scientific question at `760M` is not whether we can afford to
retrain the whole short-context stack, but whether the warm-start versus scratch
story holds at the larger and more meaningful scale.

---

## 6. Experiment Groups

The project is organized into three groups, with one optional appendix group.

### 6.1. Group A: `125M` In-Family Ladder

This is the fully reproducible proof-of-concept scale. The plan is to run:

- `S0_PRETRAIN_FA_125M`
- `S0_125M`
- `S1_125M`
- `S2_ADAPT_125M`
- `S2_125M`
- `S3_PRETRAIN_E2E_125M`
- `S3_125M`

The purpose of this group is to establish the causal ladder cleanly, quantify
the quality gap between warm-start and scratch, and provide the first
publication-grade evidence for the paper.

### 6.2. Group B: `125M` Continuation Frontier

This group consists of the two ablations:

- `S2_125M_ISOQ`
  - continue `S2_125M` in blocks
  - evaluate every `120` steps
  - stop at target or hard cap
- `S3_125M_ISOTOK`
  - continue `S3_125M` by exactly the upstream token advantage used by `S2`

The purpose of this group is to convert the main comparison from a single
endpoint result into a frontier result.

### 6.3. Group C: `760M` Author-Seeded Ladder

This is the scale-up study. The plan is to run:

- `S0_760M`
- `S1_760M`
- `S2_ADAPT_760M`
- `S2_760M`
- `S3_760M`

The scientific purpose is to test whether the `125M` conclusion survives at
larger scale. The plan does **not** depend on reproducing `760M` short-context
pretraining from scratch.

#### Revised Matched Protocol for `8x H200`

The `760M` author-seeded ladder will run under an explicit revised matched
protocol on `8x H200` rather than under the faithful batch sizes. The smallest
passing reduced batch found by the smoke gates is:

- `global_batch_size = 8`

The ladder will preserve per-stage token budgets by scaling total steps:

- `S0_760M`, `S1_760M`, `S2_760M`, `S3_760M`
  - original: `725` steps at `global_batch_size = 32`
  - revised: `2900` steps at `global_batch_size = 8`
- `S2_ADAPT_760M`
  - original: `2900` steps at `global_batch_size = 64`
  - revised: `23200` steps at `global_batch_size = 8`

All other factors remain fixed:

- same author-provided `8K` seeds
- same datasets
- same architecture
- same context lengths
- same optimizer definitions

This keeps the `760M` study causally aligned with the `125M` revised-protocol
logic: the batch size is changed only to make the stages fit, and step counts
are increased so the token budgets remain comparable.

### 6.4. Optional Appendix Work

These are explicitly secondary to the main paper:

- **Qwen cross-architecture validation**
  - useful only after the in-family story is complete
- **SWAA comparison**
  - appendix only
  - framed as an adjacent deployment-oriented comparison, not a direct
    competitor to TTT-E2E warm-start

If time or budget becomes tight, these appendix experiments will be deferred
before any of the in-family `125M` or `760M` work is cut.

---

## 7. Evaluation Strategy

### 7.1. Primary Metrics

The paper will rely on metrics that are directly tied to the language-model
objective and runtime cost:

| Metric | Use |
|:-------|:----|
| **Validation loss / `loss_ce_mean`** | Primary quality metric at `8K` and `32K` |
| **Per-position token NLL** | Diagnose where long-context gains occur |
| **Tokens per second** | Throughput / efficiency reporting |
| **GPU-hours** | Practical cost accounting |
| **Observed tokens seen** | Fair token-budget accounting |

### 7.2. Evaluation Surfaces

Two evaluation surfaces will be treated as canonical:

| Surface | Dataset | Context | Role |
|:--------|:--------|--------:|:-----|
| `DCLM 8K` | `dclm_filter_8k/val` | `8192` | Upstream pretraining / bridge quality |
| `Books 32K` | `books3/val` | `32768` | Main long-context comparison |

### 7.3. Reporting Rules

To keep the reported numbers reproducible:

- all paper-stage results must come from durable checkpoints
- all reported stages must have a successful parity evaluation
- all reported stages must have verified export manifests if the stage is
  intended to be durable beyond the run surface
- final scalar aggregation will use float32 reduction, not BF16-heavy summary
  reduction
- token and runtime accounting will come from observed artifacts, not planned
  placeholders

### 7.4. Tables and Figures

The paper will prioritize four result displays:

1. **Main ladder table**
   - `S0`, `S1`, `S2`, `S3`
   - final `32K` loss and GPU-hours
2. **Cost table**
   - marginal warm-start cost vs scratch cost
   - end-to-end branch cost vs scratch cost
3. **Continuation frontier figure**
   - `S2` iso-quality frontier
   - `S3` iso-total-tokens frontier
4. **Per-position NLL figure**
   - representative long-context position-wise behavior

---

## 8. Methodology

### 8.1. Fairness

The plan distinguishes three kinds of fairness:

- **Matched downstream task**
  - compare models on the same `Books 32K` surface
- **Matched direct extension budget**
  - `S0`, `S1`, `S2`, `S3` all use the same direct `32K` extension budget
- **Matched total branch tokens**
  - addressed explicitly through the iso-total-tokens ablation

The paper will not claim “same budget” unless the specific budget axis is
named.

### 8.2. Continuation Policy

The continuation ablations will preserve the original training trajectory:

- `load_part=all`
- same optimizer state
- same learning-rate tail
- same data and context length
- no recipe changes during iso-quality or iso-total-tokens continuation

If later we want to test a reset or retuned schedule, that will be labeled as a
separate rescue experiment rather than folded into the plain-continuation
result.

### 8.3. Runtime Policy

The plan uses the local reproduction runtime only. Reference snapshots are
read-only audit targets and are not part of the canonical experiment runtime.

---

## 9. Mechanistic Analysis

The paper’s main claims will rest on the ladder and the continuation ablations.
Mechanistic analysis is secondary, but helpful if time allows.

Planned lightweight analyses:

- per-position NLL comparisons across `S0`, `S1`, `S2`, `S3`
- weight-drift analysis for the adapted suffix blocks
- representation similarity or checkpoint-distance analysis between warm-start
  and scratch endpoints

These analyses will only be included if they clarify the main practical versus
final-quality distinction. They are not required for the core paper claim.

---

## 10. Technical Implementation

### 10.1. Runtime

All training and evaluation will use the local in-repo runtime under `ttt/`.
The study will rely on:

- `training.runtime_mode=jax_train` for training
- `training.runtime_mode=jax_eval` for evaluation
- the warm-start registry as the single source of stage definitions

### 10.2. Stage Completion Contract

Each reported stage must satisfy:

1. successful training output
2. a valid `latest.json` checkpoint pointer
3. a successful parity eval manifest
4. a verified export manifest for durable reported stages

This is part of the plan, not post hoc cleanup.

### 10.3. Dataset Policy

- `dclm_filter_8k` for short-context pretraining / bridge adaptation
- `books3` for long-context extension
- fingerprint sidecars required for all staged datasets

### 10.4. Remote Surfaces

The plan assumes `8x H200` surfaces for the larger or more sensitive runs. Any
remote surface must pass CUDA/runtime preflight before a run is allowed to
count as a paper-grade attempt.

---

## 11. Budget

### 11.1. `125M` Working Budget

The `125M` program will be budgeted as:

| Component | Working budget |
|:----------|---------------:|
| Main ladder (`S0_PRETRAIN_FA` through `S3`) | `~50 GPU-hours` |
| Continuation ablations | `~8 GPU-hours` |
| Eval / reruns / slack | `~5 GPU-hours` |
| **Total `125M` budget** | **`~63 GPU-hours`** |

This is the budget for the full `125M` paper result, not just the main ladder.

### 11.2. `760M` Working Budget

Because the short-context FA and TTT-E2E seeds are author-provided, the `760M`
budget is dominated by the reduced-batch bridge and extension stages rather
than by full pretraining:

| Component | Working budget |
|:----------|---------------:|
| `S0_760M` + `S1_760M` | `~26 GPU-hours` |
| `S2_ADAPT_760M` + `S2_760M` | `~78–112 GPU-hours` |
| `S3_760M` | `~22 GPU-hours` |
| eval / reruns / slack | `~15 GPU-hours` |
| **Total `760M` budget** | **`~141–175 GPU-hours`** |

These are planning estimates, not claims. They assume the revised matched
protocol above rather than the faithful batch sizes.

---

## 12. Execution Schedule

The plan will proceed in the following order:

1. **`125M` main ladder**
   - complete and validate all `S0`–`S3` stages
2. **`125M` continuation ablations**
   - run iso-quality and iso-total-tokens
3. **`125M` paper draft**
   - freeze the main story and tables
4. **`760M` seed import and validation**
   - fetch author checkpoints
   - audit compatibility
   - validate short smoke runs
5. **`760M` ladder**
   - run `S0`, `S1`, `S2_ADAPT`, `S2`, `S3`
6. **Optional appendix experiments**
   - Qwen and/or SWAA only if budget and time remain

---

## 13. Paper Structure

The paper will be organized around the distinction between practicality and
final quality:

1. Introduction
2. Related work
3. Problem statement and solution overview
4. TTT-E2E warm-start design
5. Experimental implementation
6. `125M` ladder results
7. Continuation ablations
8. `760M` scale-up results
9. Discussion: practicality vs final quality
10. Limitations and future work

The main narrative will not be “warm-start succeeded” in a vague sense. It will
be “warm-start is useful, but its role must be stated precisely.”

---

## 14. Risk Register

| Risk | Likely impact | Mitigation |
|:-----|:--------------|:-----------|
| Runtime instability on rented H200 surfaces | Failed or misleading runs | Preflight checks; pinned runtime surfaces; only durable runs count |
| Warm-start appears weaker than expected | Threat to a simplistic success narrative | Treat as a practicality-only result, which remains publishable |
| `760M` author checkpoint compatibility issues | Delays scale-up study | Audit/import validation before real runs |
| Appendix experiments consume time | Weakens mainline progress | Keep Qwen and SWAA explicitly optional |
| Budget mismatch makes comparisons ambiguous | Weakens claims | Use continuation ablations and name the budget axis explicitly |

---

## 15. References

- Tandon et al. (2025), **TTT-E2E**
- Wang et al. (2026), **AllMem**
- Feng et al. (2026), **In-Place TTT**
- Furfaro (2025), **TPTT**
- Behrouz et al. (2024), **Titans**
- SWAA (2025)

The final paper draft will replace these shorthand references with full
bibliographic entries.
