# Compute Budget Estimate: Efficient TTT-E2E via Warm-Starting from Pretrained Transformers

**Prepared for:** Prof. Vijay Madisetti, Georgia Tech CS8903
**Author:** Luis Montoya
**Date:** February 2026

---

## 1. Executive Summary

This document estimates the compute required to execute the full experimental
plan for our proposed paper: **"Efficient Test-Time Training for Long Context
via Warm-Starting from Pretrained Transformers."**

The paper answers the key open question flagged by the TTT-E2E authors
(Section 3.7): *Can pretrained full-attention Transformers be cheaply adapted
into TTT-E2E models, and how much meta-learning budget is needed to recover
from-scratch quality?* Nobody has answered this yet.

### The Experimental Pipeline

```
S0:  FA pretrained baseline               -> Quality ceiling
S1:  FA pretrained -> naive SWA swap       -> What does conversion lose? (negative control)
S2:  FA pretrained -> bridge adapt -> ext  -> Does warm-start work? How much budget?
S3:  From-scratch TTT-E2E                  -> Gold standard comparison
```

S1 and S2 ARE the contribution. S0 and S3 are comparison infrastructure.

### The Contribution is Cheap -- The Baselines are Expensive

The novel experiments (S1, S2, budget sweep) reuse the FA pretrained checkpoint
from S0, so their marginal cost is small. The expensive part is the from-scratch
E2E baseline (S3), needed only for comparison.

**GPU-hour breakdown -- Infrastructure vs. Contribution:**

|                               | 125M  | 760M   | 2.7B    |
|:------------------------------|------:|-------:|--------:|
| S0: FA pretrain + ext (B2)    | 4.5   | 112    | 1,055   |
| S3: E2E from scratch (B1)    | 14.2  | 360    | 3,370   |
| **Infrastructure subtotal**   | **18.7** | **472** | **4,425** |
| **S1: Naive SWA conversion**  | **0.6** | **13** | **128** |
| **S2 @ 10% (bridge + ext)**  | **2.2** | **56** | **533** |
| **S2 budget sweep (5/20/40%)**| **11.4** | **286** | **2,703** |
| **Contribution subtotal**     | **14.2** | **355** | **3,364** |
| Eval (all stages)             | ~1    | ~5     | ~30     |
| **Grand total (before buffer)** | **~34** | **~832** | **~7,819** |

The contribution represents only **~42%** of total compute -- the majority is
infrastructure baselines that any comparison study would need.

### What This Costs (with 30% overhead buffer)

| Plan | GPU-hrs | Vast.ai ($2.50/GPU-hr) |
|:-----|--------:|-----------------------:|
| 125M only | ~44 | **$110** |
| 125M + 760M | ~1,124 | **$2,810** |
| 125M + 760M + 2.7B (skip B1) | ~3,100 | **$7,750** |
| All three, full | ~11,250 | **$28,100** |

---

## 2. The Paper We Are Writing

### 2.1 Core Research Question

> Can pretrained full-attention Transformers be cheaply adapted into TTT-E2E
> models, and if so, how much meta-learning budget is needed to recover
> from-scratch quality?

This matters because:
- TTT-E2E training is **3.4x slower** than standard pretraining (the key barrier
  to adoption). If warm-starting works, it democratizes the approach.
- The TTT-E2E authors explicitly flag this as the most important open question.
- Nobody has answered it yet -- **first to answer wins.**

### 2.2 Experimental Design: The Causal Ladder

Each stage isolates one variable. The deltas between adjacent stages answer
specific questions:

| Stage | System | Question Answered |
|:------|:-------|:------------------|
| **S0** | FA pretrained -> ext 32K | What's the full-attention quality ceiling? |
| **S1** | FA pretrained -> SWA swap -> ext 32K | What does naive architecture conversion lose? |
| **S2** | FA pretrained -> meta-learn bridge (X% budget) -> ext 32K | Does warm-starting work? How much budget? |
| **S3** | From-scratch TTT-E2E -> ext 32K | Gold standard -- the target warm-start must approach |

- **S1 vs S0** = the cost of switching from FA to SWA without TTT
- **S2 vs S1** = what meta-learning buys over bare SWA conversion
- **S2 vs S3** = the warm-start quality tax (the key metric)
- **S2 cost vs S3 cost** = the compute savings (the key selling point)

### 2.3 Three Research Phases

**Phase 1: In-Family Warm-Start** (the paper's core)
- Use the TTT-E2E paper's own architecture family (125M, 760M, 2.7B)
- Full S0-S3 ladder at each scale
- Budget sweep for S2: adaptation at 5%, 10%, 20%, 40% of pretrain steps
- Evaluation: validation perplexity, RULER, NIAH, token-position loss breakdown

**Phase 2: External Pretrained Model** (the headline result)
- Import Qwen2.5-1.5B weights into the TTT-E2E architecture
- Add prime MLPs, convert FA->SWA, run adaptation bridge, extend to 32K
- Compare against Qwen2.5-1.5B with full attention and with naive SWA
- This is what upgrades the paper from "nice ablation" to "significant contribution"

**Phase 3: Mechanistic Analysis** (what separates "ran experiments" from "understand something")
- CKA similarity between warm-started and from-scratch prime MLP representations
- Gradient flow analysis during adaptation
- How quickly do randomly-initialized prime MLP params diverge from zero?
- Token-position loss breakdown: different failure patterns in warm-start vs scratch?

### 2.4 Success Criteria

- **Primary:** S2 @ 10% achieves validation loss within 0.05 nats of S3 (from-scratch)
- **Secondary:** >=60% wall-clock compute reduction in the TTT-E2E-specific stages
- **Stretch:** Warm-start also works on Qwen2.5-1.5B (Phase 2)

---

## 3. Estimation Methodology

### 3.1 FLOP Counting

We use the PaLM formula (Chowdhery et al., 2023), which accounts for both
parameterized operations and non-parameterized attention compute:

```
FLOPs_per_token = 6N + 12 * L * T * d
```

Where:
- **N** = model parameters (weight-matrix matmuls: QKV, output proj, SwiGLU, embedding)
- **L** = transformer layers, **T** = sequence length, **d** = hidden dimension
- **6N** = forward (2N) + backward (4N) for parameterized ops
- **12LTd** = forward + backward for attention logits and value reduction

Total training FLOPs: **C = D * (6N + 12LTd)**, where D = total tokens.

**Validation:** Predicts 8.18 GFLOP/token for 760M at 8K. MosaicML LLM-Foundry
benchmark on 8xH100 implies 8.23 GFLOP/token. Agreement within **1%**.

### 3.2 TTT-E2E Meta-Learning Overhead

E2E meta-learning is empirically **3.4x slower** than standard pretraining at 8K
(TTT-E2E paper, Section 3.4). At longer contexts, the overhead decreases because
FA scales O(T^2) while E2E's SWA + inner loop scales O(T):

| Context | E2E / FA Overhead | Source |
|--------:|------------------:|:-------|
| 8K | 3.4x slower | Paper Section 3.4 |
| 32K | ~1.7x slower | Log-linear interpolation (see 3.3) |
| 128K | 1.2x faster | Paper Section 3.4 |

### 3.3 Overhead Interpolation at 32K

Log-linear interpolation between the two paper anchor points (3.4x at 8K,
0.83x at 128K):

```
log2(overhead) at T = 1.77 + (-2.03 / 4) * log2(T / 8192)
overhead(32K) = 2^0.75 = 1.68x, rounded to 1.7x
```

### 3.4 GPU Throughput Model

```
GPU-hours = Total_FLOPs / (989 TFLOPS * MFU * 3600)
Wall-clock = GPU-hours / num_GPUs
```

H100 SXM5 peak BF16: **989 TFLOPS** (dense). MFU values from MosaicML
LLM-Foundry benchmarks (real H100 measurements):

| Model | MFU @ 8K | MFU @ 32K | Source |
|------:|---------:|----------:|:-------|
| 125M | ~30% | ~28% | Extrapolated (small models underutilize H100) |
| 760M | 34.84% | 31.84% | MosaicML benchmark |
| 2.7B | 40.31% | 28.84% | MosaicML benchmark |

### 3.5 vTrain Calibration

vTrain (Bang et al., MICRO 2024) validates our conservative MFU assumptions.
Their finding: naive Chinchilla-optimal estimates assuming 100% GPU utilization
are off by ~2.5x. Realistic utilization averages **35.56%**, consistent with our
30-40% range.

---

## 4. Model Architectures

From the TTT-E2E reference code (`ttte2e_reference/e2e/configs/`). All use
Llama-3 tokenizer (vocab=128,256), RoPE, SwiGLU, RMSNorm, tied embeddings.

| Config | Layers | Hidden | Heads | FFN | Suffix Len (E2E) | E2E FFN |
|:------:|-------:|-------:|------:|----:|:----------------:|--------:|
| 125M | 12 | 768 | 12 | 2,048 | 3 (25%) | 1,664 |
| 760M | 24 | 1,536 | 16 | 4,096 | 6 (25%) | 3,328 |
| 2.7B | 32 | 2,560 | 32 | 6,912 | 8 (25%) | 5,632 |

E2E FFN is reduced from FA FFN so that the added prime MLP doesn't inflate total
parameter count. Suffix blocks are the last ~25% of layers.

---

## 5. Training Configurations

### 5.1 Pretraining (8K, DCLM-filtered)

| Model | Steps | Seq Len | Batch | LR | **Tokens** |
|------:|------:|--------:|------:|---:|-----------:|
| 125M | 4,800 | 8,192 | 64 | 3e-3 | **2.52B** |
| 760M | 29,000 | 8,192 | 64 | 1.25e-3 | **15.2B** |
| 2.7B | 52,000 | 8,192 | 128 | 8e-4 | **54.5B** |

Chinchilla recipe (~20 tokens/param). AdamW, cosine schedule, 10% warmup.

### 5.2 Extension (32K, Books3) -- 5% of pretrain tokens

| Model | Steps | Seq Len | Batch | **Tokens** |
|------:|------:|--------:|------:|-----------:|
| 125M | 120 | 32,768 | 32 | **126M** |
| 760M | 725 | 32,768 | 32 | **759M** |
| 2.7B | 1,300 | 32,768 | 64 | **2.72B** |

### 5.3 Warm-Start Adaptation Bridge (8K, DCLM) -- X% of pretrain steps

This is **S2_ADAPT**: load FA pretrained weights, switch to SWA + E2E
architecture, train with meta-learning for a fraction of the full pretrain
budget. This is the core experiment variable.

| Budget | 125M steps | 125M tokens | 760M steps | 760M tokens | 2.7B steps | 2.7B tokens |
|-------:|-----------:|------------:|-----------:|------------:|-----------:|------------:|
| 5% | 240 | 126M | 1,450 | 760M | 2,600 | 2.73B |
| **10%** | **480** | **252M** | **2,900** | **1.52B** | **5,200** | **5.45B** |
| 20% | 960 | 503M | 5,800 | 3.04B | 10,400 | 10.9B |
| 40% | 1,920 | 1.01B | 11,600 | 6.08B | 20,800 | 21.8B |

---

## 6. Per-Token FLOPs

```
FLOPs/token = 6N + 12 * L * T * d
              ^^^   ^^^^^^^^^^^^^^
              weights  attention
```

| Model | 6N (GFLOP) | Attn @ 8K | **Total @ 8K** | Attn @ 32K | **Total @ 32K** |
|------:|-----------:|----------:|---------------:|-----------:|----------------:|
| 125M | 0.75 | 0.91 | **1.66** | 3.62 | **4.37** |
| 760M | 4.56 | 3.62 | **8.18** | 14.50 | **19.06** |
| 2.7B | 16.20 | 8.05 | **24.25** | 32.21 | **48.41** |

At 8K, attention FLOPs are comparable to weight FLOPs. At 32K, attention
dominates. The simple "6ND" rule underestimates by 40-120% at these contexts.

---

## 7. GPU-Hours Per Stage: The Full Pipeline

This is the heart of the document. Every stage of the experimental pipeline,
mapped to GPU-hours, with clear labels for what is **infrastructure** (needed for
comparison) vs. what is the **contribution** (our novel experiments).

### 7.1 Stage-by-Stage: 125M

| Stage | What | Mode | GPU-hrs | Category |
|:------|:-----|:----:|--------:|:---------|
| **S0_PRETRAIN** | FA pretrain 8K | pretrain | 3.9 | Infrastructure |
| **S0** | FA ext 32K (B2 result) | pretrain | 0.6 | Infrastructure |
| **S1** | Naive SWA ext 32K | pretrain | 0.6 | **CONTRIBUTION** |
| **S2_ADAPT @ 5%** | Warm-start bridge (5%) | meta | 0.7 | **CONTRIBUTION** |
| **S2_ADAPT @ 10%** | Warm-start bridge (10%) | meta | 1.3 | **CONTRIBUTION** |
| **S2_ADAPT @ 20%** | Warm-start bridge (20%) | meta | 2.7 | **CONTRIBUTION** |
| **S2_ADAPT @ 40%** | Warm-start bridge (40%) | meta | 5.3 | **CONTRIBUTION** |
| **S2 ext** (x4) | E2E ext 32K per sweep point | meta | 0.9 each | **CONTRIBUTION** |
| **S3_PRETRAIN** | E2E pretrain 8K (from scratch) | meta | 13.3 | Infrastructure |
| **S3** | E2E ext 32K (B1 result) | meta | 0.9 | Infrastructure |
| Eval | Perplexity + RULER all stages | -- | ~1 | Infrastructure |

**125M totals:**

| | GPU-hrs | % of total |
|:----------------------------|--------:|-----------:|
| Infrastructure (S0 + S3) | 19.7 | 58% |
| **Contribution (S1 + S2 sweep)** | **14.2** | **42%** |
| **Grand total** | **~34** | |

### 7.2 Stage-by-Stage: 760M

| Stage | What | Mode | GPU-hrs | Category |
|:------|:-----|:----:|--------:|:---------|
| **S0_PRETRAIN** | FA pretrain 8K | pretrain | 99 | Infrastructure |
| **S0** | FA ext 32K (B2 result) | pretrain | 13 | Infrastructure |
| **S1** | Naive SWA ext 32K | pretrain | 13 | **CONTRIBUTION** |
| **S2_ADAPT @ 5%** | Warm-start bridge (5%) | meta | 17 | **CONTRIBUTION** |
| **S2_ADAPT @ 10%** | Warm-start bridge (10%) | meta | 34 | **CONTRIBUTION** |
| **S2_ADAPT @ 20%** | Warm-start bridge (20%) | meta | 68 | **CONTRIBUTION** |
| **S2_ADAPT @ 40%** | Warm-start bridge (40%) | meta | 135 | **CONTRIBUTION** |
| **S2 ext** (x4) | E2E ext 32K per sweep point | meta | 22 each | **CONTRIBUTION** |
| **S3_PRETRAIN** | E2E pretrain 8K (from scratch) | meta | 338 | Infrastructure |
| **S3** | E2E ext 32K (B1 result) | meta | 22 | Infrastructure |
| Eval | Perplexity + RULER all stages | -- | ~5 | Infrastructure |

**760M totals:**

| | GPU-hrs | % of total |
|:----------------------------|--------:|-----------:|
| Infrastructure (S0 + S3) | 477 | 57% |
| **Contribution (S1 + S2 sweep)** | **355** | **43%** |
| **Grand total** | **~832** | |

### 7.3 Stage-by-Stage: 2.7B

| Stage | What | Mode | GPU-hrs | Category |
|:------|:-----|:----:|--------:|:---------|
| **S0_PRETRAIN** | FA pretrain 8K | pretrain | 927 | Infrastructure |
| **S0** | FA ext 32K (B2 result) | pretrain | 128 | Infrastructure |
| **S1** | Naive SWA ext 32K | pretrain | 128 | **CONTRIBUTION** |
| **S2_ADAPT @ 5%** | Warm-start bridge (5%) | meta | 158 | **CONTRIBUTION** |
| **S2_ADAPT @ 10%** | Warm-start bridge (10%) | meta | 315 | **CONTRIBUTION** |
| **S2_ADAPT @ 20%** | Warm-start bridge (20%) | meta | 630 | **CONTRIBUTION** |
| **S2_ADAPT @ 40%** | Warm-start bridge (40%) | meta | 1,261 | **CONTRIBUTION** |
| **S2 ext** (x4) | E2E ext 32K per sweep point | meta | 218 each | **CONTRIBUTION** |
| **S3_PRETRAIN** | E2E pretrain 8K (from scratch) | meta | 3,152 | Infrastructure |
| **S3** | E2E ext 32K (B1 result) | meta | 218 | Infrastructure |
| Eval | Perplexity + RULER all stages | -- | ~30 | Infrastructure |

**2.7B totals:**

| | GPU-hrs | % of total |
|:----------------------------|--------:|-----------:|
| Infrastructure (S0 + S3) | 4,455 | 57% |
| **Contribution (S1 + S2 sweep)** | **3,364** | **43%** |
| **Grand total** | **~7,819** | |

### 7.4 What Each Stage Answers (Research Questions)

| Comparison | Delta | Research Question |
|:-----------|:------|:------------------|
| S1 vs S0 | SWA swap only | Does naive architecture conversion preserve quality? (Expected: NO -- catastrophic drop proves warm-start bridge is necessary) |
| S2 vs S1 | + meta-learning bridge | What does the adaptation bridge buy over bare SWA? (Expected: large quality recovery) |
| S2 vs S3 | warm-start vs scratch | The warm-start quality tax -- how close to from-scratch at what budget? (Key result: Pareto curve) |
| S2 cost vs S3 cost | compute comparison | The practical value -- how much compute does warm-starting save? |
| S2 @ 5% vs 10% vs 20% vs 40% | budget sweep | Where is the diminishing-returns knee? (Expected: ~10-20% budget = most of the gains) |

### 7.5 The Cost of the Contribution Alone

If an advisor asks "how much compute do **your** experiments need?":

| | 125M | 760M | 2.7B |
|:------------------------------|------:|------:|--------:|
| S1 (negative control) | 0.6 | 13 | 128 |
| S2 @ 10% (adapt + ext) | 2.2 | 56 | 533 |
| S2 sweep (5%, 20%, 40% + ext) | 11.4 | 286 | 2,703 |
| **Contribution total** | **14.2** | **355** | **3,364** |
| **Contribution cost (Vast.ai)** | **$36** | **$888** | **$8,410** |

For a minimal result (S1 + S2 @ 10% only, no sweep):

| | 125M | 760M | 2.7B |
|:------------------------------|------:|------:|--------:|
| **S1 + S2 @ 10%** | **2.8** | **69** | **661** |
| **Cost** | **$7** | **$173** | **$1,653** |

These numbers are so small because the contribution experiments *reuse* the FA
pretrained checkpoint -- they don't need to train from scratch.

---

## 8. Phase 2: External Pretrained Model (Qwen2.5-1.5B)

This is what upgrades the paper from "nice ablation" to "significant
contribution." If warm-starting works on a state-of-the-art external model
(not just the paper's own architecture family), that's the headline result.

### 8.1 Pipeline

1. Import Qwen2.5-1.5B weights -> adapt architecture (add prime MLPs to suffix
   blocks, convert attention to SWA, reshape heads for GQA->MHA if needed)
2. S1: Naive SWA ext 32K (baseline -- no meta-learning)
3. S2_ADAPT: Short 8K meta-learning phase (10% budget as starting point)
4. S2: E2E extension to 32K
5. Compare against: Qwen2.5-1.5B with full attention at 32K

### 8.2 Compute Estimate

Qwen2.5-1.5B has ~1.5B parameters, 28 layers, hidden=1536, 12 heads. Closest
reference point: between our 760M and 2.7B configs.

Approximate FLOPs/token at 8K: 6*1.5e9 + 12*28*8192*1536 = 9.0 + 4.2 = **13.2 GFLOP**.

Assuming 37% MFU (interpolating between 760M and 2.7B benchmarks):

| Stage | Tokens | Overhead | GPU-hrs |
|:------|-------:|---------:|--------:|
| S1: SWA ext 32K | 750M | 1x | 11 |
| S2_ADAPT @ 10% | 3B (10% of ~30B Chinchilla) | 3.4x | 115 |
| S2: E2E ext 32K | 750M | 1.7x | 19 |
| **Phase 2 total** | | | **~145** |
| **Phase 2 cost** | | | **~$363** |

**Note:** Phase 2 does NOT require retraining Qwen from scratch. We import
the public checkpoint and only run the adaptation + extension stages. The FA
pretrain cost is zero (someone else paid for it -- that's the whole point).

---

## 9. Phase 3: Evaluation Compute

Evaluation is forward-pass only (no gradients) and relatively cheap:

| Evaluation | Per model | Notes |
|:-----------|----------:|:------|
| Validation perplexity (8K, 32K) | ~0.5 GPU-hr | 8 batches per context length |
| RULER (8K, 16K, 32K) | ~1-2 GPU-hrs | Synthetic retrieval tasks |
| NIAH / passkey retrieval | ~0.5 GPU-hr | Needle-in-a-haystack at multiple positions |
| Token-position loss breakdown | ~0.5 GPU-hr | Per-position NLL on eval set |
| **Per model total** | **~3-4 GPU-hrs** | |

For the full pipeline (S0, S1, S2 at 4 budgets, S3 = 7 models per scale):

| Scale | Eval GPU-hrs |
|------:|-------------:|
| 125M | ~25 |
| 760M | ~28 |
| 2.7B | ~28 |
| Qwen 1.5B | ~12 |

Total eval: **~93 GPU-hrs** (~$233). Negligible vs. training.

---

## 10. Mechanistic Analysis Compute (Phase 3)

These are lightweight analyses run on saved checkpoints:

| Analysis | Compute | Notes |
|:---------|--------:|:------|
| CKA similarity (prime MLPs warm-start vs scratch) | ~2 GPU-hrs | Forward passes through eval set, extract activations |
| Gradient flow during adaptation | ~4 GPU-hrs | Record gradient norms per layer during S2_ADAPT |
| Prime MLP divergence from init | ~1 GPU-hr | Compare warm-start vs scratch param trajectories |
| **Total mechanistic** | **~7 GPU-hrs** | |

Negligible cost. High impact on paper quality -- this is what separates
"we ran experiments" from "we understand something."

---

## 11. Grand Total: All Phases Combined

### 11.1 By Scale (with 30% overhead buffer)

| Component | 125M | 760M | 2.7B | Qwen 1.5B | **All** |
|:----------|-----:|-----:|-----:|-----------:|--------:|
| Infrastructure (S0+S3) | 19.7 | 477 | 4,455 | 0 | 4,952 |
| **Contribution (S1+S2 sweep)** | **14.2** | **355** | **3,364** | **145** | **3,878** |
| Eval + mechanistic | 32 | 35 | 35 | 19 | 121 |
| **Subtotal** | 66 | 867 | 7,854 | 164 | 8,951 |
| **+ 30% buffer** | **86** | **1,127** | **10,210** | **213** | **11,636** |

### 11.2 By Research Phase

| Phase | What | GPU-hrs | Cost (Vast.ai) |
|:------|:-----|--------:|---------------:|
| Phase 1 @ 125M | Full S0-S3 + sweep + eval | 86 | $215 |
| Phase 1 @ 760M | Full S0-S3 + sweep + eval | 1,127 | $2,818 |
| Phase 1 @ 2.7B | Full S0-S3 + sweep + eval | 10,210 | $25,525 |
| Phase 2 (Qwen) | S1 + S2 + eval | 213 | $533 |
| Phase 3 (mech.) | CKA + gradient + divergence | ~10 | $25 |
| **Total** | | **~11,636** | **~$29,090** |

---

## 12. Recommended Compute Plans

### Tier 1: Proof of Concept -- **$215** (or free on PACE)

**125M only.** Full S0-S3 ladder + 4-point budget sweep + eval.

- **GPU-hours:** ~86
- **Wall-clock on 8xH100:** ~6 hours (single session)
- **What you get:** Complete ablation table at one scale. The Pareto curve
  (quality vs. budget), S1 negative control, mechanistic analysis.
- **Sufficient for:** Workshop paper, course deliverable.
- **Risk:** Reviewers may question generalization beyond 125M.

### Tier 2: Two-Scale + Qwen -- **$3,600** (or free on PACE)

**125M full + 760M full + Qwen2.5-1.5B warm-start.** This is the recommended plan.

- **GPU-hours:** ~1,426
- **Wall-clock on 8xH100:** ~7.5 days
- **What you get:**
  - Multi-scale Pareto curves (125M + 760M)
  - Cross-architecture validation on Qwen (the headline result)
  - Mechanistic analysis
  - S1 negative controls at both in-family scales
- **Sufficient for:** Main conference (NeurIPS, ICML, ICLR).
- **This is the sweet spot: maximum paper quality per dollar.**

### Tier 3: Three-Scale Headline -- **$8,300** (or free on PACE)

**125M full + 760M full + 2.7B targeted + Qwen.** At 2.7B, run only S0 + S2 @ 10%
(skip the $8,400 from-scratch B1 baseline; cite the original paper's results).

- **GPU-hours:** ~3,400
- **Wall-clock on 8xH100:** ~18 days; on 32xH100: ~4.5 days
- **What you get:** Everything in Tier 2 + a 2.7B headline showing warm-start
  works at scale. Very competitive at top venues.

### Tier 4: Comprehensive -- **$29,000** (requires PACE or compute grant)

All three in-family scales with full B1 baseline + Qwen + 128K extension.

- **GPU-hours:** ~11,636
- **Only recommended if PACE allocation is available.**

### Decision Matrix

| Budget | Recommended Plan | Paper Strength |
|-------:|:-----------------|:---------------|
| $0 (PACE) | Tier 2 or 3 | Main conference |
| $200-250 | Tier 1 | Workshop / course |
| $3,000-4,000 | **Tier 2** | **Main conference** |
| $8,000-10,000 | Tier 3 | Top venue |
| $25,000+ | Tier 4 | Top venue + comprehensive appendix |

---

## 13. The "Solid" vs. "World-Class" Distinction

| Dimension | Solid Paper | World-Class Paper |
|:----------|:------------|:------------------|
| Scale | 125M + 760M in-family only | + 2.7B headline + Qwen cross-family |
| Metrics | Validation loss only | + RULER, NIAH, token-position breakdown |
| Budget sweep | 2-3 points | 4+ points with clear Pareto curve |
| Analysis | "S2 works at X% budget" | + mechanistic: CKA, gradient flow, why it works |
| Negative result | S1 fails (expected) | S1 failure mode taxonomy (what specifically breaks) |

**Tier 2 ($3,600) gets you to "solid."** Tiers 3-4 or PACE get you to "world-class."

---

## 14. Comparison with the Original TTT-E2E Paper

The TTT-E2E paper's headline: 2.7B trained with **162B tokens** (3x Chinchilla)
at 3.4x meta-learning overhead.

```
Their 2.7B pretrain alone: ~9,400 GPU-hours on H100
Their full paper (all scales, ablations): estimated 20,000-50,000 GPU-hours
  => $50K-$125K (funded by Hyperbolic Labs, SF Compute, Voltage Park, YC)
```

**Our Tier 2 plan (125M + 760M + Qwen) uses ~1,400 GPU-hours = 3-7% of their
budget.** This is possible because:

1. We use 1x Chinchilla recipe (not 3x)
2. Our contribution (warm-start) is inherently cheap -- it reuses pretrained weights
3. The expensive B1 from-scratch baseline is infrastructure, not the contribution
4. Phase 2 (Qwen) imports free public checkpoints -- zero pretrain cost

---

## 15. Key Assumptions and Risks

| Assumption | If Wrong | Mitigation |
|:-----------|:---------|:-----------|
| MFU 30-40% | 50% more GPU-hrs worst case | 30% buffer; validated vs MosaicML |
| 3.4x E2E overhead | Paper-validated on H100s | Well-established empirical number |
| 1.7x overhead at 32K | Interpolated | Conservative; could be lower |
| Vast.ai $2.50/GPU-hr | Prices fluctuate | Lambda ($3/hr) or PACE as backup |
| Warm-start actually works | Main research risk | Negative result is also publishable (why doesn't it work?) |
| Qwen architecture import works | GQA/head reshape may fail | `unify_dict_with_eqx_module` infrastructure exists; fall back to in-family only |
| Data available from GCS | May be Requester Pays | Pre-stage to HuggingFace Hub |
| PACE H100 availability | Queue contention | Off-peak scheduling; Vast.ai backup |

---

## 16. Data and Storage

| Dataset | Size (est.) | Used For |
|:--------|------------:|:---------|
| DCLM-filtered-8K (Zarr) | ~200-400 GB | Pretrain + adaptation |
| Books3 (Zarr) | ~50-100 GB | Extension |
| **Total** | **~300-500 GB** | |

| Model | Checkpoint Size | Total (10-15 saves) |
|------:|----------------:|--------------------:|
| 125M | ~0.5 GB | ~7 GB |
| 760M | ~3 GB | ~45 GB |
| 2.7B | ~11 GB | ~165 GB |

Disk per instance: **>=500 GB SSD.**

---

## 17. References

### Methodology
- **vTrain:** Bang et al., MICRO 2024.
  [arXiv:2312.12391](https://arxiv.org/abs/2312.12391) /
  [GitHub](https://github.com/VIA-Research/vTrain)
- **PaLM FLOPs formula:** Chowdhery et al., JMLR 2023.
  [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
- **Chinchilla scaling:** Hoffmann et al., NeurIPS 2022.
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **Transformer FLOPs:** A. Casson.
  [adamcasson.com](https://www.adamcasson.com/posts/transformer-flops)
- **Transformer Math 101:** EleutherAI.
  [blog.eleuther.ai](https://blog.eleuther.ai/transformer-math/)

### Benchmarks
- **MosaicML LLM-Foundry:** H100 throughput/MFU measurements.
  [GitHub](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md)
- **H100 Specifications:** NVIDIA.
  [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/)
- **llm.c GPT-2 124M:** Karpathy et al.
  [GitHub](https://github.com/karpathy/llm.c/discussions/481)

### TTT-E2E
- **Paper:** Behrouz et al., 2024.
  [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)
- **Code:** [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e)

### Cloud Pricing
- **Vast.ai:** [vast.ai/pricing](https://vast.ai/pricing)
- **H100 comparison:** [IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
