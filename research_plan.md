# Research Plan: Efficient Test-Time Training for Long Context via Warm-Starting from Pretrained Transformers

**Author:** Luis Montoya
**Course:** CS8903, Georgia Tech
**Advisor:** Prof. Vijay Madisetti
**Date:** March 2026

---

## 1. Research Question

> Can pretrained full-attention Transformers be cheaply adapted into TTT-E2E
> models, and if so, how much meta-learning budget is needed to recover
> from-scratch quality?

TTT-E2E (Tandon, Dalal et al., 2025) explicitly flags pretrained initialization
as the most important open direction (Section 3.7): *"initializing TTT-E2E from
a pretrained Transformer without TTT."*

While concurrent work has explored warm-starting pretrained LLMs into various
TTT-like architectures — AllMem (Wang et al., 2026) for Titans-style memory,
TPTT (Furfaro, 2025) via linearized attention, and In-Place TTT (Feng et al.,
ICLR 2026) using MLP fast weights — **no published work has addressed
warm-starting specifically into the TTT-E2E architecture**, nor provided a
controlled causal-ladder study isolating the meta-learning bridge contribution
from naive architecture conversion.

This work is the **first controlled warm-start study of TTT-E2E specifically.**

### Why This Matters

- **Training cost is the key barrier.** TTT-E2E meta-learning is 3.4x slower
  than standard pretraining at 8K context. If warm-starting works, any existing
  pretrained model can be cheaply converted — democratizing the approach.
- **No existing work provides the controlled decomposition.** AllMem has no
  ablations separating SWA-only from SWA+TTT. TPTT trains on 500 samples with
  no baselines. In-Place TTT has ablations but targets a different architecture.
  Our causal ladder (S0→S1→S2→S3) isolates each variable.
- **Negative results are publishable.** If warm-starting fails for TTT-E2E
  (unlike AllMem/In-Place TTT for their respective architectures), understanding
  *why* is valuable — it reveals something specific about end-to-end
  meta-learning that makes it harder to warm-start.

---

## 2. Related Work

| Paper | Approach | Warm-starts? | TTT-E2E? | Controlled study? | Scale |
|:------|:---------|:------------:|:--------:|:-----------------:|:------|
| **TTT-E2E** (Tandon et al., 2025) | Inner SGD on prime MLPs, E2E meta-learning | No (from scratch only) | Yes (defines it) | N/A | 125M–2.7B |
| **AllMem** (Wang et al., 2026) | Freeze SWA+MLP, train Titans-style memory meta-params | Yes (Qwen3) | No (Titans memory) | No ablations | 0.6B–1.7B |
| **In-Place TTT** (Feng et al., ICLR 2026) | MLP projection as fast weights, NTP-aligned objective | Yes (Qwen3-4B) | No (own design) | Some ablations | 4B |
| **TPTT** (Furfaro, 2025) | LoRA + linearized attention (LiZA) + Memory-as-Gate | Yes (LoRA) | No (Titans-style) | No ablations | 0.3B–7B |
| **qTTT** (Bansal et al., 2025) | Gradient updates on Q projections only at test time | No (test-time only) | No | No | 1.7B–32B |
| **LaCT** (Zhang et al., 2025) | Large-chunk TTT (2K–1M tokens), 40% model as state | No (from scratch) | No (own design) | Some ablations | 14B |
| **Titans** (Behrouz et al., 2024) | Neural long-term memory with L2 reconstruction loss | No | No | N/A (foundational) | Up to 2B |
| **SWAA** (2025) | Training-free FA→SWA conversion + optional LoRA | Yes (architecture swap) | No | Strategy comparison | Instruction models |
| **This work** | **S0→S1→S2→S3 causal ladder for TTT-E2E warm-start** | **Yes** | **Yes** | **Full controlled study** | **125M–760M + Qwen** |

**Our unique contributions vs. the field:**
1. TTT-E2E-specific (inner SGD on prime MLPs with full bi-level meta-learning)
2. Causal ladder isolating conversion loss, bridge recovery, and warm-start tax
3. Budget sweep (5/10/20/40%) producing a Pareto frontier
4. Multi-scale evidence (125M + 760M) with preregistered acceptance targets

---

## 3. Background: TTT-E2E in Brief

TTT-E2E (Test-Time Training End-to-End) replaces self-attention in a subset of
transformer layers with a learned inner-loop that updates *prime MLP* parameters
at test time. The key ideas:

1. **Prefix/suffix block split.** The first ~75% of layers ("prefix blocks") use
   Sliding Window Attention (SWA). The last ~25% ("suffix blocks") additionally
   contain *prime MLPs* — small feed-forward networks whose weights are updated
   by the inner loop.

2. **Meta-learning.** During training, an inner SGD loop updates prime MLP weights
   on each mini-batch. The outer optimizer (AdamW) differentiates *through* the
   inner updates (gradients of gradients). This is what makes training 3.4x slower.

3. **Linear-time inference.** At inference, the inner loop replaces O(T²) attention
   with O(T) updates. This gives TTT-E2E a massive throughput advantage at long
   contexts (128K+), while matching or exceeding Transformer quality.

4. **The training cost problem.** A 760M TTT-E2E model requires ~15B tokens at
   8K context with meta-learning. At 3.4x overhead, this costs ~360 GPU-hours on
   H100. If we could warm-start from an existing FA checkpoint and only run
   meta-learning for 10% of the budget, we'd save ~90% of that cost.

---

## 4. Experimental Design: The Causal Ladder

Each stage isolates exactly one variable. The deltas between adjacent stages
answer specific sub-questions:

| Stage | System | Role |
|:------|:-------|:-----|
| **S0** | FA pretrained → ext 32K | Matched FA control (note: TTT-E2E can beat FA at long context, so S0 is not a quality ceiling) |
| **S1** | FA pretrained → naive SWA swap → ext 32K | Negative control: what does bare architecture conversion lose? |
| **S2** | FA pretrained → meta-learn bridge (X% budget) → ext 32K | The contribution: does warm-starting work? How much budget? |
| **S3** | TTT-E2E from scratch → ext 32K | Gold standard target that warm-start must approach |

**Key comparisons:**

| Comparison | What It Reveals |
|:-----------|:----------------|
| S1 vs S0 | Cost of naive SWA conversion (expected: catastrophic drop — proves bridge is necessary) |
| S2 vs S1 | What meta-learning bridge buys over bare SWA |
| S2 vs S3 | Warm-start quality tax at each budget level |
| S2 cost vs S3 cost | Compute savings — the headline practical number |
| S2 @ 5% vs 10% vs 20% vs 40% | Diminishing-returns knee on the Pareto frontier |

S1 and S2 **are** the contribution. S0 and S3 are comparison infrastructure.

### Preregistered Acceptance Targets

Hard targets surfaced in the manuscript:

- **Primary:** S2 @ 10% achieves 32K validation loss within **0.05 nats** of S3
- **Secondary:** Warm-start achieves **>=60% wall-clock reduction** vs from-scratch TTT-E2E
- **Tertiary:** Per-position NLL curve for S2 shows the same late-position improvement pattern as S3 (the TTT-E2E signature)

If these fail, the paper reports the negative result with mechanistic analysis.

### Fairness Protocol

All comparisons follow matched reporting:

- **Token-matched:** Equal total training tokens across compared runs
- **Wall-clock-matched:** Equal total GPU-hours across compared runs
- **Token/byte accounting:** Track both token count and raw-byte count when comparing across tokenizers (Llama-3 vs Qwen)

This is especially critical for Qwen cross-architecture experiments where
tokenizer efficiency differs.

---

## 5. Checkpoint Provenance: Author-Shared 760M Weights

The TTT-E2E authors (Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja,
Marcel Rød, Yu Sun et al.) generously shared their pretrained 760M checkpoints
with us upon request:

- **760M FA checkpoint** — full-attention pretrained at 8K on DCLM (S0_PRETRAIN)
- **760M E2E checkpoint** — TTT-E2E pretrained from scratch at 8K (S3_PRETRAIN)

These checkpoints eliminate the two most expensive stages in our pipeline:

| Eliminated Stage | GPU-hours Saved | Money Saved |
|:-----------------|----------------:|------------:|
| 760M FA pretrain from scratch (S0_PRETRAIN) | 99 | $198 |
| 760M E2E from scratch (S3_PRETRAIN) | 338 | $676 |
| **Total saved** | **437** | **$874** |

---

## 6. Experiment Groups

The paper's center of gravity is the **125M + 760M in-family causal ladder**.
Qwen is one compact cross-architecture validation. SWAA is appendix material.

### 6.1. 125M — Full Ladder (from scratch, fully reproducible)

| Stage | Description | Mode | GPU-hrs | Cost |
|:------|:------------|:----:|--------:|-----:|
| S0_PRETRAIN | FA pretrain 8K (Chinchilla: 2.52B tokens) | pretrain | 3.9 | $8 |
| S0 | FA ext 32K (Books3, 126M tokens) | pretrain | 0.6 | $1 |
| S3_PRETRAIN | E2E from scratch 8K (2.52B tokens, 3.4x overhead) | meta | 13.3 | $27 |
| S3 | E2E ext 32K (126M tokens, 1.7x overhead) | meta | 0.9 | $2 |
| S1 | Naive SWA ext 32K (negative control) | pretrain | 0.6 | $1 |
| S2_ADAPT | Warm-start bridge sweep (5%, 10%, 20%, 40% of pretrain) | meta | 10.0 | $20 |
| S2 | E2E ext 32K (×4 sweep points) | meta | 3.6 | $7 |
| Eval | All stages, all metrics | eval | ~1 | $2 |
| **125M subtotal** | | | **~34** | **$68** |

Architecture: 12 layers, hidden=768, 12 heads, FFN=2048. Suffix=3 (25%).
E2E FFN=1664. Llama-3 tokenizer (vocab=128,256).

**What 125M delivers:**
- Complete 4-point Pareto curve (quality vs. adaptation budget)
- S1 negative control proving the bridge is necessary
- Full reproducibility — anyone can replicate for ~$68

### 6.2. 760M — Full Ladder (author-shared checkpoints)

| Stage | Description | Mode | GPU-hrs | Cost |
|:------|:------------|:----:|--------:|-----:|
| ~~S0_PRETRAIN~~ | ~~FA pretrain 8K~~ | | ~~99~~ | **FREE** (author checkpoint) |
| S0 | FA ext 32K (from author FA ckpt) | pretrain | 13 | $26 |
| ~~S3_PRETRAIN~~ | ~~E2E from scratch 8K~~ | | ~~338~~ | **FREE** (author checkpoint) |
| S3 | E2E ext 32K (from author E2E ckpt) | meta | 22 | $44 |
| S1 | Naive SWA ext 32K (negative control) | pretrain | 13 | $26 |
| S2_ADAPT @ 10% | Warm-start bridge (1.52B tokens) | meta | 34 | $68 |
| S2_ADAPT @ 20% | Warm-start bridge (3.04B tokens) | meta | 68 | $136 |
| S2 | E2E ext 32K (×2 sweep points) | meta | 44 | $88 |
| Eval | All stages | eval | ~5 | $10 |
| **760M subtotal** | | | **~199** | **$398** |

Architecture: 24 layers, hidden=1536, 16 heads, FFN=4096. Suffix=6 (25%).
E2E FFN=3328. Llama-3 tokenizer (vocab=128,256).

**What 760M delivers:**
- Multi-scale evidence — confirms 125M trends at 6x scale
- Two sweep points (10%, 20%) sufficient to show the trend
- Author-trained S0 and S3 baselines — maximum credibility

### 6.3. Qwen2.5 — Compact Cross-Architecture Validation

One compact experiment proving warm-starting generalizes beyond the paper's own
model family. **Not** a separate paper's worth of work — one paragraph in
the results section.

Both Qwen2.5-0.5B (cheap validation) and Qwen2.5-1.5B (if budget allows).

| Stage | Description | GPU-hrs | Cost |
|:------|:------------|--------:|-----:|
| S1 | Naive SWA ext 32K (negative control) | ~5 | $10 |
| S2_ADAPT @ 10% | Meta-learning bridge at 8K | ~14 | $28 |
| S2 | E2E ext 32K | ~5 | $10 |
| Eval | Same metrics as in-family | ~5 | $10 |
| **Qwen 0.5B subtotal** | | **~29** | **$58** |

Qwen 1.5B as stretch goal (~$120 additional).

GQA expansion at import: Qwen uses 14Q/2KV (0.5B) or 12Q/2KV (1.5B). We
replicate each KV head to match Q heads — mathematically lossless for init.

### 6.4. SWAA Comparison (appendix, not core)

Per reviewer feedback, the SWAA comparison is adjacent but not a fair
head-to-head with TTT-E2E warm-start. SWAA operates on instruction/thinking
models with generation-based benchmarks (LongMemEval, LLM-as-judge), while
our story is base-LM, loss-centric, and training-budget-centric.

**If included**, SWAA goes in appendix with explicit framing: *"SWAA and TTT-E2E
warm-start solve different subproblems — SWAA adapts deployment (inference
efficiency at long context), while warm-start adapts training (reducing
meta-learning compute). We include this comparison to contextualize the
quality-compute trade-off, not as a direct competitor."*

| Stage | Description | GPU-hrs | Cost |
|:------|:------------|--------:|-----:|
| SWAA conversion | Training-free SWA strategies (sinks, interleaving) | ~0 | $0 |
| SWAA + LoRA | 1-epoch LoRA fine-tuning | ~12 | $24 |
| Test-time LoRA | Inference-time adaptation | ~15 | $30 |
| Eval | Appendix-only matched protocol, or LongMemEval/RULER reported separately | ~20 | $40 |
| Buffer | | ~15 | $30 |
| **SWAA subtotal** | | **~62** | **$124** |

---

## 7. Evaluation Strategy

### 7.1. Metrics

**Tier 1 — Core (built into the in-repo workflow, free):**

| Metric | What It Answers | Source |
|:-------|:----------------|:-------|
| **Validation loss** (cross-entropy, 8K + 32K) | Overall LM quality — the primary comparison | In-repo eval/runtime manifests (`jax_eval` or staged eval outputs) |
| **Per-position token NLL curve** | Where in context does the model struggle? Does TTT's inner loop help at late positions? | In-repo eval artifacts / staged eval outputs |
| **Training loss curve** | Convergence speed — does warm-start converge faster? | W&B logging every step |

**Tier 2 — Publication-grade benchmark harnesses (partially implemented):**

| Metric | What It Answers | Current status |
|:-------|:----------------|:---------------|
| **Perplexity** (exp of val loss, 8K/16K/32K) | Standard LM metric, comparable across papers | Already derivable from validation loss; no separate paper script required unless we want standalone reporting |
| **NIAH** (needle-in-a-haystack, multiple positions + lengths) | Can the model retrieve information buried in long context? | Proxy implementation exists in `scripts/11_external_eval.py`; final paper needs a real harness or explicit proxy labeling |
| **Throughput** (tokens/sec, training + inference) | Compute efficiency — the practical selling point | Already logged by `jax_train` / `jax_eval` and aggregated in manifests |

**Tier 3 — Stretch (strengthens paper, not required for workshop):**

| Metric | What It Answers | Status |
|:-------|:----------------|:-------|
| **CKA similarity** (warm-start vs scratch representations) | Do warm-started models converge to similar representations? | Lightweight — forward passes + CKA library |
| **RULER** (real multi-needle, variable tracking) | Standard long-context benchmark | Full harness needed — complex |

### 7.2. Benchmark Honesty

The repo currently has proxy NIAH evaluation (`scripts/11_external_eval.py`)
and RULER-style aggregation (`ruler_runner.py`) derived from those proxy rows.
**These are proxies, not real benchmarks.** The paper will either:

- (a) Implement real NIAH/RULER harnesses, or
- (b) Explicitly label all results as "proxy" with methodology disclosure

No ambiguous benchmark claims.

### 7.3. What We Compare (the paper's tables)

**Table 1: In-family causal ladder** (the main result)

```
                    Val Loss  Val Loss  NIAH    NIAH    GPU-hrs  Compute
                    @8K       @32K      @8K     @32K    total    vs S3
125M
  S0 (FA control)    X.XX      X.XX     XX%     XX%      4.5      —
  S1 (naive SWA)     X.XX      X.XX     XX%     XX%      0.6      —
  S2 @5%             X.XX      X.XX     XX%     XX%      1.3     9%
  S2 @10%            X.XX      X.XX     XX%     XX%      2.2    15%
  S2 @20%            X.XX      X.XX     XX%     XX%      3.6    25%
  S2 @40%            X.XX      X.XX     XX%     XX%      6.2    44%
  S3 (from scratch)  X.XX      X.XX     XX%     XX%     14.2   100%
760M
  S0 (FA control)    X.XX      X.XX     XX%     XX%       13      —
  S1 (naive SWA)     X.XX      X.XX     XX%     XX%       13      —
  S2 @10%            X.XX      X.XX     XX%     XX%       47     13%
  S2 @20%            X.XX      X.XX     XX%     XX%       90     25%
  S3 (from scratch)  X.XX      X.XX     XX%     XX%      360   100%
```

**Table 2: Cross-architecture validation** (one compact paragraph)

```
Qwen2.5-0.5B       Val Loss @32K   NIAH @32K   GPU-hrs
S1 (naive SWA)          X.XX          XX%          ~5
S2 @10%                 X.XX          XX%         ~24
```

### 7.4. Key Figures

1. **Pareto curve** — x = adaptation budget (% of pretrain), y = 32K val loss.
   One line per scale (125M, 760M). Shows diminishing-returns knee.

2. **Per-position NLL curves** — overlaid lines for S0/S1/S2/S3 showing the
   TTT-E2E signature (late-position improvement). Ties directly back to the
   original paper's core story.

3. **Causal ladder bar chart** — side-by-side bars for S0/S1/S2/S3 at both
   scales. Visual proof that S1 drops and S2 recovers.

### 7.5. Benchmark Datasets Needed

| Dataset | Use | Source | Download needed? |
|:--------|:----|:-------|:-----------------|
| DCLM val split | Val loss + perplexity | Training data Zarr (already staged) | No |
| Books3 val split | Val loss at 32K | Training data Zarr (already staged) | No |
| NIAH synthetic | Needle retrieval at positions | Generated programmatically | No |
| RULER (stretch) | Multi-needle, variable tracking | RULER GitHub | Small download |

No external benchmark downloads are strictly required. Val loss and NIAH can
be computed from the training data + synthetic generation.

---

## 8. Methodology

### 8.1. FLOP Counting

PaLM formula (Chowdhery et al., 2023):

```
FLOPs_per_token = 6N + 12 * L * T * d
```

| Model | 6N (GFLOP) | Attn @ 8K | Total @ 8K | Attn @ 32K | Total @ 32K |
|------:|-----------:|----------:|-----------:|-----------:|------------:|
| 125M  | 0.75 | 0.91 | 1.66 | 3.62 | 4.37 |
| 760M  | 4.56 | 3.62 | 8.18 | 14.50 | 19.06 |

### 8.2. TTT-E2E Meta-Learning Overhead

| Context | E2E / FA Overhead | Source |
|--------:|------------------:|:-------|
| 8K | 3.4x slower | Paper Section 3.4 |
| 32K | ~1.7x slower | Log-linear interpolation |
| 128K | 1.2x faster | Paper Section 3.4 |

### 8.3. GPU Throughput

```
GPU-hours = Total_FLOPs / (989 TFLOPS × MFU × 3600)
```

H100 SXM5 peak BF16: 989 TFLOPS. MFU from MosaicML benchmarks:

| Model | MFU @ 8K | MFU @ 32K |
|------:|---------:|----------:|
| 125M  | ~30% | ~28% |
| 760M  | 34.84% | 31.84% |

### 8.4. Training Configurations

**Pretraining (8K, DCLM):**

| Model | Steps | Seq Len | Batch | Tokens |
|------:|------:|--------:|------:|-------:|
| 125M  | 4,800 | 8,192 | 64 | 2.52B |
| 760M  | 29,000 | 8,192 | 64 | 15.2B |

Chinchilla recipe (~20 tokens/param). AdamW, cosine schedule, 10% warmup.

**Extension (32K, Books3) — 5% of pretrain tokens:**

| Model | Steps | Seq Len | Batch | Tokens |
|------:|------:|--------:|------:|-------:|
| 125M  | 120 | 32,768 | 32 | 126M |
| 760M  | 725 | 32,768 | 32 | 759M |

**S2 Adaptation Bridge (8K, DCLM) — X% of pretrain steps:**

| Budget | 125M steps | 125M tokens | 760M steps | 760M tokens |
|-------:|-----------:|------------:|-----------:|------------:|
| 5%  | 240 | 126M | — | — |
| 10% | 480 | 252M | 2,900 | 1.52B |
| 20% | 960 | 503M | 5,800 | 3.04B |
| 40% | 1,920 | 1.01B | — | — |

(5% and 40% run only at 125M; 10% and 20% run at both scales.)

---

## 9. Mechanistic Analysis

Lightweight analyses on saved checkpoints, forward passes only (~$20 extra):

| Analysis | Compute | What It Reveals |
|:---------|--------:|:----------------|
| CKA similarity (warm-start vs scratch prime MLPs) | ~2 GPU-hrs | Do warm-started and from-scratch models converge to similar representations? |
| Per-position loss curves | ~1 GPU-hr | Different failure modes in warm-start vs scratch at various token positions |
| Prime MLP weight divergence from init | ~1 GPU-hr | How quickly do randomly-initialized prime MLP parameters specialize? |
| Gradient flow during adaptation | ~4 GPU-hrs | Where does the meta-learning gradient concentrate? Which layers adapt first? |

This separates "we ran experiments" from "we understand something." Also
provides diagnostic tools if warm-starting partially fails.

---

## 10. Technical Implementation

### 10.1. Runtime

All training and evaluation runs use the **local in-repo runtime** under
`ttt/`, not the read-only snapshot runtime in `ttte2e_reference/`. This keeps
the research workflow aligned with `AGENTS.md` and with the current staged
orchestration/evaluation pipeline.

Runtime modes:

- `training.runtime_mode=simulate` for orchestration dry runs
- `training.runtime_mode=token_stats` for local pilot runs and cheap eval proxies
- `training.runtime_mode=jax_train` for native in-repo JAX training
- `training.runtime_mode=jax_eval` for native in-repo JAX evaluation

Invoked via:
```
uv run --exact python scripts/23_warmstart_registry.py \
    --paper-run-id <paper_run_id> \
    --exp-folder <exp_folder> \
    --runtime-mode <simulate|token_stats|jax_train|jax_eval> \
    --dclm-root <path> \
    --books-root <path>
```

The reference snapshots in `ttte2e_reference/` and `swaa_reference/` remain
read-only comparison artifacts. Custom configs, runtime code, and experiment
orchestration live in this repo.

### 10.2. Weight Import (Qwen)

For Qwen experiments, the default workflow follows the local runtime toolchain:

- `scripts/07_prepare_external_models.py` for model profiles
- `scripts/15_import_hf_checkpoint.py` for import-root checkpoints
- `scripts/16_audit_checkpoint_compat.py` for compatibility audits
- `scripts/17_probe_warmstart_init.py` for initialization sanity checks

For later direct full-weight bootstrap work, `scripts/26_import_qwen_to_orbax.py`
provides a heavier import path. That importer is useful for validation and
legacy bootstrap experiments, but it is **not** the default phase-1 workflow.

Direct full-weight import design notes:

- **GQA expansion:** Replicate each KV head `num_q_heads / num_kv_heads` times.
  Mathematically lossless for initialization.
- **Weight mapping:** HF `[out, in]` → local model tensor layout (transpose where required).
- **Validation:** Logit or checkpoint-compatibility comparison against the HF source model.

**Status:** Import script built and tested for 1.5B. Needs adaptation for 0.5B.

### 10.3. Data

- **DCLM-filtered** (8K pretraining/adaptation): Tokenized Zarr arrays from GCS
- **Books3** (32K extension): Tokenized Zarr arrays from GCS
- **Staging:** Downloaded from GCS, uploaded to private HF Hub repo
- **Tokenizers:**
  - 125M/760M: Llama-3 (vocab=128,256) — same as original paper
  - Qwen: Qwen2.5 (vocab=151,936) — verified identical across family sizes

### 10.4. Checkpoint Format

The current research workflow uses runtime-specific checkpoints with a shared
`latest.json` lineage contract:

- **Import/token-stats stages:** JSON payload checkpoints via `Phase1Checkpointer`
- **Native JAX stages:** pickle-pytree state plus JSON sidecar via `JaxCheckpointer`
- **Legacy/bootstrap interoperability:** Orbax-backed checkpoints only where a
  direct conversion path is still being validated

---

## 11. Budget

| Component | GPU-hrs | Cost ($2/GPU-hr) |
|:----------|--------:|------------------:|
| 125M full ladder (4-point sweep) | 34 | $68 |
| 760M ladder (2-point sweep, author ckpts) | 199 | $398 |
| Qwen 0.5B (compact validation) | 29 | $58 |
| SWAA (appendix, optional) | 62 | $124 |
| Qwen 1.5B (stretch) | ~60 | ~$120 |
| **Core subtotal** | **262** | **$524** |
| + 30% buffer | 79 | $157 |
| **Core grand total** | **~341** | **~$681** |
| **With all stretch goals** | **~499** | **~$998** |

Hardware: 8×H100 SXM5 on Vast.ai ($16/hr total = $2/GPU-hr).

Wall-clock on 8×H100: ~1 week total.

---

## 12. Execution Schedule

### Phase 1: Engineering (~2-3 days, local, no GPU)

| Task | Status | Notes |
|:-----|:-------|:------|
| Model profiles + configs (WS0) | Done | `configs/model/qwen2_5_{0_5b,1_5b}.yaml` |
| HF → Orbax import script (WS1) | Done | `scripts/26_import_qwen_to_orbax.py`, tested for 1.5B |
| Tokenizer compatibility (WS2) | Done | Qwen family shares tokenizer |
| Launcher scripts | Pending | 125M, 760M, Qwen launchers needed |
| Author checkpoint integration | Pending | Format verification + conversion |
| 125M warm-start configs | Pending | S1 + S2 sweep configs |
| Evaluation scripts | Pending | Perplexity, real NIAH |
| Vast.ai setup script | Pending | Bootstrap + backup scripts |

### Phase 2: Training (~1 week on 8×H100)

```
Day 1-2:   125M full ladder (all stages, 4-point sweep)
Day 3-5:   760M ladder (S0 ext, S1, S2@10%, S2@20%, S3 ext, eval)
Day 5-7:   Qwen 0.5B (TTT-E2E warm-start + optional SWAA)
```

760M dependency chain:
```
Author FA ckpt ──→ S0 (ext 32K)
               └─→ S1 (SWA ext 32K)           [parallel with S0]
               └─→ SWA bridge (8K, pretrain)
                        └─→ S2_ADAPT @10% (8K, meta)
                        └─→ S2_ADAPT @20% (8K, meta)  [parallel]
                                 └─→ S2 ext 32K (per sweep point)

Author E2E ckpt ─→ S3 (ext 32K)                [parallel with everything]
```

### Phase 3: Analysis + Writing (~1-2 weeks)

- Generate figures: Pareto curves, per-position loss, causal ladder bars
- Write paper following structure in Section 13
- Target venue decision based on results strength

---

## 13. Paper Structure

1. **Abstract** — first controlled warm-start study of TTT-E2E
2. **Introduction** — training cost barrier, the open question, contributions
3. **Related Work** — TTT-E2E, AllMem, In-Place TTT, TPTT, qTTT, positioning
4. **Method** — causal ladder, adaptation bridge, fairness protocol, preregistered targets
5. **Experiments** — 125M results (full sweep), 760M results (confirms trend), Qwen validation
6. **Analysis** — per-position NLL, budget Pareto curve, S1 failure mode
7. **Discussion** — comparison with AllMem/In-Place TTT (different architectures, same direction), limitations
8. **Conclusion**
9. **Appendix** — SWAA comparison (if included), full hyperparameters, mechanistic analysis

Target venue: Workshop (ES-FoMo, WANT) or main conference depending on results.

---

## 14. Risk Register

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| Warm-starting doesn't work | HIGH | Negative result publishable. Mechanistic analysis explains why. Contrast with AllMem/In-Place TTT succeeding on different architectures. |
| Author checkpoints incompatible format | HIGH | Verify early. Conversion script as fallback. |
| AllMem/In-Place TTT scoop novelty | MEDIUM | Addressed: our contribution is TTT-E2E-specific + controlled methodology. Different architecture = different paper. |
| GCS data download fails | MEDIUM | Ask authors for alternative. Tokenize from scratch as last resort. |
| Benchmark claims outrun implementation | MEDIUM | Fixed: explicit "proxy" labeling or real harnesses. No ambiguity. |
| S0 framed as "ceiling" | LOW | Fixed: S0 is "matched FA control." |
| Budget overrun | MEDIUM | 30% buffer. Can cut: Qwen 1.5B, SWAA, S2@20% at 760M. |

---

## 15. References

### TTT Family
- **TTT-E2E:** Tandon, Dalal, Li, Koceja, Rød, Sun et al. (2025).
  [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)
- **AllMem:** Wang et al. (2026).
  [arXiv:2602.13680](https://arxiv.org/abs/2602.13680)
- **In-Place TTT:** Feng et al. (ICLR 2026).
  [OpenReview](https://openreview.net/forum?id=dTWfCLSoyl)
- **TPTT:** Furfaro (2025).
  [arXiv:2506.17671](https://arxiv.org/abs/2506.17671)
- **qTTT:** Bansal et al. (2025).
  [arXiv:2512.13898](https://arxiv.org/abs/2512.13898)
- **LaCT:** Zhang et al. (2025).
  [arXiv:2505.23884](https://arxiv.org/abs/2505.23884)
- **Titans:** Behrouz et al. (2024).
  [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

### Long-Context Adaptation
- **SWAA:** (2025).
  [arXiv:2512.10411](https://arxiv.org/abs/2512.10411)
- **TLM (Test-time LoRA):** (2025).
  [arXiv:2505.20633](https://arxiv.org/abs/2505.20633)

### Methodology
- **PaLM FLOPs formula:** Chowdhery et al., JMLR 2023.
  [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
- **Chinchilla scaling:** Hoffmann et al., NeurIPS 2022.
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **vTrain calibration:** Bang et al., MICRO 2024.
  [arXiv:2312.12391](https://arxiv.org/abs/2312.12391)

### Models and Infrastructure
- **Qwen2.5:** [huggingface.co/Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- **MosaicML LLM-Foundry benchmarks:** [GitHub](https://github.com/mosaicml/llm-foundry)
- **TTT-E2E code:** [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e)
- **Vast.ai:** [vast.ai/pricing](https://vast.ai/pricing)
