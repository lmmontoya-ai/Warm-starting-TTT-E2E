# External Pretrained Adaptation Plan

Primary target: `Qwen/Qwen2.5-1.5B`
Secondary target: `google/gemma-3-1b-pt`

## 1) Objective
Test whether initializing TTT-E2E from an external pretrained dense model can reduce total training wall-clock while preserving long-context quality, relative to paper-style TTT-E2E training.

## 2) Verified Model Constraints

### Qwen2.5-1.5B (primary)
1. Public base model (Apache-2.0), 1.54B params, 28 layers, GQA (12 Q heads, 2 KV heads), base context listed as 32,768 in model card.
2. Config exposes `use_sliding_window=false`, so this is a clean "dense/full-attention first" source model for adaptation experiments.
3. Config also includes `max_position_embeddings=131072`; we will treat this as implementation capacity and still keep explicit 8K->32K protocol for comparability with TTT-E2E training stages.

### Gemma-3-1b-pt (secondary)
1. The official HF checkpoint is gated by Gemma terms (license acceptance required before file access).
2. Official docs and model card indicate 1B/270M sizes are 32K context (larger sizes are 128K).
3. Gemma 3 architecture uses hybrid attention (alternating local sliding-window and global attention), which increases adaptation complexity compared to Qwen dense attention.

## 3) What "Comparable to TTT-E2E" Means
All external-pretrained experiments must be reported against TTT-E2E-style controls using the same pipeline concepts from the paper:

1. Stage split:
- Stage A: short-context training/adaptation (8K).
- Stage C: long-context extension (32K).

2. Datasets:
- Stage A: DCLM-8K equivalent corpus.
- Stage C: Books/Books3 extension corpus.

3. Fairness protocol:
- Token-matched comparison: equal total training tokens.
- Wall-clock-matched comparison: equal total GPU-hours.

4. Metrics:
- Primary: Books validation loss at 32K.
- Secondary: token-position loss breakdown and throughput.
- Efficiency: GPU-hours to target loss; time per 1K tokens.

5. Baseline set for each model family:
- FA-scratch baseline (same family architecture, random init).
- FA-import baseline (external checkpoint, no TTT conversion).
- TTT-E2E warm-start variants (proposed).

## 4) Implementation Workstreams

### W1. Checkpoint import and compatibility
1. Build importer from HF checkpoint format into local model weights.
2. Add strict shape/key audit report (missing keys, extra keys, resized tensors).
3. Add one forward-pass parity check on synthetic input (logits similarity before any training).

Exit criteria:
- Import reproducible with one command.
- Audit report is clean or has explicit, justified exceptions.

### W2. Tokenizer/data path alignment
1. Keep each model's native tokenizer and vocab (do not force cross-family tokenizer unification).
2. Re-tokenize Stage A/C datasets per model family.
3. Track both token count and raw-byte count for fairness diagnostics across different tokenizers.

Exit criteria:
- Deterministic tokenized datasets per model.
- Comparable token/byte accounting in reports.

### W3. Architecture adaptation hooks
1. Add model-family adapters for attention mode switching and TTT block insertion policy.
2. Add config fields for transition stages (dense->SWA->TTT where relevant).
3. Preserve a fallback path to run pure dense FA continuation as control.

Exit criteria:
- Same training harness can run FA controls and TTT warm-start variants.

## 5) Per-Model Methodology

## Qwen2.5-1.5B (primary, lower adaptation risk)

### Q-0: Controls
1. Q-FA-Scratch:
- Random init Qwen-shaped model.
- Stage A 8K FA train.
- Stage C 32K FA extension.

2. Q-FA-Import:
- Load pretrained Qwen checkpoint.
- Stage A 8K FA continue-train (domain/context alignment budget).
- Stage C 32K FA extension.

### Q-1: Proposed warm-start path
1. Q-P1-Bridge (main hypothesis):
- Start from pretrained Qwen.
- Stage B0 (optional, short): FA 8K stabilization on project corpus.
- Stage B1: convert dense attention blocks to SWA-style blocks (weights reused where possible), train in `pretrain` mode briefly.
- Stage B2: enable TTT meta updates (prime/suffix strategy aligned with TTT-E2E), adapt at 8K.
- Stage C: 32K TTT-E2E extension from Stage B2 checkpoint.

2. Q-P2-Direct:
- Start from pretrained Qwen.
- Skip B2 bridge.
- Go directly to Stage C 32K TTT extension.

3. Q ablations:
- Adaptation budget sweep for B2: 5%, 10%, 20% of baseline Stage A steps.
- With/without B1 SWA transition.
- `load_part=params` versus stricter resume choices only when shape-compatible.

## Gemma-3-1b-pt (secondary, higher adaptation risk)

### G-0: Controls
1. G-FA-Scratch:
- Random init Gemma-shaped model.
- Stage A/Stage C control training without TTT conversion.

2. G-FA-Import:
- Load pretrained Gemma 1B PT.
- Continue with native attention pattern as control.

### G-1: Proposed warm-start path
1. G-P1-Hybrid-Preserve (recommended first for Gemma):
- Keep Gemma hybrid attention structure semantics.
- Convert/adapt global-attention blocks to TTT-capable blocks first.
- Keep local sliding-window blocks in local-attention/SWA mode initially.
- Stage B2 8K meta adaptation.
- Stage C 32K extension.

2. G-P2-Direct:
- Pretrained Gemma -> direct 32K TTT-capable extension, no 8K bridge.

3. G ablations:
- Global-only TTT conversion vs broader conversion.
- Bridge budget sweep (5/10/20%).

## 6) Experiment Matrix (First Pass)

Run first on pilot budgets, then scale:

1. Q-FA-Scratch
2. Q-FA-Import
3. Q-P1-Bridge
4. Q-P2-Direct
5. G-FA-Scratch
6. G-FA-Import
7. G-P1-Hybrid-Preserve
8. G-P2-Direct

Then add adaptation-budget sweeps for Q-P1 and G-P1.

## 7) Step-by-Step Execution Plan

### Phase A: Pilot readiness (short runs)
1. Implement importer + parity checks for Qwen (first).
2. Implement Qwen adapters for B1/B2/C stages.
3. Run short-budget Qwen matrix (1-5% schedule).
4. Implement Gemma importer/adapters (post license access).
5. Run short-budget Gemma matrix.

### Phase B: Controlled comparisons
1. Run token-matched Qwen matrix.
2. Run wall-clock-matched Qwen matrix.
3. Run token-matched Gemma matrix.
4. Run wall-clock-matched Gemma matrix.

### Phase C: Analysis and decision
1. Plot quality-vs-compute frontiers per family.
2. Compare warm-start methods to each family's FA controls and to in-repo TTT-E2E baseline trendlines.
3. Decide whether external warm-start is worth scaling to longer runs.

## 8) Reporting Template
For each run, report:
1. Model family and checkpoint source.
2. Attention/TTT conversion policy.
3. Stage budgets (steps, tokens, GPU-hours).
4. Final 32K loss and token-position loss slices.
5. Throughput and total wall-clock.
6. Delta vs family FA control and vs TTT-E2E baseline.

## 9) Success Gates
1. Primary gate:
- >=30% wall-clock reduction at <=0.01 loss delta versus matched TTT-E2E baseline.

2. Secondary gate:
- Stable convergence and no severe long-token degradation in loss breakdown.

## 10) Immediate Next Actions
1. Build and validate Qwen importer + shape audit (first concrete engineering task).
2. Define Qwen B1 conversion rules (dense->SWA) and B2 TTT activation config.
3. Add external-model matrix launcher (parallel to existing phase-1 pilot script).
4. Start Qwen pilot runs before Gemma to de-risk methodology.

## Sources (model facts and constraints)
1. Qwen2.5-1.5B model card: https://huggingface.co/Qwen/Qwen2.5-1.5B
2. Qwen2.5-1.5B config: https://huggingface.co/Qwen/Qwen2.5-1.5B/blame/main/config.json
3. Gemma-3-1b-pt model page (gated access notice): https://huggingface.co/google/gemma-3-1b-pt
4. Gemma 3 official model card (size-specific context details): https://ai.google.dev/gemma/docs/core/model_card_3
5. Gemma 3 architecture notes (local/global hybrid attention): https://huggingface.co/docs/transformers/model_doc/gemma3
6. Gemma 3 technical report: https://arxiv.org/abs/2503.19786
