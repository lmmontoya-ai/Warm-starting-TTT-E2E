<!--
DRAFT ON HOLD (2026-09-02): the warm-start results this draft reports come from a
conversion that reinitialized every feed-forward block (see PREREGISTRATION_REVISION_V3.md).
Do not submit or convert to a template until the revision-v3 runs replace Sections 5.1-5.5
and 5.7. Target venue per advisor: TMLR.

Draft status: first OJ-CS-oriented Markdown approximation, 2026-06-13.

OJ-CS preparation notes checked from official IEEE / IEEE Computer Society
sources before drafting:
- OJ-CS publishes open-access articles on emerging topics and trends across
  computing; this draft is therefore structured as a full characterization
  article rather than a short letter.
- IEEE Computer Society author guidance requires journal submissions to use an
  IEEE article template.
- The OJ-CS call for papers points authors to the IEEE Open Journals article
  template, with Word and LaTeX versions available in the IEEE Author Center.
- IEEE Author Center guidance emphasizes checking the target journal's aims and
  scope, following the journal's submission guidelines, submitting to only one
  publication at a time, and preparing manuscript PDF/figures according to IEEE
  rules.

This Markdown file is not yet the final IEEE-formatted manuscript. It is the
content draft to be converted into the IEEE Open Journals template after venue
confirmation.
-->

# Adapting Pretrained Transformers into Long-Context Test-Time Training

Luis M. Montoya, Vijay K. Madisetti

Georgia Institute of Technology

## Abstract

Long-context language modeling often begins from a practical mismatch: strong
pretrained Transformers are usually available as short-context full-attention
models, while many long-context methods require a different training regime from
the start. Test-Time Training End-to-End (TTT-E2E) is one such regime. It uses
ordinary next-token prediction not only to train model parameters offline, but
also to train a model to update selected fast weights while processing a
sequence. This raises a simple question with practical consequences: if a
full-attention checkpoint already exists, can it be adapted into long-context
TTT-E2E, or must the TTT-E2E model be trained from scratch?

We study this question as a checkpoint-reuse problem rather than as an
unconditional efficiency claim. The central intervention is a short
seed-context adaptation bridge: after converting a pretrained full-attention
model into the TTT-E2E architecture, we train it under the TTT-E2E objective at
the seed context length before extending it to 32K tokens. A direct-conversion
control shows that this bridge is not incidental. At 125M parameters, direct
conversion without the bridge reaches Books32K loss 5.9976, while the bridged
warm-start path reaches 3.9208 across five paired seeds, a mean improvement of
2.0767 loss with 95% paired bootstrap interval [2.0500, 2.0970]. The bridged
path still does not match from-scratch TTT-E2E, which reaches 3.2722. We
therefore frame warm-starting as a reuse path that trades final quality for
avoiding a dedicated from-scratch TTT-E2E pretraining run.

Additional experiments sharpen the interpretation. A bridge-budget frontier
shows diminishing returns: a 5% bridge recovers 70.20% of the standard bridge
benefit over direct conversion, while 20% and 40% bridges continue improving
final loss with smaller marginal gains. A
continued direct-conversion run improves with extra long-context training but
remains far behind bridged warm-starting. A 500-example paired retrieval proxy
does not support a retrieval advantage for warm-starting over scratch. Finally,
existing 760M checkpoint evaluations provide a second-scale quality check: the
warm-start tax narrows from 0.6447 at 125M to 0.3188 at 760M, but this is only a
two-scale observation, not a scaling law. Taken together, the results support a
careful characterization: the bridge is empirically necessary under the matched
extension budget, bridge budget is a tunable cost-quality knob, and
from-scratch TTT-E2E remains the final-quality reference.

## 1. Introduction

Long-context language models are useful only when they preserve two things at
once. They must use information from distant tokens, and they must retain the
next-token prediction quality that made the short-context model useful in the
first place. Full attention offers a direct route to using context, but its
cost grows quadratically with sequence length. This has motivated architectures
that compress, approximate, or replace attention for long sequences.

Test-Time Training End-to-End (TTT-E2E) approaches long context from a different
angle. Instead of storing all relevant context in an attention cache, the model
learns to update selected fast weights while it processes a sequence. The inner
loop performs self-supervised gradient steps on the observed context; the outer
loop trains the model for its post-update behavior. In this sense, the model is
not only reading context. It is trained to learn from context at inference time.

This design is appealing, but it creates a practical barrier. Many organizations
already have strong pretrained full-attention Transformers. They may not have a
dedicated from-scratch TTT-E2E pretraining run. If entering the TTT-E2E regime
requires starting over, the method is harder to adopt. If an existing
full-attention checkpoint can be adapted, then TTT-E2E becomes a reuse problem:
how much quality is lost, how much adaptation is needed, and what exactly fails
when adaptation is skipped?

The first version of this study framed warm-starting primarily as an efficiency
result. That framing was too strong. From-scratch TTT-E2E was better on final
language-modeling loss, and the apparent cost advantage depended on whether the
full-attention seed was counted as a sunk asset. More importantly, the original
comparison did not isolate the bridge stage. It showed that a bridged TTT-E2E
model was much better than simple long-context baselines, but it did not answer
whether the bridge itself was necessary.

We revise the question. We ask whether a pretrained full-attention checkpoint
can be adapted into long-context TTT-E2E, what the bridge contributes, and how
the resulting model should be understood relative to from-scratch TTT-E2E. The
answer is not that warm-starting wins on final quality. It does not. The answer
is that the bridge is a real adaptation stage: without it, direct conversion
trains but remains far worse under the matched extension budget.

We make four empirical contributions.

First, we isolate the bridge. We compare bridged warm-starting against direct
conversion into the TTT-E2E architecture followed by the same 32K extension. At
125M parameters, the direct-conversion model reaches loss 5.9976 while the
bridged model reaches 3.9208 across five paired seeds. The mean bridge effect is
2.0767 loss, far above the pre-specified 0.10 margin.

Second, we characterize the bridge as a practical budget knob. A 5% bridge
recovers 70.20% of the standard bridge benefit over direct conversion, while
20% and 40% bridge budgets continue improving final loss. Because larger bridge
budgets also use more short-context tokens, this is a cost-quality frontier, not
a pure structural ablation.

Third, we keep the scratch comparison honest. From-scratch TTT-E2E remains
better on language-modeling loss: 3.2722 at 125M versus 3.9208 for the paired
bridged warm-start mean. A 40% bridge narrows this gap to 0.1740 but does not
close it. Warm-starting is therefore best understood as checkpoint reuse rather
than final-quality dominance.

Fourth, we replace an underpowered retrieval anecdote with a paired proxy
evaluation. The 500-example proxy does not support a warm-start retrieval
advantage over scratch at either 125M or 760M. This removes a tempting but
unsupported complementarity headline and leaves the paper as a characterization
study.

The rest of the paper follows this logic. Section 2 situates the work in
long-context modeling, TTT-E2E, and pretrained-checkpoint reuse. Section 3
defines the seed-context adaptation bridge. Section 4 describes the comparison
conditions, evaluation surface, and checkpoint-based measurement protocol.
Section 5 presents the bridge-isolation result, bridge-budget frontier,
continuation analysis, retrieval proxy, and 760M quality check. Sections 6 and 7
discuss when warm-starting is useful and where the evidence remains limited.
Section 8 describes reproducibility.

## 2. Background and Related Work

### 2.1. Long-Context Modeling Beyond Full Attention

Full causal attention is simple and effective, but its memory and computation
costs scale poorly with context length. Long-context research therefore includes
sliding-window attention, sparse attention, recurrent state, external memory,
learned compression, and hybrid approaches. These methods differ in their
mechanism, but share a common tension: improving context length can degrade the
local modeling quality or optimization behavior learned during short-context
pretraining.

This tension matters for adaptation. If a long-context method requires a new
architecture or a new training objective, then a strong full-attention
checkpoint may no longer be immediately usable. The practical question is not
only which long-context method works when trained from scratch, but whether
existing pretrained checkpoints can be moved into the new regime with a
controlled quality tax.

### 2.2. End-to-End Test-Time Training

Test-time training updates a model or part of a model at inference time using a
self-supervised objective. TTT-E2E trains this update rule end to end under the
ordinary language-modeling objective. In the recipe we study, the model is a
Transformer-like decoder with sliding-window attention for local context and
additional fast-weight pathways in a suffix of blocks. During sequence
processing, the inner loop updates selected fast weights on chunks of the
observed context. The outer loop differentiates through those updates and trains
the model to make good predictions after adaptation.

This creates a different kind of long-context state. Instead of storing all
past information in attention activations, the model can store some information
in temporary weight changes. But this also makes initialization important. A
model trained from scratch under TTT-E2E sees the inner-loop update rule
throughout training. A converted full-attention checkpoint does not.

### 2.3. Checkpoint Reuse and Architectural Compatibility

Recent work on in-place test-time training emphasizes a related motivation:
TTT-style methods are easier to use if they are compatible with pretrained
language models. Some approaches pursue compatibility by modifying the update
mechanism so it can reuse existing Transformer components directly. A separate
line of work studies how to adapt full-attention models to sliding-window
attention. Sliding Window Attention Adaptation (SWAA), for example, asks how to
move a full-attention pretrained model into a sliding-window attention regime
without costly pretraining, using recipes such as selective full attention,
sink-token preservation, interleaved attention patterns, prompting strategies,
and fine-tuning.

Our study is adjacent but different. SWAA studies the full-attention to
sliding-window attention transition without introducing the TTT-E2E fast-weight
training regime. In-place TTT studies compatibility by changing how test-time
updates are expressed. We instead keep the TTT-E2E-style conversion path and
isolate one missing stage inside that path: whether converted checkpoints need a
seed-context bridge before long-context extension.

The distinction is important. We do not claim to be the first to value
pretrained-checkpoint compatibility. Instead, we provide a controlled empirical
study of one adaptation route: convert the checkpoint, bridge it at the seed
context length, and then extend to long context.

### 2.4. Evaluation Beyond Language-Modeling Loss

Long-context language-modeling loss is a useful measurement surface, but it is
not the same as long-context capability. Retrieval and reasoning benchmarks can
expose failures that average next-token loss hides. For this reason, we include
a retrieval-style proxy. We treat it narrowly: it is a paired comparison of
conditions, not a claim of strong absolute retrieval capability. The main
quality surface remains Books32K validation loss because it is the shared
extension and evaluation surface for all adaptation paths.

## 3. Seed-Context Adaptation Bridge

We study a conversion problem. The starting point is a pretrained short-context
full-attention decoder-only Transformer. The target is a long-context TTT-E2E
model with sliding-window attention and fast-weight pathways in a suffix of
blocks. The conversion has three kinds of parameters.

Some parameters are inherited exactly: token embeddings, output head, layer
normalization parameters, attention projection weights, and feed-forward weights
whose shapes remain compatible. These carry the pretrained language-modeling
knowledge into the converted model.

Some parameters are new. The fast-weight pathways and their associated
normalization modules do not exist in the full-attention seed. They are freshly
initialized.

Some parameters are reinitialized because their shapes change in the converted
configuration. In our 125M model, the feed-forward intermediate dimension
changes from 2048 to 1664 in the converted bridge configuration. In the 760M
model, it changes from 4096 to 3328. Shape-aware restoration leaves these
incompatible tensors at their default initialization rather than forcing an
incorrect load.

The attention operator also changes. The full-attention seed uses full causal
attention. The converted model uses sliding-window attention. The projection
weights are compatible and can be inherited, but the sequence operator is
different.

This conversion alone is not the proposed adaptation. The central intervention
is the seed-context adaptation bridge. After conversion, we train the model
under the full TTT-E2E objective at the seed context length, 8K tokens, before
the 32K extension. The bridge uses the same inner/outer-loop training regime as
TTT-E2E. The inner loop updates only the fast-weight suffix pathways. The outer
loop updates all trainable parameters through the inner-loop computation.

The bridge is intentionally short. Its purpose is not to solve long-context
modeling at 8K. Its purpose is to put the inherited full-attention checkpoint
into a regime where the TTT-E2E update rule is meaningful before the model is
asked to train at 32K.

The critical control is direct conversion without this bridge. If direct
conversion followed by 32K extension works, then the bridge is unnecessary. If
it does not, then the bridge is a real adaptation stage rather than an
implementation detail.

## 4. Experimental Design

### 4.1. Model Scales

The main controlled study is at 125M parameters. The model has 12 layers,
hidden size 768, and 12 attention heads. The TTT-E2E fast-weight suffix covers
the final quarter of the network. Both the full-attention seed and the
from-scratch TTT-E2E seed are locally trained at 8K context for 4,800 steps with
global batch size 64.

We also report a 760M quality check from existing checkpoints. This model has
24 layers, hidden size 1536, and 16 attention heads. The 760M results are useful
for checking whether the warm-start tax narrows at a larger scale, but they are
not a full repeated protocol. In particular, we do not have a 760M
direct-conversion control, and we do not make a homogeneous 760M cost claim.

### 4.2. Data and Evaluation Surfaces

Short-context pretraining and bridge adaptation use DCLM-8K at 8K context.
Long-context extension uses Books3 at 32K context. The main comparison surface
is Books32K validation loss. All reported main losses come from restoring the
saved checkpoint and re-running the same float32 evaluation pipeline over 64
validation batches with evaluation batch size 8.

This checkpoint-based protocol matters because small differences can be
obscured by lower-precision or transient logging. The final manuscript numbers
therefore come from saved checkpoints, not from training-time log summaries.

### 4.3. Compared Adaptation Paths

We compare five paths. They are named descriptively here because their
scientific role matters more than any implementation label.

The full-attention long-context baseline starts from the pretrained
full-attention seed and extends it directly to 32K.

The sliding-window-only baseline starts from the same seed, changes the
attention operator to sliding-window attention, and extends to 32K without
TTT-E2E.

The bridged warm-start path starts from the full-attention seed, converts it
into the TTT-E2E architecture, runs the seed-context adaptation bridge at 8K,
and then extends to 32K.

The direct-conversion path starts from the full-attention seed, converts it into
the same TTT-E2E architecture, skips the bridge, and extends directly to 32K.
This is the bridge-isolation control.

The from-scratch TTT-E2E path trains the TTT-E2E model family from scratch at
8K, then extends the resulting seed to 32K.

All 32K extension comparisons are evaluated on the same Books32K surface. For
the bridge-isolation comparison, the bridged warm-start and direct-conversion
paths are run as five paired seeds and compared with paired statistics.

### 4.4. Retrieval-Style Proxy

We include a deterministic 32K needle-style retrieval proxy to replace a much
smaller preliminary probe. The same 500 examples are used across compared
conditions. Correctness is binary: the model is counted correct if its
highest-scoring candidate matches the needle token exactly. This
single-token exact-match criterion is stringent, and absolute accuracy is low
at these model scales. We therefore use the proxy only as a paired comparison
between adaptation paths, not as a claim that any model has strong retrieval
capability.

Individual accuracies are reported with Wilson intervals. Paired comparisons
use exact one-sided McNemar/binomial tests in the pre-specified direction for a
warm-start advantage. We also report the number of discordant pairs, because in
a paired binary test those pairs determine effective comparison power.

### 4.5. Pre-Specified Analysis Plan

Before interpreting the new runs, we fixed the analysis rules in a dated
artifact. For the bridge-isolation comparison, the statistic is the
direct-conversion loss minus the bridged warm-start loss, computed across five
paired seeds. We set 0.10 Books32K loss as the practical margin: the bridge is
classified as helpful if the mean effect is at least 0.10 and the paired
seed-level bootstrap interval remains positive; harmful if the sign is reversed
by the same margin; negligible if the effect lies within the margin; and
inconclusive if the point estimate and interval disagree.

The same plan fixed the retrieval comparison before results were inspected. The
primary retrieval test is bridged warm-start versus from-scratch TTT-E2E at
125M and 32K context, in the direction of a warm-start advantage. Larger-scale
and direct-conversion retrieval comparisons are secondary. The plan also fixed
Wilson intervals for individual accuracies, exact paired tests over discordant
pairs, the bridge-budget levels, the frontier-shape labels, and the
continuation rule. The purpose of stating this in the paper is not ceremony. It
is to separate the conclusions from choices made after seeing the numbers.

## 5. Results

### 5.1. Main 125M Quality Comparison

The main 125M comparison shows why the warm-start path is worth studying, but
also why it should not be oversold.

| Adaptation path | Books32K loss |
| --- | ---: |
| Full-attention extension | 6.5879 |
| Sliding-window-only conversion | 6.5423 |
| Bridged warm-start, single reference checkpoint | 3.9169 |
| From-scratch TTT-E2E | 3.2722 |

The bridged warm-start path is far better than the full-attention and
sliding-window-only baselines. It is also worse than from-scratch TTT-E2E by
0.6447 loss on this surface. This is the core trade-off. Warm-starting is a
useful adaptation path from an existing checkpoint, but from-scratch TTT-E2E
remains the final-quality reference.

**Figure 1 placeholder.** Main 125M Books32K quality comparison: full-attention
extension, sliding-window-only conversion, bridged warm-start, and from-scratch
TTT-E2E. The caption should state that lower loss is better and that all values
come from the same 64-batch checkpoint evaluation surface.

For bridge-isolation and bridge-budget arithmetic, we use the five-seed paired
warm-start mean, 3.9208, rather than the single reference checkpoint value,
3.9169. The two numbers agree within 0.004 loss. We keep this distinction
explicit to avoid mixing a single historical checkpoint with the paired
bridge-isolation design.

### 5.2. Direct Conversion Is Not Enough

The central question is whether the seed-context bridge is actually needed. It
is. When we skip the bridge and directly extend the converted TTT-E2E model to
32K, the model trains but remains much worse.

Across five paired seeds:

| Path | Mean Books32K loss |
| --- | ---: |
| Direct conversion without bridge | 5.9976 |
| Bridged warm-start | 3.9208 |

The mean bridge effect is 2.0767 loss with 95% paired bootstrap interval
[2.0500, 2.0970]. This effect is not close to the pre-specified practical
margin of 0.10. It satisfies the pre-specified rule for a helpful bridge by a
large margin.

This result changes the paper. The bridge is no longer only a plausible part of
the recipe. It is empirically isolated. Direct conversion into the TTT-E2E
architecture, even with the same 32K extension surface, does not recover the
bridged model's quality.

The result should still be interpreted behaviorally. We have not measured
fast-weight gradient norms, inner-loop update magnitudes, or pathway-specific
weight movement. We therefore do not claim that the bridge activates the
fast-weight pathway in a mechanistic sense. The supported claim is narrower and
stronger: under this matched protocol, the bridge is required for the converted
checkpoint to enter a much better long-context TTT-E2E regime.

### 5.3. The Gap Is Not Only a Late-Context Failure

A natural concern is that direct conversion might fail only at the end of the
32K context window. If so, the bridge would be mainly a tail-position repair.
Per-position negative log-likelihood does not support that interpretation. The
direct-conversion gap is broadly visible across the 32K window. The bridge
therefore improves the model's behavior throughout the long-context surface,
not only at the farthest positions.

**Figure 2 placeholder.** Per-position negative log-likelihood comparison
between direct conversion and bridged warm-start, summarized across the 32K
window. The intended message is not mechanism, but scope: the direct-conversion
gap is not only a late-context artifact.

This does not reveal the internal mechanism. It does constrain the explanation.
The bridge is not merely fixing a narrow late-context artifact.

### 5.4. How Much Bridge Is Needed?

Once the bridge is isolated, the next practical question is how much bridge
training is needed. We vary the bridge token budget while keeping the final 32K
extension surface fixed.

| Bridge budget | Books32K loss | Improvement vs. direct conversion, in standard-bridge units |
| --- | ---: | ---: |
| 0%, direct conversion | 5.9976 | 0.0000 |
| 5% | 4.5396 | 0.7020 |
| 10%, standard bridge | 3.9208 | 1.0000 |
| 20% | 3.6460 | 1.1324 |
| 40% | 3.4462 | 1.2286 |
| From-scratch TTT-E2E reference | 3.2722 | -- |

The third column expresses improvement over direct conversion in units of the
standard 10% bridge improvement, 5.9976 minus 3.9208. Values above 1.0 do not
mean that more than 100% of the gap to scratch has been closed. They mean that
the larger bridge improves beyond the standard 10% bridge endpoint.

The main shape is diminishing returns. The loss improvement from 0% to 5% is
1.4580. The next increases are smaller: 0.6188 from 5% to 10%, 0.2749 from 10%
to 20%, and 0.1998 from 20% to 40%. The pre-specified frontier rule classifies
the 5% bridge as threshold-like because it recovers 70.20% of the standard
bridge improvement, but the boundary crossing itself should not carry the
result. The full curve is more informative. Bridge budget is a cost-quality
knob: a small bridge recovers much of the adaptation benefit, while larger
bridges continue narrowing the gap to scratch.

The 40% bridge reaches loss 3.4462, 0.1740 worse than from-scratch
TTT-E2E. It still does not match scratch. The result supports reuse, not final
quality dominance.

**Figure 3 placeholder.** Bridge-budget frontier: 0%, 5%, 10%, 20%, and 40%
bridge budgets, with the from-scratch TTT-E2E reference as a horizontal line.
This should be the visual center of the paper.

This frontier also has a built-in caveat. Increasing bridge budget also
increases short-context training tokens. The frontier therefore characterizes a
practical budget trade-off. It is not a pure structural ablation that separates
the bridge mechanism from additional upstream training.

### 5.5. Direct Conversion Improves With More 32K Training, But Remains Far Behind

The direct-conversion model might simply need more long-context training. To
test this, we continued the final direct-conversion checkpoint for an additional
1,440 extension steps with optimizer state restored. This mirrors the
continuation protocol used for the original long-context continuation analyses
and is generous to the direct-conversion path: it gives the no-bridge model
substantial extra 32K training after its matched endpoint.

The result is mixed in an instructive way.

| Path | Books32K loss |
| --- | ---: |
| Direct conversion before continuation | 5.9976 |
| Direct conversion after +1,440 steps | 5.4874 |

The improvement is 0.5102 loss, larger than the conservative threshold we set
for calling the original plateau persistent. We therefore do not claim that
direct conversion is permanently stuck at its first endpoint.

At the same time, continued direct conversion remains 1.5666 worse than the
paired bridged warm-start mean and more than 2.0 worse than the 40% bridge. More
long-context training helps, but it does not erase the bridge advantage even
under this generous continuation.

**Figure 4 placeholder.** Direct-conversion continuation curve, compared with
the bridged warm-start reference and the 40% bridge endpoint. The caption should
state that continuation weakens a hard plateau claim but leaves a large bridge
advantage.

### 5.6. Retrieval Proxy Does Not Support a Warm-Start Advantage

The original small retrieval probe suggested that warm-starting might have an
advantage over scratch on needle-style recall. That result was based on too few
examples to support a claim. We replaced it with a 500-example paired proxy.

| Scale | Comparison | Accuracy, warm-start | Accuracy, scratch | Discordant pairs | Exact one-sided p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| 125M | bridged warm-start vs scratch | 0.060 | 0.084 | 68 | 0.943 |
| 760M | bridged warm-start vs scratch | 0.078 | 0.078 | 62 | 0.550 |

The proxy does not support a warm-start retrieval advantage. At 125M, scratch is
directionally higher. At 760M, the result is tied.

The low absolute accuracy is not used to make an absolute capability claim. It
follows partly from the design of the proxy: a prediction counts only when the
highest-scoring candidate exactly matches the needle token. That stringent
binarization creates a low floor at these model scales. The relevant point is
therefore paired and comparative: with the same 500 examples across conditions
and adequate discordant counts, we do not detect a meaningful
warm-start-over-scratch retrieval advantage on this proxy.

**Retrieval table placeholder.** The final paper should keep this as a table
with accuracy, Wilson intervals, discordant-pair counts, and exact one-sided
p-values rather than turning it into a figure.

### 5.7. Larger-Scale Quality Check

The 760M results provide a second-scale quality check. They are inherited
checkpoint results evaluated on the same kind of checkpoint-based surface, not a
new full protocol.

| Scale | Bridged warm-start loss | From-scratch TTT-E2E loss | Warm-start tax |
| --- | ---: | ---: | ---: |
| 125M | 3.9169 | 3.2722 | 0.6447 |
| 760M | 2.9940 | 2.6752 | 0.3188 |

This table uses the single 125M bridged warm-start checkpoint so that the 125M
and 760M rows are both single-checkpoint quality comparisons. The paired 125M
mean, 3.9208, is used for the bridge-isolation and bridge-budget analyses.

The warm-start tax narrows across the two tested scales. This is suggestive and
useful, but it is not a scaling law. We do not have a third scale, and we do not
have a 760M direct-conversion control. The bridge-isolation result is therefore
established at 125M; the 760M result is a quality-only second-scale check.

## 6. Discussion

### 6.1. What the Bridge Result Means

The bridge result is strong because it removes a simpler explanation. Before
the direct-conversion control, one could argue that the bridged warm-start model
was better than naive baselines simply because it entered the TTT-E2E
architecture. That is not enough. Direct conversion enters the same architecture
but performs much worse. The missing ingredient is the seed-context adaptation
stage.

The bridge should not be described as a solved mechanism. We do not yet know
which internal quantities change in the fast-weight pathways, whether
inner-loop gradients become better conditioned, or how much of the improvement
comes from new modules versus inherited parameters. The evidence supports a
behavioral statement: the bridge is necessary under the matched extension
budget. Mechanistic diagnosis is future work.

### 6.2. When Warm-Starting Is Rational

Warm-starting is rational when the full-attention seed already exists and the
quality tax is acceptable. Under that condition, the bridge plus 32K extension
is much cheaper than training a TTT-E2E seed from scratch. If the full-attention
seed must also be trained solely for this purpose, the cost advantage is much
smaller.

This is why the paper avoids the word "efficient" as its main claim.
Warm-starting is not uniformly more efficient in a context-free sense. It is a
reuse path. Reuse is valuable in exactly the cases where strong full-attention
checkpoints are already part of the development pipeline.

### 6.3. Why the Retrieval Null Matters

The retrieval proxy could have changed the paper's story. If warm-starting had
beaten scratch on retrieval, then the two training paths might have looked
complementary: scratch for language-modeling loss, warm-start for recall. The
powered proxy does not support that. This is useful because it prevents the
paper from inheriting an underpowered anecdote.

The revised story is therefore simpler. The bridge changes the entry path into
TTT-E2E and strongly affects language-modeling quality under the matched
extension budget. It does not, on current evidence, create a distinct retrieval
advantage. Both warm-starting and scratch share the low retrieval floor on this
proxy, which is consistent with the broader concern that compression-style
long-context methods can trade context length for recall.

## 7. Limitations

The bridge-isolation experiment is at 125M parameters. The 760M result is a
quality-only second-scale check, not a repeated bridge-isolation protocol. The
paper should therefore say that the bridge is isolated at 125M and that the
quality gap narrows across two tested scales. It should not claim that the
bridge effect is validated at 760M.

The bridge-budget frontier uses one seed for the 5%, 20%, and 40% bridge
points. The 0% and 10% anchors are five-seed means. This is sufficient for a
frontier characterization, but not for a variance claim at every budget.

The retrieval proxy is in-house and uses a stringent single-token exact-match
criterion. It is useful as a paired comparison and it replaces an underpowered
preliminary probe, but it is not a full long-context benchmark suite.

The work does not instrument the internal fast-weight pathway. We do not yet
measure inner-loop gradient norms, pathway-specific update magnitudes, or
parameter movement. Such instrumentation would be needed to turn the behavioral
bridge result into a mechanistic account.

The study uses one model family for the controlled 125M ladder. Applying the
same adaptation path to external pretrained models such as Qwen or Gemma would
strengthen the generality story, but is not required for the present
checkpoint-reuse characterization.

## 8. Reproducibility and Artifacts

All main reported losses come from saved checkpoint restoration followed by
float32 evaluation on the standard Books32K validation surface. The
bridge-isolation comparison uses five paired seeds. The direct-conversion
control and bridged warm-start path share the same 32K extension surface, data
fingerprints, and evaluation protocol.

The artifact package includes restored checkpoints, evaluation manifests,
paired-loss tables, retrieval-proxy manifests, bridge-budget summaries, and
download instructions. The artifact bundle records which result surfaces are
authoritative and which are superseded. This is part of the method: the paper's
claims depend on checkpoint-level reproducibility rather than training-log
snippets.

The final reproducibility statement should remain compact and include:

- checkpoint artifact links,
- exact restoration commands,
- dataset fingerprint policy,
- evaluation batch count and precision,
- paired seed ids for the bridge-isolation comparison,
- and a statement that 760M costs are not used for main-text cost claims.

## 9. Conclusion

We studied whether a pretrained short-context full-attention Transformer can be
adapted into long-context TTT-E2E. The answer is yes, but with an important
qualification. The seed-context bridge is essential under the matched extension
budget: direct conversion trains but remains far worse. At the same time,
from-scratch TTT-E2E remains better on final language-modeling loss.

This makes warm-starting a checkpoint-reuse method rather than a replacement
for scratch TTT-E2E. Its value is practical: when a full-attention checkpoint
already exists, a bridge can move it into the TTT-E2E regime and produce a much
stronger long-context model than direct conversion. Bridge budget then becomes a
cost-quality knob. A small bridge recovers much of the bridge benefit; larger
bridges continue narrowing the gap to scratch.

The revised claim is therefore narrower than the first version of this work,
and stronger because of that. Warm-starting does not win everywhere. It gives a
measured, reproducible path from existing full-attention checkpoints into
long-context TTT-E2E, with a clear quality tax and a now-isolated bridge
contribution.

## Acknowledgments

Acknowledgments will be finalized after the submission venue, compute-credit
language, and author-order details are confirmed.

## References

Working reference list. This will be converted to IEEE numbered bibliography
format when the paper moves into the IEEE Open Journals template.

- Tandon et al. *End-to-End Test-Time Training for Long Context*. arXiv,
  2025.
- Sun et al. *In-Place Test-Time Training for Large Language Models*. arXiv,
  2026.
- Zhang et al. *Test-Time Training Done Right / Large Chunk Test-Time
  Training*. arXiv, 2025.
- Behrouz et al. *Titans: Learning to Memorize at Test Time*. NeurIPS, 2025.
- Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context
  Language Models?* 2024.
- Yen et al. *HELMET: How to Evaluate Long-Context Language Models
  Effectively and Thoroughly*. 2024.
- Bai et al. *LongBench v2: Towards Deeper Understanding and Reasoning on
  Realistic Long-Context Multitasks*. 2024.
- Yu, Liu, Wu, Wang, and Pei. *Sliding Window Attention Adaptation*. arXiv,
  2025.
