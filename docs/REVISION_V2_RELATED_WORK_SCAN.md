# Revision V2 Related-Work Currency Scan

Status: completed initial scan, 2026-06-12.

Purpose: satisfy the Phase 5 related-work currency item before manuscript
drafting. This scan is not a full related-work section; it is a dated checklist
of papers and status facts that the final manuscript should account for.

## Search Scope

Queries covered:

- `End-to-End Test-Time Training for Long Context`
- `In-Place Test-Time Training`
- `Large Chunk Test-Time Training`
- `warm start pretrained checkpoint test-time training`
- `architecture compatible LLM warm start TTT`
- recent long-context TTT / memory papers since the TTT-E2E preprint

## TTT-E2E Publication Status

As of this scan, Tandon et al., *End-to-End Test-Time Training for Long
Context*, should be cited as an arXiv preprint unless advisor/publication
updates say otherwise.

Evidence:

- arXiv page: https://arxiv.org/abs/2512.23675
- arXiv records submission on 2025-12-29 and latest revision v2 on 2025-12-31.
- The arXiv page lists code in the comments and does not show a journal
  reference in the browsed page.
- OpenReview revision page found: https://openreview.net/revisions?id=b1IBlEzRDq
  It names the paper and authors but the public revision page alone does not
  establish an accepted venue.

Manuscript action:

- Cite TTT-E2E as the reference recipe/source method.
- Do not imply a conference/journal acceptance unless confirmed separately.

## New / Relevant Work To Account For

### In-Place Test-Time Training

Source: https://arxiv.org/html/2604.06169v1

Why it matters:

- This is the closest current work to our revised framing.
- It explicitly argues that TTT for LLMs needs architectural compatibility and
  defines compatibility as the ability to warm start from a pretrained
  checkpoint.
- It proposes a drop-in approach that repurposes existing MLP blocks rather
  than introducing specialized replacement layers.
- It evaluates both pre-trained LLM enhancement and scratch-style comparisons.

Manuscript positioning:

- Our paper is not claiming the general idea that pretrained-checkpoint reuse
  matters; In-Place TTT makes that point directly.
- Our distinct contribution is an empirical, checkpoint-based characterization
  of adapting an existing full-attention Transformer into the published
  TTT-E2E regime via a bridge stage, with a preregistered no-bridge control.
- Related-work paragraph should say: In-Place TTT pursues architectural
  compatibility by changing the fast-weight design in-place; our study keeps
  the TTT-E2E-style conversion/bridge pathway and asks whether a pretrained FA
  seed can enter that regime without scratch TTT-E2E pretraining.

### Test-Time Training Done Right / LaCT

Sources:

- arXiv: https://arxiv.org/abs/2505.23884
- Project page: https://tianyuanzhang.com/projects/ttt-done-right/

Why it matters:

- LaCT argues for large-chunk TTT updates to improve hardware utilization and
  scalability.
- It broadens the TTT long-context landscape beyond fine-grained online update
  designs.

Manuscript positioning:

- Mention as contemporary TTT systems work focused on update granularity,
  accelerator utilization, and scaling.
- It is not the same question as warm-starting a full-attention checkpoint into
  TTT-E2E, but it strengthens the claim that the fast-weight/update design
  space is active and evolving.

### In-Context / Neural Memory Architectures

Primary source:

- Titans OpenReview page:
  https://openreview.net/forum?id=8GjSf9Rh7Z

Why it matters:

- Titans presents a neural long-term memory module that learns to memorize
  historical context and was a NeurIPS 2025 poster.
- It is adjacent long-context memory work rather than the same TTT-E2E bridge
  question.

Manuscript positioning:

- Use as context for the broader move from pure attention windows toward
  learned memory/state mechanisms.
- Do not over-position it as direct competition unless evaluation surfaces
  overlap in the final paper.

### 3D / Multimodal TTT Follow-Ons

Sources:

- ZipMap / Linear-Time Stateful 3D Reconstruction via Test-Time Training:
  https://arxiv.org/html/2603.04385v3
- tttLRM project/arXiv surfaced in search:
  https://arxiv.org/html/2602.20160v1

Why it matters:

- These papers show TTT/LaCT-style fast-weight mechanisms spreading beyond
  language into 3D reconstruction and multimodal long-context settings.
- ZipMap explicitly uses local attention plus global large-chunk TTT blocks and
  reuses/preinitializes components from a pretrained vision model.

Manuscript positioning:

- Optional related-work sentence only. It supports the timeliness of
  pretrained-component reuse with TTT-like blocks, but the domain and metrics
  differ from our language-model setting.

## Venue/Framing Implication

The scan strengthens the revised "reuse / lower barrier to entry" framing, but
it also raises the bar for precision:

- Avoid saying we are the first to care about TTT warm-start compatibility.
- Say we provide a controlled empirical study of the bridge needed to adapt a
  pretrained FA Transformer into TTT-E2E.
- Put E1 at the center: the bridge contribution is isolated by comparing S2 to
  S2-minus under matched extension.
- Put E3 as practical guidance: small bridges recover most of the matched-budget
  bridge effect, while larger bridge budgets keep improving quality.

## Follow-Up Before Submission

- Re-run this scan immediately before final submission because the area is
  moving quickly.
- Ask Vijay/advisor whether they know of any unpublished or newly accepted
  TTT-E2E follow-up that should be cited or avoided in claims.
