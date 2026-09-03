# Final Experimental Program Before E5 — Goal-Oriented Plan

**Scope.** Everything that must run and be verified before any time or money goes to an external model (E5). When the last DoD here is met, the paper is submission-complete on in-house evidence, and E5 becomes a pure bonus with a kill-switch — never a dependency.

**State entering this plan (all preregistered gates discharged):**
- E1: bridge helps, Δ = 2.0767, 95% CI [2.0500, 2.0970] on the 64-batch manuscript surface; 5 paired seeds; mechanism = S2-minus trains smoothly to ~6.0 then plateaus; per-position gap uniform (1.90–2.01 across deciles).
- E2a: no retrieval complementarity (primary S2 vs S3 at 125M: 0.060 vs 0.084, 68 discordant, p=0.943; 760M tie 0.078/0.078). Complementarity branch deleted; E2b dropped.
- A0 closed: 64-batch re-evals reproduce historical ledger (S0 6.5879, S1 6.5423, S2 3.9169, S3 3.2722); 8-batch drift was sampling noise.
- Infrastructure proven twice: HF mirrors + restore, preflight/validity gates, `data=2` runtime profile on Prime Intellect 2×H100, golden-plan regression, cost ledger (~$80 spent of ~$1,000).

**Non-negotiables (inherited, restated):**
- Repo + `CANONICAL_RESULTS.md` are the source of truth. Every reported scalar: checkpoint restore → current float32 pipeline → `eval_batches=64`, `eval_batch_size=8`, 32K context (the manuscript surface).
- New runs get new `paper_run_id`s under a documented namespace; curated reports and reference snapshots read-only.
- Interpretation rules written **before** results are inspected (Phase 0 here).
- Same topology (`data=2`) and runtime profile for every run that enters a comparison.
- One GPU session for all remaining training; box terminated immediately after upload.

---

## Phase 0 — Pre-commit interpretation rules (Mac, $0)

**Goal:** extend the preregistration file so E3 and the S2-minus continuation are read by committed rules, not eyeballed — same discipline as E1/E2a.

**Implementation:**
1. Add to `PREREGISTRATION_REVISION_V2.md`:
   - **E3 frontier definition.** The frontier is Books32K 64-batch loss vs bridge token budget over five points: 0% (= S2-minus, seed-mean 5.9976), 5%, 10% (= S2, seed-mean 3.9208), 20%, 40%. New arms run at **one committed seed each** (name it, e.g. seed001) — the frontier is a characterization, not a hypothesis test, so single-seed points are acceptable *and must be labeled as such*; the E1 five-seed CI on the 0%→10% segment is the calibration for how much seed noise a single point carries (E1 seed std was ~0.01 on loss, ~0.025 on delta — record these as the noise yardstick).
   - **Shape classification rule** (descriptive, committed): let `G = loss(0%) − loss(10%) = 2.0767` (the full bridge effect). Classify the frontier as **threshold-like** if the 5% arm recovers ≥ 70% of G (i.e., loss(5%) ≤ loss(0%) − 0.7·G ≈ 4.54); **smooth** if 5% recovers between 30% and 70%; **budget-hungry** if 5% recovers < 30% and 20%/40% continue improving materially (> 0.10 each step); **saturating** if 20% and 40% are within 0.10 of the 10% anchor. Multiple labels may co-apply (e.g., threshold-like *and* saturating); report whichever the numbers earn.
   - **Curve-capture requirement:** each E3 arm logs extension-stage training loss at the canonical logging cadence; the deliverable plot is loss-vs-step for all arms overlaid on the S2 and S2-minus seed bands from E1.
   - **S2-minus continuation rule.** Mirror the historical S2 continuation protocol exactly: resume the final S2-minus checkpoint (committed seed, e.g. seed001) with optimizer state intact, original schedule/data/context, **+1,440 steps** (same as historical S2's +1,440). Preregistered read: the plateau-objection is **closed** if total improvement over +1,440 steps is < 0.25 (i.e., the continued S2-minus remains ≥ ~1.7 above the bridged path — vs the 2.08 gap, leaving the headline qualitatively intact; chosen as 5× the historical S2 continuation gain of 0.05, a generous allowance). If improvement ≥ 0.25, the plateau claim is weakened: report the continuation trajectory honestly, refit the E1 mechanism paragraph ("the gap narrows under extended training but remains ≥ X"), and do not claim persistence without this caveat. Either way the E1 *decision* (bridge helps at the matched budget) is untouched — continuation informs the mechanism narrative, not the preregistered gate.
2. Commit and push before provisioning.

**Verify / DoD:**
- Prereg diff committed; `uv run pytest -q` green (extend the prereg-content test to assert the new rules exist).
- The four frontier labels + the 0.25 continuation threshold are in the file with their justifications, datestamped before any new run starts.

---

## Phase 1 — Registry stages + dry-run validity (Mac, $0)

**Goal:** all remaining runs exist as first-class registry stages that pass the same validity machinery E1 did.

**Implementation:**
1. Add stages: `S2_BRIDGE_5PCT_125M`, `S2_BRIDGE_20PCT_125M`, `S2_BRIDGE_40PCT_125M` — each = bridge (scaled steps at batch 64: 240 / 960 / 1,920 steps for 5/20/40%, vs the canonical 480 @ 10%) → standard 32K extension (identical to the S0–S3/E1 extension config) → 64-batch eval. Parent: shared FA seed. Token accounting per arm recorded in the stage manifest.
2. Add `S2_MINUS_CONT_125M`: resume `S2_MINUS_seed001` final checkpoint **with optimizer state** (this is a continuation, not a fresh stage — assert `load_part=all`, mirroring the historical S2-continuation config), +1,440 steps, same data/schedule/context.
3. Extend the preflight validity report to cover the new stages:
   - E3 arms: extension config identical to canonical (golden-plan-protected) extension; bridge configs differ **only** in step count (assert: same LR schedule shape, batch, data fingerprints, seed policy); budget→steps→tokens arithmetic asserted (a 40% bridge must plan 4× the bridge tokens of 10%).
   - Continuation: resumes the correct parent checkpoint (assert run-id lineage), optimizer state restored (the *inverse* of E1's fresh-optimizer assertion — make this an explicit branch in the checker, not a skipped check), step budget = +1,440.
4. Golden-plan regression: canonical stages and the E1 stages still resolve to byte-identical plans after registry edits.

**Verify / DoD:**
- `uv run pytest -q` green including new stage tests, budget-arithmetic tests, the optimizer-state-direction test, and golden-plan snapshots.
- Dry-run plans for all four new stages emit correct parents, budgets, output roots (`revision_v2_e3_*`, `revision_v2_s2minus_cont_v1`), and the `data=2` profile.
- Validity report enumerates every executed check; any skipped check = FAIL.

---

## Phase 2 — Single GPU session (provision → run → eval → upload → terminate)

**Goal:** all remaining training in one box-day. Budget ceiling for the session: **$120** (expected ~$80–95).

**Implementation sequence:**
1. Provision the same class of box (2×H100, the committed `data=2` profile). Restore shared parents once (FA seed; `S2_MINUS_seed001` final for the continuation). Verify fingerprints; stage data; rerun the 10-minute calibration + kill-resume smoke (new box ≠ old box; re-verify, don't assume).
2. Launch order (longest first to overlap with monitoring): 40% bridge arm → 20% → 5% → S2-minus continuation. Approx GPU-hr: 40% ≈ 9.5, 20% ≈ 5.5, 5% ≈ 2.5, continuation ≈ 4.5 (it's 3× the extension's 480 steps) → ~22 GPU-hr ≈ $50–55 training.
3. On completion: 64-batch manuscript-surface eval of all four new final checkpoints **plus** per-position NLL for the three E3 arms and the continued S2-minus (same eval pass, near-free, feeds the mechanism narrative).
4. Export training curves (all four runs) at full logging resolution.
5. Upload everything (checkpoints, manifests, curves, eval reports) to HF under the new namespaces; verify upload completeness (file count + spot-restore one checkpoint); update cost ledger; **terminate the box.**

**Verify / DoD:**
- Four runs complete with train + eval manifests; any failure documented with cause.
- 64-batch losses exist for all four; per-position NLL exists for all four.
- Training curves serialized (not just W&B-resident — committed CSV/parquet in the report bundle).
- HF upload verified; box terminated same day; ledger entry ≤ $120.

---

## Phase 3 — Preregistered read of the frontier + continuation (Mac, $0)

**Goal:** numbers → committed labels → one analysis memo. No improvisation.

**Implementation:**
1. Assemble the five-point frontier table: budget %, bridge tokens, total upstream tokens, Books32K 64-batch loss, Δ vs 0% anchor, fraction of G recovered. Single-seed points labeled; E1 seed-noise yardstick quoted alongside.
2. Apply the Phase-0 shape classification; produce the frontier figure (loss vs budget, log-x, with the E1 seed bands at 0% and 10%) and the overlaid training-curve figure.
3. Apply the continuation rule: total improvement over +1,440 steps vs the 0.25 threshold; produce the continuation trajectory figure mirroring the paper's existing Fig-4 style (S2-minus continuation vs the S2/S3 reference lines).
4. State the token-budget confound verbatim per the preregistration: more bridge = more upstream short-context tokens; the frontier characterizes the *budget*, not a pure structural ablation.
5. Write `reports/revision_v2/e3_frontier_analysis.md`: tables, both figures, the earned shape label(s), the continuation verdict, the confound paragraph, and the practitioner sentence the result supports (e.g., "a bridge of ~N% of pretraining tokens recovers ~M% of the full bridge effect").

**Verify / DoD:**
- Every number in the memo traces to a 64-batch eval manifest (cite run-ids inline).
- Shape label matches the committed rule arithmetic (show the arithmetic).
- Continuation verdict states which branch of the preregistered rule fired and what the manuscript's mechanism paragraph may now claim.
- Memo pushed; HF bundle linked.

---

## Phase 4 — Claims-evidence audit (Mac, $0; can start in parallel with Phase 2)

**Goal:** the artifact that disciplines the manuscript — one row per claim the paper will make; no claim without a row.

**Implementation:**
1. Create `docs/CLAIMS_AUDIT.md` with columns: **claim (verbatim as the paper will state it) | evidence artifact (ledger entry / report file / run-id) | strength label | section**. Strength labels: `preregistered-confirmed` / `characterized` / `observed` / `hypothesis-future-work`.
2. Seed it with the known spine (each row to be finalized against Phase 3 outputs): bridge isolation (preregistered-confirmed, E1); warm-start tax 0.65 at 125M / 0.32 at 760M on the 64-batch surface (characterized); gap narrows across two tested scales — no trend claim (observed); no retrieval complementarity at adequate paired power (preregistered-confirmed null, E2a); both entry routes share the retrieval floor — bridge changes the entry path, not the capability profile (observed, discussion-grade); marginal-reuse cost case at 125M (characterized; 760M cost appendix-only); frontier shape + practitioner guidance (characterized, from Phase 3); plateau persistence under continuation (per Phase-3 verdict); reproducibility infrastructure (stated, with HF links).
3. Adversarial pass: for each row, write the strongest one-line reviewer objection and where the paper answers it. Any row whose objection lacks an answer either gets demoted (weaker claim) or deleted.
4. Cross-check against the two desk rejections: every named critique (bridge not established / limited ablations / narrow eval / cost-quality generality) must map to specific rows that now answer it — this mapping becomes the cover-letter skeleton.

**Verify / DoD:**
- Every abstract-level and contributions-level claim has a row; every row has a real artifact path.
- Zero rows with unanswered objections at full strength.
- The rejection-critique mapping is complete: four critiques, each pointing at confirmed rows.
- Reviewed once by you in a separate sitting from when it was written (cheap bias check).

---

## Phase 5 — Manuscript-readiness gate (the exit of this plan)

**Goal:** declare the experimental program closed and the writing phase open; E5 becomes optional.

**Exit checklist — all must hold:**
1. Phases 0–4 DoDs met; `uv run pytest -q` green on the final commit.
2. `CANONICAL_RESULTS.md` updated with the E3/continuation rows (64-batch surface) and asserts in the ledger test.
3. Cost ledger current; total program spend ≤ $250 (expected ~$170–200).
4. No GPU resources running.
5. The two external dependencies are **in motion** (not necessarily resolved): (a) OJ-CS submission status requested from Vijay in writing; (b) scoop/currency scan scheduled for the related-work pass (search for warm-starting/conversion into TTT-style architectures published since the TTT-E2E preprint, and the TTT-E2E paper's own publication status).
6. A one-paragraph decision note in the repo log: experimental program closed on [date]; remaining open item = E5, classified **bonus-tier, time-boxed (3–4 engineering days, profile→import→audit→probe before any training dollar), kill-switch on audit failure, cannot delay submission**.

**DoD:** the note in (6) is committed. From that commit forward, all project time is manuscript time unless E5's time-box is explicitly opened — and the paper must already be drafted before it is.

---

## Sequence and budget summary

| Step | Where | Cost | Wall-clock |
|---|---|---|---|
| Phase 0 — prereg extension | Mac | $0 | half-day |
| Phase 1 — stages + dry-run | Mac | $0 | half-day to a day |
| Phase 2 — GPU session (E3 ×3 + continuation + evals) | box | ~$80–95 (cap $120) | one box-day |
| Phase 3 — preregistered read | Mac | $0 | half-day |
| Phase 4 — claims audit | Mac (parallel w/ Phase 2) | $0 | one day |
| Phase 5 — readiness gate | Mac | $0 | an hour |

**Total new spend: ~$80–95. Program total: ~$170–200 of $1,000.** Everything after Phase 5 is writing — and E5, if and only if the draft exists first.