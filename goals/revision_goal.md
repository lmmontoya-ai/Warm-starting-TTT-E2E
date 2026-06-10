**Goal**
Run revision-v2 experiments in a cheap NVIDIA GPU environment, with the Mac used only as the control plane. The GPU machine becomes the single execution plane for checkpoint restore, data staging, preflight, training, and eval.

**Definition Of Done**
Done means:

1. Docs/code are committed and pushed.
2. One NVIDIA environment is provisioned for both preflight and training.
3. Shared read-only parent checkpoints are restored exactly once.
4. Tokenized DCLM/Books data is staged with matching fingerprints.
5. E1 preflight passes.
6. Current-pipeline S1/S2/S3 re-eval completes.
7. Five paired E1 seeds complete.
8. E1 is interpreted only through the preregistered thresholds.

---

**Phase 0: Lock Local State**
Commit and push:

- checkpoint download docs
- preregistration
- S2-minus registry promotion
- E1 preflight code/tests
- golden-plan protections

DoD:

```text
Remote branch has all no-compute gates.
No secrets/checkpoints/datasets committed.
```

---

**Phase 1: Provision One NVIDIA Execution Box**
Use one box for everything, not a tiny preflight box followed by a second training box. Data staging is the expensive part; avoid doing it twice.

Recommended:

- A100/H100/H200 if reasonably cheap
- enough disk for checkpoints + tokenized data + outputs
- Ubuntu/Linux
- network access to Hugging Face
- `uv`, git, JAX runtime deps

DoD:

```text
Repo cloned.
Dependencies installed.
HF public result repos reachable.
This same box can run preflight, eval, and E1 seeds.
```

---

**Phase 2: Restore Shared Read-Only Parents**
This is the big correction: do **not** restore parents per seed.

Restore each parent once into a shared, read-only parent run/root, then make all E1 seed runs resolve to those exact copies.

Required parents:

```text
FA parent for S2-minus:
S0_PRETRAIN_FA_125M / pretrain-125m-fa

Bridge parent for S2:
S2_ADAPT_125M / adapt-125m-e2e-8K-from-fa
```

Source:

```text
Repo: Luxel/ttt-e2e-125m-results
Paper run: protocol_r_125m_main_v1
```

Example target namespace:

```text
revision_v2_shared_parents
```

All ten E1 runs then read from:

```text
checkpoints/revision_v2_shared_parents/pretrain-125m-fa
checkpoints/revision_v2_shared_parents/adapt-125m-e2e-8K-from-fa
```

If the registry cannot express a shared parent root, implement the small orchestrator/registry extension before launching E1. Do not duplicate parent restores across seeds.

DoD:

```text
FA parent restored once.
Bridge parent restored once.
Both have latest.json and restore manifests.
Both are fingerprinted/verified once.
All E1 seed plans resolve parents to these shared paths.
Parent dirs are treated read-only.
```

---

**Phase 3: Stage Tokenized Data**
This is a real logistics gate, not a footnote.

Preferred source: existing tokenized DCLM-8K and Books32K copies from project storage/cloud backup. Do not assume raw Books3 can be re-downloaded; Books3 has takedown history.

Required roots:

```text
/root/ttt-e2e-data/dclm_filter_8k
/root/ttt-e2e-data/books3
```

Expected path:

1. Locate existing tokenized data archive/storage.
2. Transfer to GPU box.
3. Generate/verify fingerprint sidecars.
4. Compare fingerprints against canonical expected fingerprints.

Fallback only if necessary:

```text
Re-tokenize from available raw data, then prove fingerprints match.
```

If fingerprints do not match, stop. Do not run production E1 with `--allow-missing-fingerprints`.

DoD:

```text
DCLM-8K and Books32K token roots exist.
Fingerprint sidecars exist.
Fingerprints match canonical comparison surfaces.
Expected staging time is recorded.
```

---

**Phase 4: E1 Internal-Validity Preflight**
Preflight must assert that S2 and S2-minus differ only by bridge lineage.

Required assertions:

- S2 resumes from shared bridge parent:

```text
adapt-125m-e2e-8K-from-fa
```

- S2-minus resumes from shared FA parent:

```text
pretrain-125m-fa
```

- Same:
  - Books32K train/eval fingerprints
  - LR schedule
  - optimizer
  - warmup
  - global batch
  - steps
  - dtype
  - sharding
  - eval config
  - model architecture except parent lineage

- S2 extension starts with fresh optimizer state.
- S2-minus uses params-only partial restore.
- Golden-plan regression still passes.

DoD:

```text
Preflight PASS.
Allowed differences are only lineage, seed/run IDs, and intentional output paths.
No optimizer/data/config confound.
```

---

**Phase 5: Restore And Re-Evaluate Current Baselines**
Restore final baseline checkpoints for current-pipeline eval:

```text
S1_125M / ext-125m-swa-32K-from-fa
S2_125M / ext-125m-e2e-32K-from-fa-bridge
S3_125M / ext-125m-e2e-32K
```

Optional:

```text
S0_125M / ext-125m-fa-32K
```

Then re-evaluate through the current checkpoint-based float32 pipeline. Do not use ledger losses as E1 comparators.

DoD:

```text
S1/S2/S3 restored.
Current float32 eval completed.
Results live under a new revision-v2 eval paper_run_id.
```

---

**Phase 6: Run E1 Five Paired Seeds**
Run five paired seeds up front.

```text
S2_seed001       vs S2_MINUS_seed001
S2_seed002       vs S2_MINUS_seed002
S2_seed003       vs S2_MINUS_seed003
S2_seed004       vs S2_MINUS_seed004
S2_seed005       vs S2_MINUS_seed005
```

All runs use the same shared parents from Phase 2.

Phase 6 produces numbers only:

```text
delta_bridge_s = loss(S2_MINUS_s) - loss(S2_s)
```

Interpretation is deferred entirely to Phase 7’s preregistered thresholds. No point-estimate conclusions here.

DoD:

```text
5 paired deltas produced.
Seed-level bootstrap CI computed.
Paired per-batch bootstrap computed as eval sanity check.
No conclusions written before Phase 7.
```

---

**Phase 7: E1 Decision Gate**
Use the preregistered rule, not eyeballing.

```text
mean_delta = mean(loss(S2_MINUS) - loss(S2))
```

Decision ladder:

- **Helps:** `mean_delta >= 0.10` and CI lower bound `> 0`
- **Harmful:** `mean_delta <= -0.10` and CI upper bound `< 0`
- **Negligible:** CI is tightly inside the negligible band
- **Inconclusive:** 5-seed CI cannot resolve the registered threshold

For inconclusive:

```text
If budget is healthy, expand to 10 paired seeds.
If still inconclusive, report unresolved and do not make a bridge headline claim.
```

DoD:

```text
E1 decision memo exists.
It states result, CI, preregistered category, and manuscript implication.
```

---

**Phase 8: E2a Retrieval Proxy**
Run after E1 exists.

Primary confirmatory test:

```text
S2_125M vs S3_125M
```

Secondary:

```text
S2_125M vs S2_MINUS_125M
S2_MINUS_125M vs S1_125M
```

Stats:

- same examples across conditions
- Wilson intervals
- McNemar paired test
- discordant-pair count
- fixed binarization rule if needed

Proxy-only evidence can support a complementarity hypothesis, but should not become the headline unless E2b corroborates.

---

**Parallel Track: 760M Re-Eval**
This is not on the E1 critical path. It can run any time after the GPU environment and data are staged.

Restore:

```text
S2 / ext-760m-e2e-32K-from-fa-bridge
S3 / ext-760m-e2e-32K
S2_ADAPT / adapt-760m-e2e-8K-from-fa, only if needed
```

Use only for:

```text
quality/capability second-scale checks
```

Do not use for homogeneous cost claims.

---

**Phase 9: Optional E3 Budget Ablation**
Only run if E1 shows the bridge helps.

Sweep:

```text
5%
20%
40%
```

Anchor against existing 10%.

Interpretation caveat: more bridge budget also means more upstream tokens, so gains cannot be attributed purely to bridge structure.

---

**Stop-Loss**
Keep the tighter tripwires:

```text
Pause at $100 if shared parents, data fingerprints, preflight, and S1/S2/S3 current re-eval are not complete.

Pause at $300 if E2a is not complete.

Pause at $600 unless E3 or optional E2b/E5 has a clear manuscript need.
```

---

**Immediate Next Actions**
1. Commit/push the checkpoint download docs.
2. Provision one NVIDIA box for both preflight and E1.
3. Restore shared parents once:
   - `pretrain-125m-fa`
   - `adapt-125m-e2e-8K-from-fa`
4. Stage tokenized DCLM/Books and verify fingerprints.
5. Run E1 preflight.
6. Restore/re-eval S1/S2/S3.
7. Launch the five paired E1 seeds.
<conversation with my advisor>
8:45 PM
Good clarification — if everything you're seeing is on-demand on both platforms, the comparison simplifies, and this Prime Intellect listing actually wins it.

Per-GPU math: $4.70/hr ÷ 2 = $2.35/GPU-hr — beating the Vast 2× NVL ($2.59) and even matching the Vast 8× Japan box ($2.33) that you couldn't use efficiently anyway. And the spec sheet has no weak points for your workload: 2× H100 80GB (per-device batch 4 at global batch 8 — the clean topology), 2.5TB disk (tokenized data + parents + ten seed runs of outputs, no squeeze), 256GB RAM and 40 CPUs (comfortable for data loading and tokenization fallback if Phase 3 forces it), US location (good HF transfer speeds), ~3 min spin-up.

Two genuine advantages over the Vast option beyond price:

It's secure cloud (Massed Compute datacenter), not a marketplace host. Vast's listings are individual hosts of varying provenance; this is a managed datacenter VM. For a multi-day occupancy where a host disappearing mid-seed-run costs you a re-stage, that reliability difference is worth more than the 24¢/GPU-hr you'd save... except here you're saving money and getting it. No trade-off to weigh.

A VM, not a container. Vast gives you a Docker container; this gives you a full VM. Marginally easier for your case — persistent environment across reboots, no container-image constraints on the JAX/CUDA stack, simpler to snapshot.

One spec to note, not a problem: it's PCIe, not SXM (the badge says PCIE despite your SXM5 filter — the filter was set to "All" socket-wise in effect). PCIe H100s have lower NVLink bandwidth between the pair, which matters for tensor-parallel workloads but barely registers for data=2 data-parallel on a 125M model — your inter-GPU traffic is one small gradient all-reduce per step. Irrelevant at this scale.

Two checks before deploying:

Spot toggle is on in your filter but this is priced on-demand — flip through to see if a spot/interruptible variant of this same cluster exists. Even at on-demand $2.35/GPU-hr you're fine ($65 for the five seeds), but if a spot version sits at ~$1.50, that's $40 saved for runs that checkpoint anyway. If spot doesn't exist here, on-demand at this price is completely defensible — at these totals, reliability is worth more than the delta.
Bandwidth/egress pricing — Vast charges "plus bandwidth"; check whether Prime Intellect/Massed Compute meters transfer. With a few hundred GB inbound for data staging, confirm it's free or trivial.
If both check out: take this one. Then the sequence from before applies unchanged, with one edit — the runtime profile freezes data=2 on this box as the committed topology, preflight treats topology as must-match, and you rerun the 10-minute calibration + kill-resume test on this machine before launching seeds. At $4.70/hr total, your full E1 (staging + preflight + re-evals + five paired seeds) lands around $80–100 all-in, finishing in roughly a day of wall-clock.

Spot is not available, this is on demand, bandwidth is free

8:47 PM
Then it's decided — deploy it. On-demand at $2.35/GPU-hr with free bandwidth and datacenter reliability beats everything else you've surveyed, and the spot question dissolves: there's nothing cheaper to fall back to, and the reliability you'd normally pay extra for comes included. Free bandwidth also quietly removes the last cost unknown — data staging is now purely a time cost, and the budget has no asterisks left.

Updated numbers on this box, all-in at $4.70/hr:

Item	Wall-clock	Cost
Setup + staging + preflight + S1/S2/S3 re-eval	~3–5 hr	~$15–25
Five paired seeds (~25 GPU-hr at data=2)	~12–15 hr	~$60–70
E1 complete	~1 day	~$80–95
10-seed expansion if inconclusive	+~13 hr	+~$60
E2a + 760M re-eval (inference)	a few hr	~$25–40
Worst realistic path through the decision gate: ~$200 total, against your $1k. The stop-loss tripwires don't move.

One behavior change now that you're on-demand rather than spot: interruptions are no longer the thing to engineer around — idle time is. The box bills $4.70/hr whether it's training or waiting, so ~$113/day idle. The work itself fits in about a day, but the program has built-in human pauses (the Phase 7 decision memo to Vijay being the big one). So: when E1's seeds finish and the deltas are computed, package outputs, push results/manifests off the box, and tear it down before the interpretation pause — 3-minute spin-up and documented HF restore means resurrection is trivial. Still run the kill-resume test in your smoke sequence, though: on-demand reduces interruption risk, it doesn't eliminate VM restarts, and a multi-hour extension run should survive one regardless.

Deploy sequence, consolidated: provision → clone branch, install deps → commit the revision_v2 runtime profile pinned to data=2 on this topology → restore shared parents once (FA + bridge) → stage tokenized DCLM/Books, verify fingerprints → preflight green (all checks executed) → 10-min calibration + kill-resume test → S1/S2/S3 current-pipeline re-eval → launch the five paired seeds.

Next time we talk, the conversation changes kind: it stops being plans and reviews and becomes five numbers read against thresholds you froze weeks ago. Good luck with the launch.
</conversation with my advisor>