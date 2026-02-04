# Reproduction Roadmap: TTT-E2E (End-to-End Test-Time Training for Long Context)

**Scope**
- Primary goal: reproduce the headline claims on long-context scaling, latency, and key benchmarks.
- Secondary goal: validate core ablations and failure modes (NIAH recall) on a smaller scale if compute is limited.

**Step 1: Collect Original Artifacts**
- Create a local `og_repo/` folder to keep the original code and paper together.
- Clone the official repo into `og_repo/e2e/`.
- Download the paper PDF into `og_repo/TTT-E2E.pdf`.
- Record the exact commit hash used for reproduction: `f73017b516781a7afee51237489476372012c171`.

**Step 2: Define Target Reproduction Claims**
- Target claims for our scaled reproduction (8× RTX 6000 Pro):
  - Scaling with context length (8K–32K) for a 760M model (fallback: 350M).
  - Constant inference latency vs context length and speedup vs full attention at 32K.
  - NIAH (RULER) recall gap between TTT-E2E and full attention.

**Step 3: Provision Compute and Storage**
- GPU: 8× RTX 6000 Pro (96GB) planned for a scaled reproduction at 760M @ 32K.
- Storage: ensure enough space for tokenized datasets (DCLM filter 8K and Books3).
- Networking: access to GCS (Requester Pays if needed).

**Step 4: Environment Setup (Repo Guidance)**
- Use the repo’s preferred workflow: JAX + CUDA with `uv`-managed Python deps.
- Install CUDA 12.8.1, cuDNN 9.8.0, NCCL 2.26.2 as indicated by the repo.
- Configure Weights & Biases (WANDB) for logging.

**Step 5: Acquire Datasets**
- From GCS:
  - `gs://llama3-dclm-filter-8k/`
  - `gs://llama3-books3/`
- Update config paths (`deploy_paths`) to point to local storage.

**Step 6: Reproduce the Training Pipeline**
- Stage A: Pre-train at 8K context length on DCLM-Baseline (760M; fallback 350M).
- Stage B: Extension fine-tune to 32K context length on Books.
- Use repo-provided configs in `configs/experiment/` and keep hyperparameters aligned with the paper (window size 8K, TTT mini-batch size, layers updated).

**Step 7: Reproduce Core Baselines**
- Full attention.
- SWA and Hybrid SWA+full.
- Mamba 2, Gated DeltaNet, and TTT-KVB (if supported in the repo).

**Step 8: Evaluation and Metrics**
- LM loss on held-out Books partition.
- Loss breakdown by token index (early vs late tokens).
- RULER NIAH evaluation for long-context recall.
- Long-sequence decoding: prefill 8K Books, decode 8K; evaluate with Qwen-3-8B-Base.

**Step 9: Efficiency and Latency Tests**
- Measure prefill latency across contexts (8K–128K).
- Compare prefill speed against full attention on the same hardware.
- Record training throughput and identify gradient-of-gradients overhead.

**Step 10: Validation Against Paper**
- Recreate Figures:
  - Scaling with context length (loss vs tokens).
  - Latency vs context length.
  - NIAH recall comparison.
- Compare deltas and document any deviations.

**Step 11: Document Reproduction Outcomes**
- Write a replication report with:
  - Configs and commit hashes.
  - Hardware, dataset versions, and training tokens.
  - Metrics tables and plots.
  - Deviations and suspected causes.

**Optional Step 12: Repository Mirror for Auditability**
- Keep `og_repo/` as a clean mirror of the original code + paper.
- Add a local README listing repo URL, commit hash, and paper citation.

---

**Cost & Time Estimates (TTT‑E2E + Full Attention + Mamba)**

**Assumptions**
- Tokens and steps come from the official TTT‑E2E configs (pretrain + extension).
- FLOPs ≈ `6 × parameters × tokens` (standard dense transformer estimate).
- Dense BF16 throughput used (vendor specs list sparse numbers; dense is ≈ 1/2).
- Full‑attention extension overhead:
  - 32K: 2.0–2.5× over SWA extension
  - 128K: 6.0–8.5× over SWA extension
- Mamba throughput: 1–5× faster than transformer baseline (range reflects uncertainty).
- RTX 6000 Pro throughput is estimated via bandwidth ratio and uses a wider efficiency band.

**Model Token/FLOPs Summary (from repo configs)**

| Model | Tokens (B) | FLOPs (1e18) |
| --- | --- | --- |
| 125M @ 32K | 2.642 | 2.0 |
| 350M @ 32K | 7.431 | 15.6 |
| 760M @ 32K | 15.965 | 72.8 |
| 1B @ 32K | 27.525 | 165.2 |
| 3B @ 32K | 57.252 | 1030.5 |
| 3B @ 128K | 57.252 | 1030.5 |

**RTX 6000 Pro (8× @ $6/hr)**

| Model | TTT‑E2E (SWA) cost/time | Full Attention cost/time | Mamba cost/time | Total (all 3) cost/time |
| --- | --- | --- | --- | --- |
| 125M @ 32K | $3–$9 (0.6–1.5h) | $4–$9 (0.6–1.6h) | $1–$9 (0.1–1.5h) | $8–$27 (1.3–4.5h) |
| 350M @ 32K | $28–$69 (4.6–11.5h) | $29–$74 (4.8–12.3h) | $6–$69 (0.9–11.5h) | $62–$212 (10.3–35.3h) |
| 760M @ 32K | $129–$321 (21.4–53.6h) | $135–$344 (22.4–57.4h) | $26–$321 (4.3–53.6h) | $289–$987 (48.1–164.5h) |
| 1B @ 32K | $292–$729 (48.6–121.5h) | $305–$781 (50.9–130.2h) | $58–$729 (9.7–121.5h) | $655–$2239 (109.2–373.2h) |
| 3B @ 32K | $1819–$4549 (303–758h) | $1906–$4874 (318–812h) | $364–$4549 (61–758h) | $4089–$13971 (682–2329h) |
| 3B @ 128K | $1819–$4549 (303–758h) | $2253–$6173 (375–1029h) | $364–$4549 (61–758h) | $4436–$15270 (739–2545h) |

**A100 SXM 80GB (8× @ $6/hr)**

| Model | TTT‑E2E (SWA) cost/time | Full Attention cost/time | Mamba cost/time | Total (all 3) cost/time |
| --- | --- | --- | --- | --- |
| 125M @ 32K | $4–$9 (0.7–1.5h) | $5–$9 (0.8–1.6h) | $1–$9 (0.1–1.5h) | $10–$27 (1.7–4.5h) |
| 350M @ 32K | $35–$69 (5.8–11.6h) | $36–$74 (6.1–12.4h) | $7–$69 (1.2–11.6h) | $78–$213 (13.0–35.6h) |
| 760M @ 32K | $162–$324 (27.0–54.0h) | $170–$347 (28.3–57.9h) | $32–$324 (5.4–54.0h) | $364–$995 (60.7–165.9h) |
| 1B @ 32K | $368–$735 (61.3–122.5h) | $385–$788 (64.2–131.3h) | $74–$735 (12.3–122.5h) | $826–$2258 (137.7–376.3h) |
| 3B @ 32K | $2294–$4588 (382–765h) | $2403–$4915 (401–819h) | $459–$4588 (77–765h) | $5155–$14090 (859–2348h) |
| 3B @ 128K | $2294–$4588 (382–765h) | $2840–$6226 (473–1038h) | $459–$4588 (77–765h) | $5592–$15401 (932–2567h) |

**H100 SXM (8× @ $12/hr)**

| Model | TTT‑E2E (SWA) cost/time | Full Attention cost/time | Mamba cost/time | Total (all 3) cost/time |
| --- | --- | --- | --- | --- |
| 125M @ 32K | $3–$6 (0.2–0.5h) | $3–$6 (0.2–0.5h) | $1–$6 (0.0–0.5h) | $6–$17 (0.5–1.4h) |
| 350M @ 32K | $22–$44 (1.8–3.7h) | $23–$47 (1.9–3.9h) | $4–$44 (0.4–3.7h) | $49–$135 (4.1–11.2h) |
| 760M @ 32K | $102–$204 (8.5–17.0h) | $107–$219 (8.9–18.2h) | $20–$204 (1.7–17.0h) | $230–$628 (19.1–52.3h) |
| 1B @ 32K | $232–$464 (19.3–38.6h) | $243–$497 (20.2–41.4h) | $46–$464 (3.9–38.6h) | $521–$1424 (43.4–118.7h) |
| 3B @ 32K | $1446–$2893 (120–241h) | $1515–$3100 (126–258h) | $289–$2893 (24–241h) | $3251–$8886 (271–741h) |
| 3B @ 128K | $1446–$2893 (120–241h) | $1791–$3926 (149–327h) | $289–$2893 (24–241h) | $3527–$9712 (294–809h) |

**H200 SXM (8× @ $16/hr)**

| Model | TTT‑E2E (SWA) cost/time | Full Attention cost/time | Mamba cost/time | Total (all 3) cost/time |
| --- | --- | --- | --- | --- |
| 125M @ 32K | $4–$7 (0.2–0.5h) | $4–$8 (0.2–0.5h) | $1–$7 (0.0–0.5h) | $8–$23 (0.5–1.4h) |
| 350M @ 32K | $29–$58 (1.8–3.7h) | $31–$63 (1.9–3.9h) | $6–$58 (0.4–3.7h) | $66–$179 (4.1–11.2h) |
| 760M @ 32K | $136–$272 (8.5–17.0h) | $143–$292 (8.9–18.2h) | $27–$272 (1.7–17.0h) | $306–$837 (19.1–52.3h) |
| 1B @ 32K | $309–$618 (19.3–38.6h) | $324–$662 (20.2–41.4h) | $62–$618 (3.9–38.6h) | $695–$1899 (43.4–118.7h) |
| 3B @ 32K | $1929–$3857 (120–241h) | $2021–$4133 (126–258h) | $386–$3857 (24–241h) | $4335–$11847 (271–741h) |
| 3B @ 128K | $1929–$3857 (120–241h) | $2388–$5235 (149–327h) | $386–$3857 (24–241h) | $4702–$12950 (294–809h) |

**B200 (8× @ $24/hr)**

| Model | TTT‑E2E (SWA) cost/time | Full Attention cost/time | Mamba cost/time | Total (all 3) cost/time |
| --- | --- | --- | --- | --- |
| 125M @ 32K | $2–$4 (0.1–0.2h) | $2–$5 (0.1–0.2h) | $0–$4 (0.0–0.2h) | $5–$14 (0.2–0.6h) |
| 350M @ 32K | $17–$35 (0.7–1.4h) | $18–$37 (0.8–1.5h) | $3–$35 (0.1–1.4h) | $39–$107 (1.6–4.4h) |
| 760M @ 32K | $81–$162 (3.4–6.7h) | $85–$173 (3.5–7.2h) | $16–$162 (0.7–6.7h) | $182–$497 (7.6–20.7h) |
| 1B @ 32K | $184–$367 (7.6–15.3h) | $192–$393 (8.0–16.4h) | $37–$367 (1.5–15.3h) | $412–$1127 (17.2–47.0h) |
| 3B @ 32K | $1145–$2290 (47.7–95.4h) | $1200–$2454 (50.0–102.2h) | $229–$2290 (9.5–95.4h) | $2574–$7034 (107.2–293.1h) |
| 3B @ 128K | $1145–$2290 (47.7–95.4h) | $1418–$3108 (59.1–129.5h) | $229–$2290 (9.5–95.4h) | $2792–$7688 (116.3–320.3h) |
