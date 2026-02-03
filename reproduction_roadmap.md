# Reproduction Roadmap: TTT-E2E (End-to-End Test-Time Training for Long Context)

**Scope**
- Primary goal: reproduce the headline claims on long-context scaling, latency, and key benchmarks.
- Secondary goal: validate core ablations and failure modes (NIAH recall) on a smaller scale if compute is limited.

**Step 1: Collect Original Artifacts**
- Create a local `og_repo/` folder to keep the original code and paper together.
- Clone the official repo into `og_repo/e2e/`.
- Download the paper PDF into `og_repo/TTT-E2E.pdf`.
- Record the exact commit hash used for reproduction.

**Step 2: Define Target Reproduction Claims**
- Minimum set:
  - Scaling with context length (8K–128K) for a 3B model.
  - Constant inference latency and ~2.7× speedup vs full attention at 128K on H100.
  - NIAH (RULER) recall gap between TTT-E2E and full attention.
- Optional (if compute allows):
  - Scaling with training compute across model sizes.
  - Long-sequence decoding evaluation (Qwen-3-8B evaluator).

**Step 3: Provision Compute and Storage**
- GPU: H100 or A100 recommended for 3B @ 128K; otherwise scale down to 125M–760M and 16K–32K contexts.
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
- Stage A: Pre-train at 8K context length on DCLM-Baseline.
- Stage B: Extension fine-tune to target context length (16K/32K/64K/128K) on Books.
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
