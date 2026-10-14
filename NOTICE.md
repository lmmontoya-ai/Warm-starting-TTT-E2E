# Notice

This repository is an independent JAX implementation of the warm-start
experiments it describes. It re-implements the parts of TTT-E2E it needs
instead of vendoring the original release.

TTT-E2E itself is the work of Tandon et al.,
[End-to-End Test-Time Training for Long Context](https://test-time-training.github.io/e2e.pdf),
with code at [test-time-training/e2e](https://github.com/test-time-training/e2e).
The tokenized DCLM and Books3 datasets and the upstream checkpoints come from
that release and are not redistributed here. Their terms are set by their
owners, separately from this repository's MIT license. Books3 consists of
copyrighted books, so check that your use of it is permitted where you work.

Evaluation code refers to RULER, HELMET, LongBench v2, and other benchmarks.
Each has its own license and access terms.
