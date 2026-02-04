# Reproducing TTT-E2E

This workspace organizes our reproduction of the TTT-E2E paper.

- Target reproduction claims:
- Scaling with context length (8K–128K) for a 3B model.
- Constant inference latency and ~2.7× speedup vs full attention at 128K on H100.
- NIAH (RULER) recall gap between TTT-E2E and full attention.

- Original artifacts are in `og_repo/`.
- Paper: `og_repo/TTT-E2E.pdf`.
- Code snapshot (no git history): `og_repo/e2e/`.
- Snapshot metadata: `og_repo/README.md`.
- Research summary: `report.md`.
- Reproduction plan: `reproduction_roadmap.md`.

Original code snapshot commit: `f73017b516781a7afee51237489476372012c171`.
