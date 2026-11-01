# rq-replication

Partial replication of **"Rhetorical Questions in LLM Representations: A Linear Probing Study"**
(Yao, Anand, Zhuang, Jiang — [ACL 2026](https://aclanthology.org/2026.acl-long.5/),
[official code](https://github.com/ruyi101/rq-representation-probing)) at reduced model scale
(Qwen3-4B on a free Colab T4), for DS310 (NLP).

The study probes whether rhetorical-question intent is linearly decodable from LLM hidden
states, and — the key point — that probes with similar AUROC can find *different* directions
(decodability ≠ representational alignment).

## Status

Planning complete (see `docs/`), implementation started — Phase 0 (environment).

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffolding, locked env, Colab smoke test | **done** (pin step pending one Colab T4 run) |
| 1–2 | Data layer + metrics (CPU) | data layer **done**; metrics pending |
| 3–6 | Extraction → PCA → probes → layer-wise AUROC (Tier 1) | pending |
| 7–9 | Mean pooling + alignment (Tier 2), robustness, writing | pending |

Planning documents: [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) ·
[EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) ·
[REPOSITORY_DESIGN.md](docs/REPOSITORY_DESIGN.md) ·
[IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) ·
[RISK_REGISTER.md](docs/RISK_REGISTER.md) ·
[REPRODUCTION_DECISIONS.md](docs/REPRODUCTION_DECISIONS.md)

## Setup

```bash
uv sync
uv run pytest          # once tests exist (Phase 1+)
```

## Colab smoke test (Phase 0)

In a GPU-enabled Colab notebook:

```python
!git clone <this-repo-url> && %cd rq-replication
!pip install uv && uv sync
!uv run python scripts/colab_smoke_test.py
```

The script prints the exact versions + HF revision to pin (task M0 acceptance).

## Data and licensing notes

- `data/raw/RQ.csv` ships with the upstream repository (no LICENSE file there) and is
  **not redistributed here** — download it from the upstream repo into `data/raw/`.
  Splits: 3,200 train / 797 dev / 1,000 test; 2,332 rhetorical of 4,997.
- Model: `Qwen/Qwen3-4B` (Apache-2.0). Fallback: `Qwen/Qwen3-8B` (8-bit).
- Our code: MIT. Vendored upstream code retains attribution headers.
- The paper PDF is ACL-copyrighted and gitignored.

This is a course-scale **partial replication**, not a full reproduction — claims are
scoped in [docs/REPRODUCTION_DECISIONS.md](docs/REPRODUCTION_DECISIONS.md) §D8.
