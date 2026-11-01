# IMPLEMENTATION ROADMAP

10 weeks × 10–12 h. Critical path = M1→M6 (Tier 1). M7–M9 are droppable to "report what exists" without invalidating the minimum reproduction; M10 is not droppable.

| Milestone | Tasks | Dependencies | Expected outputs | Acceptance criteria |
|---|---|---|---|---|
| **M0** (W1) Audit & environment | This audit (done); `uv init`, pyproject, `uv.lock`; clone upstream as reference; decide Python/torch/transformers pins on a Colab smoke test | — | repo skeleton; locked env; audit notes | `uv sync` works locally and on Colab; upstream `train.py` failure reproduced & understood |
| **M1** (W1–2) Data layer | `data.py` + validation; copy `RQ.csv` verbatim; stats table; tests | M0 | `rq-probe validate` output; dataset stats | All checks in EXPERIMENT_PROTOCOL §1 pass on CPU; split counts match paper §3.1 |
| **M2** (W2) Test scaffolding | pytest suite for pooling, PCA, metrics (synthetic fixtures; hand-computed cases) | M0 | passing `uv run pytest` | Tests listed in §Testing below green; suite runs <1 min CPU |
| **M3** (W3) Extraction pilot | `extraction.py`; Qwen3-4B fp16 on Colab T4, 200 examples, batch-size sweep {8,16,32}, timing + VRAM log; pin HF revision | M1, M2 | pilot artifact + resource report | Last-token gather matches batch=1 reference; resume works after manual kill; no OOM at chosen batch |
| **M4** (W4) Full extraction | Full 4,997-example run (last_token); artifact validation; retry/resume from interruptions | M3 | `outputs/embeddings/RQ_..._last_token.pt` (~1.9 GB) | 4,997 rows, no NaN/Inf, metadata complete; re-extract of 50 random rows ≡ saved rows (tolerance 1e-4 fp16) |
| **M5** (W5) PCA + probes | `pca_step.py` (fit all / fit train); vendor trainer + fixes; diffMean; train logistic+hinge per layer | M4 | PCA bases + PCA-space embeddings; probe files for 37 layers × 3 probes | Explained-variance of 64th PC <1% (paper App. A consistency check); validation never empty; probe train AUROC > test AUROC sanity |
| **M6** (W6) Layer-wise evaluation → **Tier 1** | `evaluate` + Fig-1-style plot + AUROC table with Hanley–McNeil CIs | M5 | AUROC table + figure | **Pattern matches paper Fig. 1 col. 1**: trained probes 0.8–0.9 mid-layers, diffMean lower, early-layer signal present |
| **M7** (W7) Mean pooling + alignment → Tier 2 | mean_token extraction run; alignment metrics (cosine/Spearman/Jaccard) | M6 | mean-vs-last comparison; alignment panel (Fig-2-style) | hinge–logistic cosine ≈ 1 ≫ diffMean–trained; top-20% Jaccard < bottom-20% (paper §5 pattern) |
| **M8** (W8) Robustness & error analysis | PCA fit-split comparison; error analysis; length-proxy check (paper §7 Table 2 analog); optional 3-seed probe stability | M7 | leakage-variant table; error-analysis section inputs | Conclusions unchanged under train-fitted PCA (or delta ≪ CI), else flagged in report |
| **M9** (W9) Writing | Report + slides; results tables auto-generated from metrics JSONL | M6 (+M7/M8 as available) | draft report + slides | Every number in the report traces to a config + artifact |
| **M10** (W10) Reproducibility pass | Fresh `uv sync` → README walkthrough on Colab; fix gaps; freeze `uv.lock`; final submission | all | final repo + README | Task §13 acceptance criteria 1–12 each demonstrably true |

**Critical-path dependencies:** M1→M2 can interleave; M3 blocks M4; M5 blocks M6; everything else is parallelizable with writing.

**Drop rules:** if GPU quota collapses in W3–4 → shrink M4 to train+test splits only (3,997 train+dev still allows probing but no clean test — instead drop mean-pooling M7 first, then M8, then reduce `selected_layers` to {0, 25%, 50%, 75%, last} before ever shrinking data integrity checks). Never drop: split preservation, join-by-id, test-set discipline.

## Test suite plan (M2; unit vs paper-validation split)

**Unit tests (synthetic, CPU):**
- Data: schema errors raise; `dev`→validation mapping; duplicate/missing ids; empty text; label map; per-split counts.
- Extraction: last-valid-token gather equals batch=1 reference on a tiny model (`hf-internal-testing/tiny-random-Qwen3ForCausalLM` or similar `[UNVERIFIED]` availability — else a stubbed forward); masked mean excludes padding; NaN/Inf detector trips.
- PCA: output dim; `fit_split=train` never sees test rows (deliberate contaminated-fixture test); basis reuse deterministic; layer ℓ transform uses layer ℓ basis only.
- Probes: labels {0,1}; diffMean direction = µ⁺−µ⁻ on a constructed tensor; hinge converts {0,1}→{−1,1}; logistic converges on separable blob; training deterministic under fixed seed (two runs → identical W).
- Metrics: AUROC vs `sklearn` on random scores; Hanley–McNeil CI matches upstream function output on a fixed vector; Spearman vs scipy; Jaccard top/bottom sizes = 20% and values match brute-force computation; direction invariance (scaling scores doesn't change AUROC/Spearman/Jaccard).
- Map-back: random W_pca + basis → reconstruct scores s(x_PCA) == s(x_orig) on random x (App. B closed form).

**Validated against paper-reported results (not unit-testable):** shape of layer-wise AUROC curve (M6 gate), alignment pattern (M7 gate), 64th-PC variance <1% (M5 gate). These are acceptance gates, recorded as such.
