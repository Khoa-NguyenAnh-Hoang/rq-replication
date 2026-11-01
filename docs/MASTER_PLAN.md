# MASTER PLAN — Replication of "Rhetorical Questions in LLM Representations: A Linear Probing Study" (ACL 2026)

Course project for DS310 (NLP). Planning document — no pipeline implemented yet.

Paper: Yao, Anand, Zhuang, Jiang (ACL 2026 Long, pp. 155–172), [2026.acl-long.5](https://aclanthology.org/2026.acl-long.5/). Official code: [ruyi101/rq-representation-probing](https://github.com/ruyi101/rq-representation-probing) (audited 2026-11-01 @ default branch).

---

## 1. Research objectives

Reproduce the paper's central claims at reduced model scale, preserving its methodological distinction between **classification performance** and **representational alignment**:

- **RQ1 (Layer-wise):** At which Transformer layers is rhetorical intent linearly decodable? Paper finding: signals emerge early; last-token representations are most stable; trained probes peak ≈ 0.85–0.9 AUROC on RQ at intermediate layers (§4, Fig. 1).
- **RQ2 (Probe comparison):** diffMean vs logistic vs hinge. Paper finding: hinge and logistic are nearly identical; both beat diffMean on RQ but barely on SRAQ (§5, Fig. 1).
- **RQ3 (Alignment):** Do probes with similar AUROC find the same direction? Paper finding: **no** — hinge–logistic agree (cosine ≈ 1), diffMean disagrees (cosine < 0.7 on RQ, ≈ 0.5 on SRAQ); top-20% Jaccard ≈ 0.25 within-dataset, < 0.2 under transfer (§5–6, Figs. 2–4).

The AUROC ≠ alignment distinction (§5, §6, §8) must survive into our implementation, metrics, and report.

## 2. Scope and boundaries

**In scope (minimum reproduction):** RQ dataset only, one decoder-only model (Qwen3-4B), last-token representations, per-layer PCA-64 (published protocol), all three probes, test AUROC + Hanley–McNeil CIs.

**Out of scope / explicitly excluded:** full-size models (Qwen3-32B, Llama-3.3-70B), causal steering (App. F — needs generation + GPT-5.1 scoring), Grassmann/geometric analysis (App. C), ModernBERT comparison (App. E), Vietnamese dataset. See REPRODUCTION_DECISIONS.md.

## 3. Paper-to-code traceability matrix

Legend: **P** = paper page/section, **C** = upstream file. `[UNVERIFIED]` = could not be confirmed from released artifacts.

| # | Paper component | Original methodology | Official implementation | Planned implementation | Deviation |
|---|---|---|---|---|---|
| 1 | Datasets (P §3.1 p.156) | RQ: 4,997 Twitter questions, 2,332 rhetorical, 3,200/797/1,000 split; `question_with_context` formulation | `data/RQ.csv` (verified: 4,997 rows, splits 3,200/797/**dev**/1,000, `label`≡`binary_label`, ids 0..4996) | Same CSV, untouched, validated at load | None (SRAQ excluded — Tier 3 only) |
| 2 | Last-token repr. (P §3.2 p.157) | `h_T` from every layer, decoder-only | `generate_embeddings.py`: batch=1, `h[0,-1,:]`, no truncation, `dtype="auto"`, no resume | Batched fp16 extraction, per-example last-valid-token gather, `max_length=512`, checkpoint/resume, full metadata | Engineering only; math identical |
| 3 | Mean pooling (P Eq. 1 §3.2) | mean over tokens t=1..T | `generate_embeddings_mean.py`: `h[0,:,:].mean(dim=0)` (batch=1, no padding) | Attention-mask-weighted mean (identical at batch=1, correct when batched) | None |
| 4 | PCA-64 (P §3.2 p.157, App. A p.166) | k=64 per dataset×model×formulation; App. B p.166: mean µ "computed on the **training split**" | `pca_transform.py`: sklearn PCA per layer, fit on **ALL rows (train+dev+test)**, float32, CPU | Primary = published protocol (fit on all rows, documented); leakage-controlled variant = fit on train only, config flag `pca.fit_split` | **Paper↔code contradiction, documented; both variants produced** |
| 5 | diffMean (P Eq. 2 §3.3) | w = µ⁺ − µ⁻ on training split; score w·h | `diffmean_analysis.ipynb`: train-split only ✓, direction **unit-normalized** before projection, AUROC + Hanley–McNeil CI | Same math, ported to script; normalization kept (AUROC-invariant) | None |
| 6 | Logistic probe (P §3.3) | linear, cross-entropy, train split, validation-selected | `train.py`/`model.py`/`trainer.py`: `LinearModel` (no bias), SGD lr=0.1, wd=0, batch 256, max 1000 epochs, early stop patience 20 on dev loss; seeds 42 | Reuse upstream code with 1 fix (below); hyperparams = code defaults, sensitivity check in Tier 3 | Hyperparams ambiguous (see §4.2) |
| 7 | Hinge probe (P §3.3) | linear SVM objective | `utils.hinge_loss`: margin 1.0, {0,1}→{−1,1}, mean; same trainer | Same as #6 | Same as #6 |
| 8 | Split handling (P §3.1 "original splits") | train/val/test | `dataset.py` selects `dataset=="validation"`; shipped CSV uses **`dev`** → validation empty → `trainer.py` raises "Dev split is empty" → **`train.py` crashes on shipped `RQ.csv`** | Map {`dev`,`validation`,`val`} → validation at load; add test | **Upstream bug; minimal fix required to run at all** |
| 9 | AUROC (P §3.4) | per layer, per split, held-out | `diffmean_analysis.ipynb` / `projection_AUROC.ipynb`: sklearn `roc_auc_score` on projections of **unit-normalized** directions; Hanley–McNeil 95% CI (z=1.96) | Same function, ported; bootstrap CI added in Tier 3 | None |
| 10 | Spearman (P Eq. 3 §3.4, Fig. 2 mid) | rank correlation of two probes' scores | `spearman_corr.ipynb`: scipy `spearmanr` on **test-split** projections, **PCA space**; bootstrap CI n=2000, seed 0 | Same, ported; space recorded in artifact metadata | None |
| 11 | Jaccard top/bottom-p (P Eq. 4, Fig. 2 bottom: p=20%) | Jaccard of top/bottom 20% ranked by each probe | `jaccard_index.ipynb` (within-dataset): frac=**0.2**, test split ✓; `spearman_and_jaccard.ipynb` (transfer): frac=**0.10** (function default) | frac=0.2 everywhere; upstream 0.1 noted | **Upstream inconsistency vs paper, resolved in favor of paper** |
| 12 | Cosine between directions (P Fig. 2 top) | layer-wise cosine between probes | `within_dataset_cosine.ipynb`: unit-normalized W vectors, **PCA space** | Same; normalization explicit | None |
| 13 | PCA→embedding map-back (P App. B p.166–167) | w_x = Wᵀw_z, b_x = b − w_xᵀµ | `map_back.ipynb`: `einsum('lk,lkd->ld')` ✓; no bias (model has none, b=0); mean not re-added (cosine-invariant) | Same math + bias term for completeness; unit test vs closed-form | None material |
| 14 | Cross-dataset transfer (P §6, Figs. 3–4) | directions mapped to original space, projected onto other dataset's raw embeddings | `proj_transfer.ipynb`: ✓ original space; committed cell computes AUROC on **all rows** (test-split code commented out) | Test split only (paper §3.3: "all results reported on the test split") | **Upstream notebook-state deviation, resolved in favor of paper** |
| 15 | Qualitative analysis (P §7, Tables 1–2) | diffMean rankings at layer 48 of Qwen3-32B | Not in released code (private notebooks) | Tier 3: same procedure on our model/layer | Model differs |
| 16 | Steering (P App. F) | activation steering + GPT-5.1 scoring | Not released | **Excluded** | See REPRODUCTION_DECISIONS.md |

## 4. Upstream audit summary

### 4.1 Reusable components (with attribution comments kept in our sources)

| File | Purpose | Reusable? | Problems | Planned treatment |
|---|---|---|---|---|
| `data/RQ.csv` | dataset | Reuse directly | Split named `dev`, not `validation` | Copy verbatim into `data/raw/`; loader maps `dev` |
| `generate_embeddings.py` / `_mean.py` / `_bert.py` | extraction | Reuse with modifications | batch=1 (slow), no truncation/resume/metadata, `dtype="auto"` (bf16 unusable on T4) | Rewrite as one batched, checkpointing extractor; logic kept equivalent |
| `pca_transform.py` | per-layer PCA | Reuse with modifications | Fits on all rows (leakage per App. B); no fit-split option; docstring says 256 | Add `fit_split` option; keep math |
| `linear_model_training/{dataset,model,trainer,utils,train}.py` | probes | Reuse with modifications | **Crashes on shipped data** (`dev` vs `validation`); reloads .pt per layer; no AUROC out | Vendor with attribution + minimal fixes; load embeddings once; add AUROC |
| `run_training.sh`, `run_training_loop.sh` | hyperparam examples | Exclude | Leftover private config (`Hong-1201`, `embeddings_synthetic_last`), values contradict `train.py` defaults | Hyperparams taken from `train.py` defaults; ambiguity documented |
| `diffmean_analysis.ipynb` | diffMean + AUROC | Port logic | Committed cell points at `embeddings_mean_pca_64`; JSON dump of [L,D] directions | Port to `scripts/` |
| `linear_model_analysis/*.ipynb` | metrics | Port logic | Private paths (`Hong-*`, `synthetic`), inconsistent roots | Port to `scripts/` |
| `pca_map_back/*.ipynb` | map-back + transfer | Port logic | Transfer AUROC on all rows; transfer Jaccard frac 0.1; two referenced notebooks gitignored (not public) | Port to `scripts/`; resolve per paper |

### 4.2 Findings that shape our design

1. **PCA leakage contradiction** (P App. B vs `pca_transform.py`): the paper's own math says train-split mean; the code fits mean+components on all rows. We reproduce the published protocol as primary and add a train-fitted variant (§6.2 of task).
2. **Training entry point is broken on shipped data** — validation split silently empty then hard error. Any reviewer running the upstream README workflow verbatim hits this.
3. **Probe hyperparameters ambiguous**: `train.py` defaults (lr 0.1, wd 0) vs `run_training.sh` (1e-3/1e-2) vs `run_training_loop.sh` (1e-5/1e-4). Paper states none. We use `train.py` defaults as primary `[UNVERIFIED which produced the paper]` and run a small lr sensitivity check before finalizing.
4. **Notebook provenance is incomplete**: two analysis notebooks are gitignored; committed cells reference private benchmarks and `embeddings_mean_pca_64`. Metric *logic* is recoverable; exact paper-run paths are not.
5. No LICENSE file, no dependency pins, no tests, no seeds beyond probe training (torch/random/np = 42 in `train.py`).

## 5. Reproduction tiers

- **Tier 1 — minimum viable (must ship):** RQ + Qwen3-4B + last-token + PCA-64 (published protocol) + diffMean/logistic/hinge + layer-wise test AUROC + Hanley–McNeil CIs + Fig-1-style plot. Expected: AUROC ≈ 0.8–0.9 mid-layers for trained probes (paper Fig. 1, first column).
- **Tier 2 — core (planned):** mean-pooled comparison; alignment suite (cosine, Spearman, top/bottom-20% Jaccard — Figs. 2-style); leakage-controlled PCA variant; error analysis.
- **Tier 3 — optional (pick ≤2):** ① cross-dataset transfer with SRAQ (public, CC-BY-4.0, `data/{train,dev,test}.jsonl` with `paragraph`/`full_turn` — [Abby-OGV/sraq](https://github.com/Abby-OGV/sraq)); ② second model (Qwen3-8B int8) or pooling variants (last-5/last-10/question-span, App. D); ③ bootstrap probe-stability across 3 seeds; ④ qualitative ranking comparison (§7-style).

## 6. Experiment sequence

1. Validate data (CPU) → 2. pilot extraction (200 examples, all layers, timing/memory logged) → 3. full extraction last-token → 4. PCA-64 (both fit variants) → 5. diffMean + probes per layer → 6. layer-wise AUROC (Tier 1 done) → 7. mean-pooling extraction → 8. alignment metrics → 9. figures/tables → 10. (Tier 3) transfer / robustness.

## 7. Timeline (10 weeks × 10–12 h)

| Week | Focus | Critical? |
|---|---|---|
| 1 | Audit (done), env setup, data validation | Yes |
| 2 | Repo scaffolding, tests for data/pooling/metrics | Yes |
| 3 | Pilot extraction on T4 (Qwen3-4B, 200 ex., Colab) | Yes |
| 4 | Full last-token extraction + artifact validation | Yes |
| 5 | PCA (both variants) + three probes | Yes |
| 6 | Layer-wise AUROC → **Tier 1 complete** | Yes |
| 7 | Mean pooling + alignment metrics → Tier 2 | High |
| 8 | Error analysis, figures, leakage-variant comparison | High |
| 9 | Report + slides | Yes |
| 10 | Reproducibility pass, README, buffer | Yes |

Weeks 7–8 are droppable to "report what exists" without invalidating Tier 1; week 4 can absorb week 3 if GPU quota is scarce (pilot runs inside the full session).

## 8. Deliverables

1. Slides; 2. academic report (with explicit "partial replication at reduced scale" framing and the decodability-vs-alignment distinction); 3. repo: `uv`-managed, tested, config-driven pipeline + figures + result tables.

## 9. Success criteria

Task §13 acceptance criteria (1–12) verbatim, plus scientific gates:

- Tier-1 result lands in the paper's qualitative pattern: trained probes > diffMean; peak AUROC 0.8–0.9 mid-layers; diffMean ≈ 0.6–0.8. Exact values are expected to differ (different model); the *shape* is the replication target.
- Alignment metrics reproduce the qualitative claim (hinge–logistic ≫ diffMean–trained on cosine/Spearman) even if absolute overlaps differ.
- No causal or "full reproduction" claims anywhere in the report.

Failure to match is itself reportable *if* the pipeline is verified correct — see RISK_REGISTER.md #10.
