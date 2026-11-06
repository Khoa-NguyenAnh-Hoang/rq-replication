# EXPERIMENT PROTOCOL

Scientific protocol for the reduced-scope replication. Paper refs: sections/pages of [2026.acl-long.5](https://aclanthology.org/2026.acl-long.5.pdf); code refs: files of ruyi101/rq-representation-probing @ default branch (audited 2026-11-01). Untested choices are marked `[UNVERIFIED]`.

## 1. Dataset

**RQ** (Zhuang & Riloff, ACL SRW 2020), distributed verbatim in the official repo (`data/RQ.csv`).

Verified statistics (computed from the shipped CSV, 2026-11-01; canonical sha256 `316ccc067eea564b96ea90ef23a09cc769efb922bddce1255c6d968ae7a75236` — the upstream git blob as served on Linux; a Windows checkout with autocrlf hashes differently while content stays identical, re-verified by `rq-probe validate` on every load):

| Split | Rows | Rhetorical (1) | Informational (0) |
|---|---|---|---|
| train | 3,200 | 1,508 | 1,692 |
| dev (→ validation) | 797 | 367 | 430 |
| test | 1,000 | 457 | 543 |
| **total** | **4,997** | **2,332** | **2,665** |

Matches paper §3.1 p.156 ("4,997 instances, 2,332 rhetorical, 3,200/797/1,000"). Columns: `context, question, label, dataset, id, question_with_context, binary_label`. Verified invariants: `label` ≡ `binary_label`; `id` = 0…4996 sequential = row order; no empty `question_with_context`. Text formulation: `question_with_context` (prior tweet + target question concatenated), paper main setting. Mean length 38.9 Llama-3.3-70B tokens (paper; Qwen3 tokenizer count to be reported at M3).

**SRAQ** (Ikumariegbe et al., EMNLP 2025) — Tier 3 only. Legitimately available: [github.com/Abby-OGV/sraq](https://github.com/Abby-OGV/sraq), `data/{train,dev,test}.jsonl` (384/103/484 — matches paper), fields `paragraph` (paper's main formulation) and `full_turn`, license CC-BY-4.0. No scraping involved. Project must remain runnable without it.

**Data integrity checks (run before every stage, `rq-probe validate`):** schema; unique sequential ids; split mapping {dev,val,validation}→validation; label ∈ {0,1}; no missing text; per-split class distribution recorded; embeddings joined to labels **by id, never by assumed order**; truncation statistics (n > max_length) reported after tokenization.

## 2. Model

| | Primary | Fallback |
|---|---|---|
| Checkpoint | `Qwen/Qwen3-4B` @ revision `1cfa9a7208912126459214e8b04321603b3df60c` (pinned 2026-11-04; stable since the 2026-11-01 audit) | `Qwen/Qwen3-8B` (+ 8-bit loading) |
| Why | Official README's own example model; same family as paper's Qwen3-32B; ungated, Apache-2.0 | Exact paper appendix model (Figs. 13–15 report Qwen3-8B on RQ/last-token) |
| Config (from HF config.json, 2026-11-01) | 36 layers, hidden 2,560, bf16 checkpoint | 36 layers, hidden 4,096 |
| fp16 weights ≈ | 8.1 GB → fits T4-16G with activation headroom | 16.4 GB fp16 → does NOT fit; int8 ≈ 8.5 GB via bitsandbytes |
| Deviation from paper | Smaller model (paper main: 32B/70B; appendix: 8B) | Quantized activations ≠ paper's precision `[UNVERIFIED]` effect on probe AUROC |

Excluded: `Llama-3.1-8B-Instruct` (gated — HF token + license acceptance; access risk for graders), `gpt-oss-20b` (paper App. G reports unstable results for it), Qwen3-32B / Llama-3.3-70B (do not fit T4). If budget allows (~$10–20), Colab Pro L4 (24 GB) runs Qwen3-8B in bf16 unquantized as a stretch option.

## 3. Representation extraction

Paper §3.2 p.157. For each example, one forward pass with `output_hidden_states=True`; hidden states = tuple of L+1 = 37 tensors (embedding output + 36 layers).

- **last_token** (primary): hidden state of the final *valid* token. Upstream uses `h[0,-1,:]` — safe at batch=1 only. We batch, so per-example index = `attention_mask.sum(dim=1) − 1` (verified by test vs batch=1 reference).
- **mean_token** (Tier 2): Σ h_t·m_t / Σ m_t with attention mask m (upstream mean over all tokens at batch=1 ≡ unmasked mean; identical there, ours correct when batched). Special tokens included, matching upstream tokenizer defaults.
- Precision: fp16 weights, fp32 accumulation of saved embeddings; upstream saved bf16 and upcast later — we save fp32 directly (≈ 1.9 GB per artifact for Qwen3-4B: 4,997 × 37 × 2,560 × 4 B).
- Truncation: `max_length=512` with statistics reported (upstream had none). Measured on the real run (2026-11-04): **0/4,997 rows truncated** — max_length is a no-op on RQ with the Qwen3 tokenizer.
- Loop requirements (task §7): `model.eval()`, `torch.inference_mode()`, no grad, configurable batch/device/dtype, shard checkpoint every 500 examples + resume, progress log, ordering = CSV id order.
- CPU vs GPU: extraction = GPU — **measured on a free Colab T4 (2026-11-04)**: batch 16 fp16, 0.04–0.09 s/example, full 4,997-example run ≈ 5–7 min compute + ~8 GB model download once per runtime; peak VRAM not captured in that first run (est. < 9 GB; the extractor now self-reports). PCA = CPU (sklearn, minutes); probes = CPU; metrics = CPU.

## 4. PCA procedure

Paper §3.2 + App. A (p.166): k=64, per dataset×model×formulation, all probes in the same projected space per setting.

- Implementation: sklearn `PCA(n_components=64)` per layer, float32, CPU (port of `pca_transform.py`), with one deliberate engineering deviation: **`svd_solver="full"`** — sklearn's default auto-selects randomized SVD at this scale, which is nondeterministic without a fixed seed; the exact solver is deterministic, seed-free, and numerically equivalent up to SVD precision. Re-runs are bit-identical (unit-tested).
- **Fit split — the known contradiction:** paper App. B p.166 says the PCA mean µ is "computed on the training split", but `pca_transform.py` fits PCA on **all N rows (train+dev+test together)**. We run BOTH:
  - `fit_split: all` — published protocol, primary results (documented as containing test-set information in the basis; implication assessed in the report);
  - `fit_split: train` — leakage-controlled variant, reported side-by-side. If AUROC deltas are ≪ CI width (App. A found <1 pp for full-space vs PCA-64), conclusions stand under both.
- **First real-data run (2026-11-04, Qwen3-4B last-token artifact, 8-core CPU):** 64th-component explained variance min/median/max = **0.13% / 0.21% / 0.23%** across all 37 layers (both fit variants) — consistent with paper App. A (< 1%). The 64 PCs capture ~64% of total variance (median layer). All-vs-train mean divergence is small: relative ‖mean_all − mean_train‖ / ‖mean_all‖ median 0.84%, max 1.32%. The decisive comparison is probe AUROC under both fits (Phase 6).
- Basis saved (components, means, explained_variance_ratio) for map-back; layer ℓ of the artifact uses layer ℓ of the basis only — no cross-layer or cross-basis mixing (unit-tested).

## 5. Probes

Paper §3.3. All fit on **train split only** in PCA-64 space; hyperparameters from upstream `train.py` defaults `[UNVERIFIED]` (see REPRODUCTION_DECISIONS #4):

| Probe | Objective | Upstream code (reused) | Our protocol |
|---|---|---|---|
| diffMean | w = µ⁺ − µ⁻ (class means, Eq. 2) | `diffmean_analysis.ipynb` | train split; unit-normalized before projection (AUROC-invariant, kept for fidelity) |
| logistic | BCEWithLogits, linear, no bias | `model.py`, `trainer.py`, `utils.py` | SGD lr 0.1, wd 0, batch 256, ≤1000 epochs, early stop patience 20 on validation loss, best state restored, seeds 42 |
| hinge | mean clamp(1 − y·logit), y∈{−1,1} | `utils.hinge_loss` | same trainer/protocol |

- **Validation split:** used only for early stopping/model selection (upstream intent; broken upstream due to `dev` naming — fixed).
- **Test split:** touched only by final evaluation; no threshold tuning on it (AUROC is threshold-free anyway).
- Known upstream quirk kept deliberately: probes have **no bias term** although App. B's map-back math includes b — b≡0 upstream, bias-free scoring changes neither AUROC nor rankings (affine shift), and map-back reduces to `w_x = Wᵀw_z`, `b_x = −w_xᵀµ`. Documented, unit-tested.
- lr sensitivity check (3 values × 1 layer) before the full sweep; recorded in REPRODUCTION_DECISIONS. Measured (2026-11-05, logistic, layer 18, fit-all): dev AUROC 0.889 / 0.901 / 0.902 for lr 0.1 / 0.01 / 0.001 — spread ≈ 1.3 pp, moderately insensitive; upstream default kept.

## 6. Evaluation metrics

Paper §3.4. Split discipline: per-layer metrics on **test** split (paper §3.3: "all results below are reported on the test split"); train/dev numbers also logged for diagnosis, clearly labeled.

- **AUROC** (primary): sklearn `roc_auc_score` on probe-score projections, per layer. 95% CI: Hanley–McNeil analytic formula (exact port of upstream `auc_confint_hanley_mcneil`, z=1.96). Tier 3 adds bootstrap CI over examples (resampling unit = test examples, B=2000).
- **Cosine similarity** between probe directions: unit-normalized vectors, per layer, PCA space (matches upstream `within_dataset_cosine.ipynb`); transfer cosine in original space after map-back (matches `map_back.ipynb` + Fig. 4).
- **Spearman ρ** between two probes' test-split scores (raw projections, PCA space); bootstrap CI n=2000, seed=0, unit=examples (port of `spearman_ci`).
- **Jaccard overlap** of top-20% and bottom-20% sets by each probe's scores on test split, `argpartition`, frac=0.2 (paper Fig. 2 caption; upstream transfer notebook's frac=0.1 is treated as a bug).
- Metric-space table (what runs where — verified from upstream):

| Metric | Space | Scores used | Split |
|---|---|---|---|
| AUROC | PCA-64 | projection on unit direction | test (+train/dev logged) |
| cosine | PCA-64 (within) / original (transfer) | normalized W vectors | all layers |
| Spearman | PCA-64 (within) / original (transfer) | raw projection scores | test |
| Jaccard | PCA-64 (within) / original (transfer) | raw projection scores | test, top/bottom 20% |

- **First real-data run (2026-11-07, fit-all + fit-train, test n=1,000): M6 gate passed.** Trained probes mid-layer (13–25) mean test AUROC **0.825** (logistic peak 0.867 [0.844, 0.890] @ L28; hinge peak 0.860), diffMean lower throughout (mid-layer 0.771, peak 0.799), early-layer signal present (L0 ≈ 0.63–0.64). The paper Fig. 1 col. 1 pattern is replicated qualitatively at 4B scale.
- **Negative control** (shuffled train labels → logistic retrained per layer → untouched real test labels): AUROC min/mean/max = 0.292/0.556/0.693 — near chance, far below any trained probe. The small positive offset vs 0.5 has a known mechanism: early stopping selects on the *real* dev labels, so epoch choice favours noise directions aligned with true signal; a fully blind control would shuffle dev labels too (not run).
- **PCA fit-split comparison:** fit-all vs fit-train test-AUROC deltas are ≤ 0.015 per (probe, layer) — immaterial, as predicted by paper App. A and D3.

## 7. Leakage checks (task §6.2)

1. PCA fit split recorded in every artifact; both variants reported (§4).
2. diffMean + probes: train split only — asserted in code (no all-split means).
3. Early stopping/selection: validation loss only.
4. Test set: read exactly once per (probe, layer) at evaluation; no test-derived statistics fed back anywhere.
5. Join-by-id between CSV rows, embedding rows, and labels — never positional assumption without check (upstream checked max id; we keep that test).
6. Transfer protocol (Tier 3): source direction trained on source-train, mapped back to original space (App. B), applied to target embeddings; scored on target **test** split (upstream committed cell used all rows — corrected).

## 8. Expected outputs

1. Dataset statistics table (§1). 2. Embedding artifacts + metadata. 3. Layer-wise AUROC table (3 probes × 37 layers, with CIs). 4. Fig-1-style AUROC×layer plot. 5. Probe comparison table (RQ2). 6. Alignment panel: cosine / Spearman / Jaccard curves (RQ3). 7. Leakage-variant comparison table. 8. Error analysis: agreement/disagreement examples, length statistics (§7-style, Tier 2/3).

## 9. Interpretation limits (mandatory in the report)

A linear probe establishes **decodability**, not causal use. The paper itself separates separability (Fig. 1) from alignment (Figs. 2–4) and from causal steering (App. F — excluded here). Our report claims only: (i) linear decodability at reduced scale, (ii) (dis)agreement among probes, (iii) consistency with the paper's qualitative pattern. No "LLMs understand rhetorical questions" phrasing. Cross-model claim ("rhetoric is heterogeneously encoded") is reported as *consistent-with*, not *demonstrated-by*, our 4B single-model experiment.

## 10. Deviations from the paper (summary — full rationale in REPRODUCTION_DECISIONS.md)

1. Model scale: Qwen3-4B (or 8B int8) instead of 32B/70B — hardware.
2. Datasets: RQ only in Tiers 1–2 — SRAQ access/effort; Tier 3 adds it (it is publicly available, CC-BY-4.0).
3. max_length=512 truncation added (upstream: none).
4. PCA fit-split variant added (paper-protocol run kept as primary).
5. Transfer evaluation restricted to test split (correcting upstream notebook state toward the paper's stated protocol).
6. Excluded: steering (App. F), Grassmann geometry (App. C), ModernBERT (App. E), pooling grid (App. D — optional Tier 3).
7. Probe hyperparameters: upstream code defaults `[UNVERIFIED]` which produced the paper figures.
