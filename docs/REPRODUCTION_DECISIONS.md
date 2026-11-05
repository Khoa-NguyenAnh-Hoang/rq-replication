# REPRODUCTION DECISIONS

Every deviation from the original paper, with justification and its effect on what we may claim. Paper = Yao et al., ACL 2026 (pp. 155–172); upstream = ruyi101/rq-representation-probing (audited 2026-11-01).

## D1. Model: Qwen3-4B (primary), Qwen3-8B int8 (fallback) — instead of Qwen3-32B / Llama-3.3-70B

**Why appropriate:**
- Hardware: 70B/32B models cannot run inference on a free Colab T4 (16 GB) at any precision. Qwen3-4B fp16 ≈ 8.1 GB weights (verified: 36 layers × hidden 2,560, HF config.json), leaving activation headroom.
- Lineage: same Qwen3 family and architecture as the paper's main model (Qwen3-32B), so the *representational-geometry* claims are tested within-family at reduced scale.
- Precedent: the paper itself validates that findings hold for smaller models — Appendix G (p.170, Figs. 13–15) reports Qwen3-8B and Llama-3.1-8B-Instruct with "broadly consistent" trends. Our experiment therefore sits closest to a *published appendix setting*, not an invented one.
- The official repo's own README example uses Qwen3-4B.
- Qwen3-4B is ungated (Apache-2.0): no HF token, no license clickwalls for graders. Llama-3.1-8B-Instruct is gated — rejected for access friction.
- Qwen3-8B exact-fp16 (16.4 GB) does not fit T4; the fallback runs it 8-bit (bitsandbytes), adding quantization noise — acceptable for a fallback, flagged in every table if used.

**Effect on claims:** we can claim replication of the paper's *qualitative pattern* (layer trend, probe ordering, decodability-vs-alignment dissociation) at 4B/8B scale. We **cannot** claim reproduction of the main-text numbers (Qwen3-32B curves) nor anything about 70B-scale representations. Every result line reads "partial replication at reduced scale".

## D2. Datasets: RQ only (Tiers 1–2); SRAQ optional Tier 3

**Why:** RQ ships in the upstream repo with splits matching the paper exactly (verified 3,200/797/1,000; 2,332 rhetorical). SRAQ is *not* in the upstream repo; a legitimate copy exists ([Abby-OGV/sraq](https://github.com/Abby-OGV/sraq), CC-BY-4.0, 384/103/484 JSONL matching paper §3.1) — so Tier 3 transfer needs no scraping, but costs a second full extraction + new PCA spaces.

**Effect:** Tiers 1–2 support all within-dataset claims (RQ1, RQ2, within-dataset RQ3). Cross-dataset claims (Figs. 3–4) are only made if Tier 3 runs. The "rhetoric is not a single direction" conclusion is *weakened but not invalidated* by within-dataset evidence alone, because the paper's within-dataset alignment analysis (Fig. 2) already shows the dissociation.

## D3. PCA fit on all rows (published protocol) kept as primary + train-fitted variant added

**Why both:** paper App. B (p.166) states the PCA mean is "computed on the training split", but the released `pca_transform.py` fits PCA on all 4,997 rows including test. We do not silently change the published procedure (primary = protocol-as-released), and we do not silently inherit the discrepancy (variant = train-fitted, reported side-by-side). Paper App. A already shows PCA-64 vs full-space AUROC deltas < 1 pp, predicting the fit-split choice matters little — M8 measures it.

**Effect:** primary numbers are directly comparable to the paper; leakage discussion is honest. If the delta exceeds CI width, the leakage-controlled result leads and the published-protocol fragility is reported.

## D4. Probe hyperparameters: upstream `train.py` defaults (SGD lr 0.1, wd 0, batch 256, ≤1000 epochs, patience 20, early stop on validation loss) `[UNVERIFIED]`

**Why:** the paper states no probe hyperparameters. Upstream is self-contradictory: `train.py` defaults (0.1/0) vs `run_training.sh` (1e-3/1e-2, leftover private config) vs `run_training_loop.sh` (1e-5/1e-4). The README's canonical workflow passes no overrides → defaults. We adopt defaults, and add a 3-point lr sensitivity check on one layer; if results are lr-sensitive, we report the sensitivity rather than pick the best-looking value.

**Effect:** probe *training* details may differ from the paper's actual runs. AUROC comparisons remain internally valid (all three probes share the protocol); absolute values carry this caveat.

## D5. Test-split discipline for transfer; Jaccard frac = 0.2

**Why:** the committed `proj_transfer.ipynb` computes transfer AUROC on **all rows** (test-split code commented out), and the transfer Jaccard cell uses the function default frac=0.10 while the paper says 20% everywhere (Fig. 2/3 captions). Paper §3.3 states results are on the test split. We resolve both discrepancies **in favor of the paper**.

**Effect:** our transfer numbers (Tier 3) may differ slightly from the paper's; they will be methodologically cleaner. Within-dataset metrics unaffected.

## D6. Extraction engineering: batched fp16, max_length=512, checkpoint/resume, fp32 storage

**Why:** upstream is batch=1, `dtype="auto"` (bf16 — unusable on T4), no truncation, no resume — designed for A100-class hardware and uninterrupted sessions. The math we change nothing about: last-token via attention-mask gather and masked mean pooling are provably identical to upstream at batch=1 (unit-tested against a batch=1 reference). Truncation at 512 with reported statistics guards against pathological lengths (RQ mean ≈ 39 tokens; tail `[UNVERIFIED]` until pilot).

**Effect:** none on representations (identical math, fp32 accumulation); enables the T4 + interruption constraints.

## D7. Exclusions and their cost

| Excluded | Paper ref | Why | Cost |
|---|---|---|---|
| Causal steering | App. F | Needs generation loops + GPT-5.1 scoring (paid API) — out of budget and scope | We make **no causal/controllability claims**; report quotes the paper's own caution that separability ≠ controllability |
| Grassmann subspace geometry | App. C | Analysis sophistication beyond course scope; requires SRAQ | None for within-dataset claims |
| ModernBERT encoder comparison | App. E | Different model class; adds a full pipeline variant | None |
| Pooling grid (last-5/10/question-span) | App. D | Tier 3 optional; mean vs last already covers the paper's main contrast | None if dropped |
| Qualitative Table-1 reproduction | §7 | Needs layer-48-of-32B analog; subjective curation | Report becomes slightly less rich; optional Tier 3 |

## D8. What our results can and cannot support

**Can:** (1) rhetorical intent is linearly decodable from last-token representations at 4B scale, layer-wise; (2) trained probes ≥ diffMean (RQ); (3) mean pooling does not beat last-token; (4) probes with similar AUROC find non-identical directions (within-dataset); (5) consistency of all of the above with the paper's claims at larger scale.

**Cannot:** (1) exact-value reproduction of any main-text figure; (2) claims about 32B/70B representations; (3) cross-dataset generalization (unless Tier 3 runs); (4) causal or controllability statements; (5) "rhetorical questions are represented by multiple directions in LLMs" as a general law — we can only say our 4B experiment is consistent with it.

**Framing rule for the report:** every headline sentence carries "at reduced scale (Qwen3-4B) on the RQ dataset"; the paper is cited for the full-scale claim it makes.

## D9. Items recorded as resolved (updated 2026-11-04 after the first full T4 run)

1. **Versions** — resolved: the Colab run installs from the committed `uv.lock` (torch 2.15.0 CUDA, transformers 5.22.1, sklearn 1.9.1, Python 3.13.15); the lock is the pin, no separate `==` pins needed.
2. **HF revision** — resolved: `1cfa9a7208912126459214e8b04321603b3df60c`, pinned in `configs/experiment.yaml`; verified unchanged 2026-11-01 → 2026-11-04 and embedded in the artifact metadata.
3. **Tokenizer length stats / truncation** — resolved: **0/4,997 rows exceed max_length=512** (Qwen3 tokenizer; mean ≈ 24 tokens on the sample checked) — truncation is a no-op on RQ.
4. **lr sensitivity** — resolved (2026-11-05, Phase 5): logistic probe on one mid layer (18) in the fit-all PCA space, scored on validation — lr 0.1 → dev AUROC 0.889 (best epoch 4); lr 0.01 → 0.901 (epoch 70); lr 0.001 → 0.902 (epoch 490). Spread ≈ 1.3 pp: moderately insensitive, with the upstream default (0.1) weakest on dev. Per D4 the default is kept for primary results and the sensitivity is reported rather than a value re-picked on dev; the decisive comparison is test-split AUROC against Hanley–McNeil CI width in Phase 6.
5. **Batch size** — resolved: 16 on T4, 0.04–0.09 s/example (drift consistent with Colab host throttling), no OOM; full run ≈ 5–7 min compute.
6. **BOS/EOS** — resolved: Qwen3 adds no special tokens by default; mean pooling matches upstream exactly.
7. **Peak VRAM** — not captured in the first run (the notebook measured its own process instead of the extractor subprocess — fixed: the extractor now logs peak VRAM itself; est. < 9 GB from 8.1 GB weights + batch-16 activations).
