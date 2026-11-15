# REPRODUCTION CHECKLIST — Phase 10 (M10) pass

One pass on a fresh machine/runtime proving the repo reproduces end-to-end from the
committed code alone. Check items off; fix gaps before freezing the submission.
Expected numbers come from docs/EXPERIMENT_PROTOCOL.md §6 (recorded real runs,
2026-11-04 … 2026-11-08). Anything Colab-dependent is marked `[UNVERIFIED]` until
that step is executed on a live runtime.

## A. Fresh local environment (CPU)

- [ ] Fresh clone into an empty directory: `git clone https://github.com/Khoa-NguyenAnh-Hoang/rq-replication.git && cd rq-replication`
- [ ] `uv sync` — no resolution step, installs from the committed `uv.lock` (warns if the lock is out of date)
- [ ] `uv run pytest` — **76 passed** (CPU-only, no data/raw needed except one integration test that skips gracefully)
- [ ] `uv run rq-probe validate --file data/raw/RQ.csv` (after downloading the CSV per README) — sha256 `316ccc067eea564b96ea90ef23a09cc769efb922bddce1255c6d968ae7a75236`, splits 3,200/797/1,000, rhetorical 2,332

## B. GPU pipeline on a fresh Colab T4 `[UNVERIFIED until run]`

- [ ] Open `notebooks/colab_phase3_extraction.ipynb`, T4 runtime
- [ ] For a Tier-1 last-token reproduction: set **`POOLING = "last_token"`** in cell 1 (the committed default is the Tier-2 `mean_token` run); for the Tier-2 artifact leave it
- [ ] Cells 1–4: GPU check → locked env → CSV checksum gate → smoke test prints the PIN block
- [ ] Cell 6 (full extraction, skip the pilot): ≈ 5–7 min compute at batch 16 fp16, 0/4,997 truncated
- [ ] Cell 7: artifact 4,997×37×2560 fp32, ids contract, all finite; cell 8 → Drive → download
- [ ] Same session or a second run with the other `POOLING` value for both artifacts

## C. CPU downstream on the fresh artifacts

- [ ] `uv run rq-probe pca --embedding outputs/embeddings/RQ_…_{last_token,mean_token}.pt` (both `--fit-split` values for last_token) — PCA is bit-identical on rerun (`svd_solver="full"`); 64th-component evr max ≪ 1% (recorded: 0.23–0.27%)
- [ ] `uv run rq-probe train` (+ `--fit-split train`): with the same seed (42) the probe artifacts are reproducible — test AUROC tables must match the recorded values within float noise (mid-layer means ≈ 0.825 last-token fit-all; mean-token ≈ 0.582–0.632)
- [ ] `uv run rq-probe evaluate` (+ `--fit-split train`, + `--pca …_mean_token_…`): **M6 gate** — trained probes ≈ 0.8–0.9 mid-layers, diffMean lower; negative control `--negative-control` ≈ 0.556 mean
- [ ] `uv run rq-probe analyze` (+ `--pca …_mean_token_…`): **M7a gate** — cosine logistic+hinge ≈ 0.92–0.98 ≫ diffMean pairs; top-20% Jaccard < bottom-20%; map-back |ΔAUROC| = 0
- [ ] `uv run rq-probe robustness`: length ruled out (word-count AUROC ≈ 0.58 ≪ 0.87; tercile AUROC flat); seed std caveat as recorded
- [ ] `uv run python scripts/make_report_assets.py` — `report/assets/` regenerates incl. `numbers.json`

## D. Freeze & submit

- [ ] `uv lock --check` (or `uv sync --locked`) — lock is in sync with `pyproject.toml`
- [ ] Docs updated with the pass results (README phase table row 10; PROTOCOL §6 one line)
- [ ] `git status` clean; artifacts remain in gitignored `outputs/`; no large files committed
- [ ] Course task §13 acceptance criteria 1–12 each demonstrably true `[UNVERIFIED — map to the task sheet]`
- [ ] Tag the submission commit (e.g. `git tag submission && git push --tags`)

## Known-good reference numbers (from the recorded runs)

| Quantity | Recorded value | Where it comes from |
|---|---|---|
| last-token mid-layer (13–25) mean AUROC | logistic/hinge 0.825, diffMean 0.771 | evaluate JSONL |
| logistic peak | 0.867 [0.844, 0.890] @ L28 | evaluate JSONL |
| shuffled-label control | 0.292/0.556/0.693 (min/mean/max) | evaluate --negative-control |
| cosine logistic+hinge | 0.922 (last) / 0.976 (mean) | analyze JSONL |
| fit-split max delta | ≤ 0.015 | evaluate both fits |
| mean-vs-last mid-layer Δ | −0.18 … −0.24 | pooling comparison |
| word-count-only AUROC | 0.576 (terciles 0.855/0.888/0.844) | robustness |
| seed std (across 42/1/2) | mean 0.033, max 0.133 (caveat) | robustness --seed-stability |
