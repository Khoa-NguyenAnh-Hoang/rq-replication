# COLAB RUNBOOK — running the repo on Colab GPUs

Researched 2026-11-01. Google ships an **official Colab CLI** (since June 2026) that
provisions GPU runtimes and executes local code from the terminal — exactly what this
repo's Phase 3+ needs. Primary source: [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
and the [announcement](https://developers.googleblog.com/introducing-the-google-colab-cli).

## 0. Platform constraint

**The CLI supports Linux and macOS only — not Windows.** This project is developed on
Windows, so pick one:

- **A (recommended): WSL.** WSL2 is present but only with the `docker-desktop` distro;
  add a real distro once: `wsl --install -d Ubuntu`, then do everything below inside it
  (repo lives in the WSL filesystem, e.g. `~/rq-replication`).
- **B (fallback): browser notebook.** A Colab notebook whose first cell clones the repo
  and runs the same commands — no CLI, fully supported, same result, manual UI clicks.

The CLI consumes the same free-tier compute units as the browser UI (T4 is the free GPU;
`--high-mem` and some accelerators need Colab Pro).

## 1. Install and authenticate (once, in WSL)

```bash
uv tool install google-colab-cli    # or: pip install google-colab-cli
colab usage                          # shows compute-unit balance after auth
```

Auth: default `--auth adc` (Application Default Credentials) or `--auth oauth2` with a
client config at `~/.colab-cli-oauth-config.json` (see the repo README for setup).

## 2. Smoke test (Phase 0 gate) — one-shot session

```bash
colab run --gpu T4 scripts/colab_smoke_test.py --full
```

`colab run` provisions a T4, transmits the file, executes it, retrieves output, tears
down. The `--full` load of Qwen3-4B (~8 GB) happens inside the runtime. The printed
"PIN THESE" block is the Phase 0 deliverable.

## 3. Persistent session workflow (Phase 3+ extraction)

```bash
colab new -s rq --gpu T4                    # provision; name is remembered while it lives
colab ls                                    # verify; sessions die when the kernel idles out

# one-time env setup inside the runtime (repo must be pushed to GitHub):
echo 'git clone https://github.com/<you>/rq-replication.git && cd rq-replication && pip install uv && uv sync' | colab exec -s rq

# pilot then full extraction (outputs land in outputs/ inside the runtime):
colab exec -s rq -f scripts/colab_smoke_test.py     # quick GPU re-check
echo 'cd rq-replication && uv run python -m rq_probe extract --max-examples 200' | colab exec -s rq
echo 'cd rq-replication && uv run python -m rq_probe extract' | colab exec -s rq

# pull artifacts home, then release the session:
colab download -s rq rq-replication/outputs/embeddings ./outputs/embeddings
colab stop -s rq
```

Notes:
- `colab exec -f FILE` transmits a **single file**; for repo code use the git-clone
  bootstrap above (or `colab upload -s rq LOCAL REMOTE` per path).
- Everything CPU-side (PCA, probes, metrics, figures) runs locally on Windows — only
  `extract` needs the GPU session.
- Download artifacts **before** `colab stop` (or mount Drive via `colab drivemount`,
  or `colab run --keep`), else they die with the runtime.
- `colab log -s rq` captures execution logs; `colab status -s rq` checks liveness.
- If a session dies mid-extraction: our extractor resumes from shards — re-provision,
  `colab upload` the partial shards, continue.

## 4. Free-tier fallback chain

1. CLI / browser with free T4 (primary plan, RISK_REGISTER #1).
2. Colab Pro (~$10, within budget) → `colab new --gpu L4` (24 GB) — enables Qwen3-8B bf16 unquantized.
3. Kaggle Notebooks CLI (`kaggle kernels push`, free 2×T4 ~30 h/week) — code-compatible
   fallback if Colab quota is exhausted; needs a kaggle.json token.
