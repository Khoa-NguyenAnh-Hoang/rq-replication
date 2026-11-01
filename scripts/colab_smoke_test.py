"""Phase-0 smoke test for Colab (task M0 acceptance: pin versions + revision).

Run on a GPU-enabled Colab runtime after `uv sync`:

    !uv run python scripts/colab_smoke_test.py          # fast (~1 min, tokenizer only)
    !uv run python scripts/colab_smoke_test.py --full   # + real Qwen3-4B fp16 load & forward (~8 GB download)

Prints a "PIN THESE" block at the end: the numbers to copy into pyproject.toml
and docs/EXPERIMENT_PROTOCOL.md. Validates the extraction path (hidden-states
tuple length, last-token gather) on a tiny random-config Qwen3 model, so the
fast run needs no large download.
"""

import argparse
import platform
import sys
import time

PIN_MODEL = "Qwen/Qwen3-4B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also load real Qwen3-4B fp16 and measure VRAM")
    args = parser.parse_args()

    print(f"python : {platform.python_version()}  ({sys.executable})")

    import torch
    import transformers

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"torch    : {torch.__version__}  (cuda build: {torch.version.cuda})")
    print(f"cuda     : available={cuda}", end="")
    if cuda:
        print(f"  device={torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}  bf16={torch.cuda.is_bf16_supported()}")
    else:
        print()
    print(f"transformers: {transformers.__version__}")

    import sklearn
    import scipy
    import numpy
    import pandas

    print(f"numpy {numpy.__version__} | pandas {pandas.__version__} | sklearn {sklearn.__version__} | scipy {scipy.__version__}")

    if not cuda:
        print("\nWARNING: no GPU — smoke test continues on CPU, but pin decisions need a T4 session.")

    from huggingface_hub import HfApi

    revision = HfApi().model_info(PIN_MODEL).sha
    print(f"\n{PIN_MODEL} latest revision: {revision}")

    # --- tokenizer (small download) ---
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(PIN_MODEL)
    print(f"tokenizer loaded in {time.time() - t0:.1f}s  vocab={len(tok)}")
    sample = "Well we cleary were not gonna have an undefeated season .. @user yea the rams Finna last huh ?"
    n_tok = len(tok(sample)["input_ids"])
    print(f"sample RQ row -> {n_tok} tokens (adds_special: adds={tok(sample, add_special_tokens=True)['input_ids'] != tok(sample, add_special_tokens=False)['input_ids']})")

    # --- extraction-path check on a tiny random-config Qwen3 (no download) ---
    from transformers import AutoModelForCausalLM, Qwen3Config

    tiny_cfg = Qwen3Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    tiny = AutoModelForCausalLM.from_config(tiny_cfg).to(device=device, dtype=torch.float16).eval()
    batch = tok([sample, sample + " more context text"], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        out = tiny(**batch, output_hidden_states=True, use_cache=False)
    n_layers = tiny_cfg.num_hidden_layers
    assert len(out.hidden_states) == n_layers + 1, f"expected {n_layers + 1} hidden states, got {len(out.hidden_states)}"
    seq_len = batch["input_ids"].shape[1]
    last_idx = batch["attention_mask"].sum(dim=1) - 1
    gathered = out.hidden_states[-1][torch.arange(2, device=device), last_idx]
    assert gathered.shape == (2, tiny_cfg.hidden_size)
    assert torch.isfinite(gathered).all(), "non-finite values in hidden states"
    print(f"extraction path OK: {n_layers + 1} hidden states, per-example last-token gather shape {tuple(gathered.shape)} (padded seq_len={seq_len})")

    # --- optional real-weight check ---
    if args.full:
        print(f"\nloading {PIN_MODEL} fp16 (~8 GB download, first run only)...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(PIN_MODEL, dtype=torch.float16).to(device)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        load_s = time.time() - t0
        inputs = tok([sample], return_tensors="pt").to(device)
        t0 = time.time()
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        fwd_s = time.time() - t0
        hs = out.hidden_states
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        print(f"full model OK: load {load_s:.0f}s, forward {fwd_s:.2f}s, {len(hs)} hidden states, dtype={hs[0].dtype}, peak VRAM {peak_gb:.1f} GB")
        del model, out, hs

    print("\n--- PIN THESE ---")
    print(f"python        = {platform.python_version()}")
    print(f"torch         = {torch.__version__}")
    print(f"transformers  = {transformers.__version__}")
    print(f"sklearn       = {sklearn.__version__}")
    print(f"model revision= {revision}")
    print("-----------------")
    print("Copy into pyproject.toml (exact pins) and docs/EXPERIMENT_PROTOCOL.md §2.")
    print("Colab preinstalled-torch note: if you used Colab's torch instead of uv-synced,")
    print("record that too — see docs/RISK_REGISTER.md #7.")


if __name__ == "__main__":
    main()
