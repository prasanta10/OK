# MTP Decoding Benchmark

Benchmarks **multi-token-prediction (MTP)** models under **three decoding modes**
across **four datasets**, reporting **accuracy** and **throughput**.

| | |
|---|---|
| **Models** | Two openly-loadable MTP-capable models (Qwen3.5-9B + Qwen3.5-4B by default) |
| **Modes** | `standard` (1 token/pass) · `speculative` (propose K, verify, lossless) · `parallel` (propose K, accept all, lossy) |
| **Datasets** | GSM8K (math) · HumanEval (code) · MMLU (knowledge) · ARC-Challenge (science) |
| **Metrics** | task accuracy · generated tokens/sec on the target model |

## What this measures

The study's thesis: removing or relaxing the verification layer speeds up
generation, but **non-verified parallel decoding eventually loses logical
coherence**. So you should expect:

- `standard` and `speculative` → **identical accuracy** (speculative is lossless),
  with `speculative` faster.
- `parallel` → **higher throughput but lower accuracy**, and the drop is largest
  on multi-step reasoning (GSM8K, ARC) where early errors compound.

## Setup

```bash
conda env create -f environment.yml
conda activate mtp-bench
huggingface-cli login          # needed if any chosen repo is gated
```

If `nvidia-smi` shows a CUDA version other than 12.4, edit the `cu124` index URL
in [environment.yml](environment.yml).

## Run

```bash
# fast smoke test
python benchmark.py --limit 10

# include HumanEval code execution (RUN IN A CONTAINER/VM — see warning below)
python benchmark.py --limit 100 --allow-code-exec

# a single slice of the matrix
python benchmark.py --models Qwen3.5-9B --tasks gsm8k --modes standard parallel
```

Results print as a table and are saved to `results/results.json`.

## File map

| File | Role |
|---|---|
| [config.py](config.py) | All knobs: models, datasets, modes, limits |
| [mtp_decoding.py](mtp_decoding.py) | Model loading + the three decode loops |
| [datasets_tasks.py](datasets_tasks.py) | Prompts + answer extraction + scoring |
| [benchmark.py](benchmark.py) | Driver: sweeps the matrix, reports metrics |

## Important caveats (read before citing numbers)

1. **Substituted models.** The study named `facebook/multi-token-prediction`
   (gated, custom non-transformers code) and `unsloth/Qwen3.5-9B-MTP-GGUF` (GGUF
   for llama.cpp). To stay license-free and run in plain Python, we substitute
   transformers-format Qwen3.5 models. **Verify the exact repo names** in
   [config.py](config.py) with `huggingface-cli`; edit if they differ.

2. **Emulated MTP heads.** A real MTP model proposes the next K tokens from a
   *single* forward pass via its extra heads. We emulate that with a small
   **draft model** (`DraftProposer` in [mtp_decoding.py](mtp_decoding.py)). The
   three decode loops are written against a generic `propose(ids, k)` call, so to
   run the *real* thing you only reimplement that one method to read your model's
   native MTP heads — the loops don't change.

3. **No KV cache.** The decode loops use full forward passes for readability.
   Absolute tokens/sec is therefore lower than production; the **relative**
   speedup between modes is what's meaningful. For faithful absolute throughput,
   serve the model with vLLM (`speculative_config` with `method="mtp"`).

4. **Code execution risk.** `--allow-code-exec` runs model-generated Python to
   score HumanEval. It runs each candidate in a subprocess with a 10s timeout,
   but that is **not a security sandbox** — only enable it inside a container/VM.

5. **Small `--limit` by default.** Bump `--limit` for statistically meaningful
   accuracy; the default (50) is for quick iteration.
