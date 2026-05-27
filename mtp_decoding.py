"""The heart of the benchmark: load a model and decode with it three ways.

All three modes share ONE runtime (Hugging Face transformers) and ONE KV cache
strategy, so their throughput numbers are directly comparable.

  standard     plain greedy, one token per (cached) target pass.
  speculative  draft proposes a block, the target VERIFIES it in one pass and
               keeps the matching prefix -> output identical to `standard`.
  parallel     draft fills a block that is ACCEPTED without verification.

Why the cache is handled differently per mode
----------------------------------------------
`standard` and `parallel` only ever *append* tokens, so an append-only KV cache
is exactly right and works even on Qwen3.5's hybrid (linear + full attention)
cache. `speculative` must *roll the cache back* whenever a drafted token is
rejected; doing that correctly on a hybrid cache is subtle, so we delegate it to
transformers' built-in assisted generation, which is the same lossless
algorithm. Everything still runs on the same model object.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # VLM fallback class (Qwen3.5 ships as *ForConditionalGeneration)
    from transformers import AutoModelForImageTextToText
except ImportError:  # older transformers
    AutoModelForImageTextToText = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class LoadedModel:
    tokenizer: AutoTokenizer
    target: AutoModelForCausalLM   # the model we actually evaluate
    draft: AutoModelForCausalLM    # cheap proposer standing in for MTP heads
    device: str


def _load_lm(repo: str, torch_dtype, device):
    """Load a causal LM, falling back to the image-text-to-text class for the
    Qwen3.5 *ForConditionalGeneration (VLM) checkpoints. We only ever feed text,
    so either class exposes the same next-token `.logits`."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch_dtype, device_map=device).eval()
    except (ValueError, KeyError) as e:
        if AutoModelForImageTextToText is None:
            raise
        print(f"    {repo}: not a plain causal LM ({e}); loading as VLM.")
        return AutoModelForImageTextToText.from_pretrained(
            repo, torch_dtype=torch_dtype, device_map=device).eval()


def load_model(repo: str, draft_repo: str, dtype: str, device: str) -> LoadedModel:
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = _load_lm(repo, torch_dtype, device)
    draft = _load_lm(draft_repo, torch_dtype, device)
    return LoadedModel(tokenizer, target, draft, device)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
@dataclass
class DecodeResult:
    text: str
    new_tokens: int        # how many tokens we generated
    target_passes: int     # target forward passes (-1 if not tracked)
    seconds: float


def _eos_ids(lm: LoadedModel) -> set:
    """All end-of-sequence ids (Qwen uses several, e.g. <|im_end|>)."""
    ids = set()
    gc = getattr(lm.target, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(e, int):
        ids.add(e)
    elif isinstance(e, (list, tuple)):
        ids.update(e)
    if lm.tokenizer.eos_token_id is not None:
        ids.add(lm.tokenizer.eos_token_id)
    return ids


@torch.no_grad()
def _step(model, input_ids, past):
    """One cached forward pass. Returns (last-position logits, new cache)."""
    out = model(input_ids, past_key_values=past, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def _finish(lm, tokens, eos_ids, passes, t0):
    if any(t in eos_ids for t in tokens):
        cut = next(i for i, t in enumerate(tokens) if t in eos_ids)
        tokens = tokens[:cut + 1]
    text = lm.tokenizer.decode(tokens, skip_special_tokens=True)
    return DecodeResult(text, len(tokens), passes, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Mode 1: standard greedy decoding (append-only KV cache)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_standard(lm: LoadedModel, ids: torch.Tensor, max_new: int) -> DecodeResult:
    eos_ids = _eos_ids(lm)
    t0 = time.perf_counter()
    logits, past = _step(lm.target, ids, None)   # prefill the prompt
    passes = 1
    tokens = []
    for _ in range(max_new):
        tok = int(logits.argmax(-1))
        tokens.append(tok)
        if tok in eos_ids:
            break
        nxt = torch.tensor([[tok]], device=lm.device)
        logits, past = _step(lm.target, nxt, past)
        passes += 1
    return _finish(lm, tokens, eos_ids, passes, t0)


# ---------------------------------------------------------------------------
# Mode 2: verified speculative decoding (lossless) via HF assisted generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_speculative(lm: LoadedModel, ids: torch.Tensor, max_new: int,
                       k: int) -> DecodeResult:
    eos_ids = _eos_ids(lm)
    # Ask the assistant (draft) to propose ~k tokens per verification step.
    try:
        gc = lm.draft.generation_config
        gc.num_assistant_tokens = k
        gc.num_assistant_tokens_schedule = "constant"
    except Exception:
        pass

    t0 = time.perf_counter()
    out = lm.target.generate(
        ids, assistant_model=lm.draft, do_sample=False,
        max_new_tokens=max_new, pad_token_id=lm.tokenizer.pad_token_id)
    dt = time.perf_counter() - t0

    tokens = out[0, ids.shape[1]:].tolist()
    res = _finish(lm, tokens, eos_ids, passes=-1, t0=t0)
    return DecodeResult(res.text, res.new_tokens, -1, dt)  # passes not tracked


# ---------------------------------------------------------------------------
# Mode 3: non-verification parallel decoding (fastest, lossy)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_parallel(lm: LoadedModel, ids: torch.Tensor, max_new: int,
                    k: int) -> DecodeResult:
    """Each block = one strong target token (the anchor / "head 0") followed by
    k-1 cheap draft tokens accepted blindly. Both models keep their own
    append-only cache; the target's is advanced over the whole block at once."""
    eos_ids = _eos_ids(lm)
    t0 = time.perf_counter()

    t_logits, t_past = _step(lm.target, ids, None)   # prefill target
    _, d_past = _step(lm.draft, ids, None)           # prefill draft
    passes = 1
    tokens = []
    stop = False

    while len(tokens) < max_new and not stop:
        anchor = int(t_logits.argmax(-1))            # strong token from target
        tokens.append(anchor)
        block = [anchor]
        if anchor in eos_ids:
            break

        # Draft k-1 look-ahead tokens, fed one at a time into the draft cache.
        cur = anchor
        for _ in range(k - 1):
            d_logits, d_past = _step(
                lm.draft, torch.tensor([[cur]], device=lm.device), d_past)
            cur = int(d_logits.argmax(-1))
            tokens.append(cur)
            block.append(cur)
            if cur in eos_ids or len(tokens) >= max_new:
                stop = True
                break

        # Catch the draft cache up to the last block token it hasn't consumed.
        if not stop:
            _, d_past = _step(
                lm.draft, torch.tensor([[block[-1]]], device=lm.device), d_past)

        # Advance the target cache over the whole accepted block -> next anchor.
        block_t = torch.tensor([block], device=lm.device)
        t_logits, t_past = _step(lm.target, block_t, t_past)
        passes += 1

    return _finish(lm, tokens, eos_ids, passes, t0)


# ---------------------------------------------------------------------------
# Single entry point used by the benchmark driver
# ---------------------------------------------------------------------------
def generate(lm: LoadedModel, prompt_ids: torch.Tensor, mode: str,
             max_new: int, k: int) -> DecodeResult:
    if mode == "standard":
        return decode_standard(lm, prompt_ids, max_new)
    if mode == "speculative":
        return decode_speculative(lm, prompt_ids, max_new, k)
    if mode == "parallel":
        return decode_parallel(lm, prompt_ids, max_new, k)
    raise ValueError(f"unknown decode mode: {mode}")
