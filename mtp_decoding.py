"""The heart of the benchmark: load an MTP model and decode with it three ways.

Read this top-to-bottom — it is deliberately written as plain greedy loops with
no KV-cache tricks so the difference between the three modes is obvious.

The key idea
------------
A multi-token-prediction (MTP) model can cheaply guess the next K tokens at once
(its "heads"). We capture that ability behind ONE function:

    propose(ids, k) -> list of k guessed token ids

The three execution modes differ only in what they do with those guesses:

  standard     ignore extra heads, take 1 token per target pass.
  speculative  take k guesses, VERIFY with one target pass, keep matching prefix.
  parallel     take k guesses, accept ALL of them, never verify.

Substitution note
------------------
We emulate the MTP heads with a small, fast *draft* model (DraftProposer). To run
the real thing instead, implement `Proposer.propose` to read your model's native
MTP heads (one forward, k logits) and pass that proposer in — the decode loops
below do not change at all.
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
# The proposer (stand-in for MTP heads)
# ---------------------------------------------------------------------------
class DraftProposer:
    """Guesses the next k tokens by running the small draft model k times.

    This plays the role of an MTP model's K extra heads. A real MTP model would
    return all k guesses from a *single* forward pass; functionally the decode
    loops treat both the same way.
    """

    def __init__(self, draft_model, device):
        self.model = draft_model
        self.device = device

    @torch.no_grad()
    def propose(self, ids: torch.Tensor, k: int) -> list[int]:
        guesses = []
        cur = ids
        for _ in range(k):
            logits = self.model(cur).logits[:, -1, :]
            tok = int(logits.argmax(-1))
            guesses.append(tok)
            cur = torch.cat([cur, torch.tensor([[tok]], device=self.device)], dim=1)
        return guesses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _target_next_token(target, ids: torch.Tensor) -> int:
    """Greedy next token according to the *target* model."""
    logits = target(ids).logits[:, -1, :]
    return int(logits.argmax(-1))


def _cat(ids: torch.Tensor, tok: int, device) -> torch.Tensor:
    return torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)


@dataclass
class DecodeResult:
    text: str
    new_tokens: int        # how many tokens we generated
    target_passes: int     # forward passes through the expensive target model
    seconds: float


# ---------------------------------------------------------------------------
# Mode 1: standard greedy decoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_standard(lm: LoadedModel, ids: torch.Tensor, max_new: int) -> DecodeResult:
    eos = lm.tokenizer.eos_token_id
    start_len = ids.shape[1]
    passes = 0
    t0 = time.perf_counter()
    for _ in range(max_new):
        tok = _target_next_token(lm.target, ids)
        passes += 1
        ids = _cat(ids, tok, lm.device)
        if tok == eos:
            break
    dt = time.perf_counter() - t0
    new = ids.shape[1] - start_len
    text = lm.tokenizer.decode(ids[0, start_len:], skip_special_tokens=True)
    return DecodeResult(text, new, passes, dt)


# ---------------------------------------------------------------------------
# Mode 2: verified speculative decoding (lossless, faster)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_speculative(lm: LoadedModel, proposer: DraftProposer,
                       ids: torch.Tensor, max_new: int, k: int) -> DecodeResult:
    eos = lm.tokenizer.eos_token_id
    start_len = ids.shape[1]
    passes = 0
    t0 = time.perf_counter()

    while ids.shape[1] - start_len < max_new:
        # 1) draft proposes k tokens.
        draft_toks = proposer.propose(ids, k)
        candidate = ids
        for t in draft_toks:
            candidate = _cat(candidate, t, lm.device)

        # 2) ONE target pass scores every draft position in parallel.
        logits = lm.target(candidate).logits  # (1, len, vocab)
        passes += 1
        # Target's own greedy token at each draft position, PLUS the token that
        # follows the last draft (k+1 predictions total). The extra one is the
        # free "bonus" token used when every draft is accepted.
        anchor = ids.shape[1] - 1
        target_preds = logits[0, anchor:anchor + k + 1, :].argmax(-1).tolist()

        # 3) accept the longest prefix where draft agrees with target.
        accepted = 0
        for d, t in zip(draft_toks, target_preds[:k]):
            if d == t:
                accepted += 1
            else:
                break

        for t in draft_toks[:accepted]:
            ids = _cat(ids, t, lm.device)
        # 4) target_preds[accepted] is the correction at the first mismatch, or
        #    -- when all k were accepted -- the genuine next token. Always free.
        ids = _cat(ids, target_preds[accepted], lm.device)

        if eos in ids[0, start_len:].tolist():
            break

    dt = time.perf_counter() - t0
    # trim anything generated past an eos
    seq = ids[0, start_len:].tolist()
    if eos in seq:
        seq = seq[:seq.index(eos) + 1]
    new = len(seq)
    text = lm.tokenizer.decode(seq, skip_special_tokens=True)
    return DecodeResult(text, new, passes, dt)


# ---------------------------------------------------------------------------
# Mode 3: non-verification parallel decoding (fastest, lossy)
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_parallel(lm: LoadedModel, proposer: DraftProposer,
                    ids: torch.Tensor, max_new: int, k: int) -> DecodeResult:
    eos = lm.tokenizer.eos_token_id
    start_len = ids.shape[1]
    passes = 0
    t0 = time.perf_counter()

    while ids.shape[1] - start_len < max_new:
        # Anchor token from the strong target (this is "head 0"); one pass.
        anchor_tok = _target_next_token(lm.target, ids)
        passes += 1
        ids = _cat(ids, anchor_tok, lm.device)
        if anchor_tok == eos:
            break

        # Fill the rest of the block from the cheap heads -- accepted blindly.
        guesses = proposer.propose(ids, k - 1)
        stop = False
        for t in guesses:
            ids = _cat(ids, t, lm.device)
            if t == eos:
                stop = True
                break
        if stop:
            break

    dt = time.perf_counter() - t0
    seq = ids[0, start_len:].tolist()
    if eos in seq:
        seq = seq[:seq.index(eos) + 1]
    new = len(seq)
    text = lm.tokenizer.decode(seq, skip_special_tokens=True)
    return DecodeResult(text, new, passes, dt)


# ---------------------------------------------------------------------------
# Single entry point used by the benchmark driver
# ---------------------------------------------------------------------------
def generate(lm: LoadedModel, proposer: DraftProposer, prompt_ids: torch.Tensor,
             mode: str, max_new: int, k: int) -> DecodeResult:
    if mode == "standard":
        return decode_standard(lm, prompt_ids, max_new)
    if mode == "speculative":
        return decode_speculative(lm, proposer, prompt_ids, max_new, k)
    if mode == "parallel":
        return decode_parallel(lm, proposer, prompt_ids, max_new, k)
    raise ValueError(f"unknown decode mode: {mode}")
