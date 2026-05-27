"""Dataset loading, prompt building and scoring for the four benchmarks.

Every task is treated as a *generative* task (the model writes an answer in
plain text) so that all three decode modes are exercised the same way. Each task
provides three things:

    examples(limit)  -> list of dicts with a "prompt" and the gold answer
    score(example, generated_text) -> 1.0 if correct else 0.0

GSM8K  : math word problems   -> exact-match on the final integer.
HumanEval: Python functions   -> run the official unit tests (pass@1).
MMLU   : 4-way multiple choice -> extract chosen letter.
ARC    : science multiple choice -> extract chosen letter.
"""
from __future__ import annotations

import re

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Prompt formatting shared helper
# ---------------------------------------------------------------------------
def build_chat_prompt(tokenizer, user_text: str) -> str:
    """Use the model's chat template when it has one; else fall back to raw.

    Qwen3.5 is a "thinking" model: by default the template opens a long
    <think>...</think> chain-of-thought that easily exhausts the token budget
    before the final answer. We disable it (`enable_thinking=False`) so the model
    answers directly; the kwarg is ignored by templates that don't support it.
    """
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": user_text}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
    return user_text + "\n"


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _strip_think(text: str) -> str:
    """Remove a <think>...</think> reasoning block (and any unclosed leftover)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r".*</think>", "", text, flags=re.DOTALL)


def _last_number(text: str):
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def _final_number(text: str):
    """Prefer an explicit 'Answer: <n>'; otherwise the last number in the text."""
    text = _strip_think(text)
    m = re.search(r"[Aa]nswer\s*[:\-]?\s*\$?(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    return _last_number(text)


class GSM8K:
    name = "gsm8k"

    def __init__(self, spec, tokenizer):
        self.tok = tokenizer
        self.ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)

    def examples(self, limit):
        out = []
        for row in self.ds.select(range(min(limit, len(self.ds)))):
            q = row["question"]
            gold = row["answer"].split("####")[-1].strip()
            user = (f"Solve the problem. Think step by step, then write the final "
                    f"answer on its own line as 'Answer: <number>'.\n\n{q}")
            out.append({"prompt": build_chat_prompt(self.tok, user), "gold": gold})
        return out

    def score(self, ex, gen):
        pred = _final_number(gen)
        gold = _last_number(ex["gold"])
        return float(pred is not None and gold is not None and pred == gold)


# ---------------------------------------------------------------------------
# HumanEval  (executes generated code -> gated behind allow_code_exec)
# ---------------------------------------------------------------------------
class HumanEval:
    name = "humaneval"

    def __init__(self, spec, tokenizer, allow_code_exec):
        self.tok = tokenizer
        self.allow = allow_code_exec
        self.ds = load_dataset(spec.hf_path, split=spec.split)

    def examples(self, limit):
        out = []
        for row in self.ds.select(range(min(limit, len(self.ds)))):
            user = ("Complete the following Python function. Return ONLY the full "
                    "function definition inside a ```python code block.\n\n"
                    f"```python\n{row['prompt']}```")
            out.append({
                "prompt": build_chat_prompt(self.tok, user),
                "entry_point": row["entry_point"],
                "test": row["test"],
                "func_prompt": row["prompt"],
            })
        return out

    def _extract_code(self, gen, func_prompt):
        m = re.search(r"```(?:python)?\n(.*?)```", gen, re.DOTALL)
        code = m.group(1) if m else gen
        # If the model only emitted the body, prepend the signature.
        if "def " not in code:
            code = func_prompt + code
        return code

    def score(self, ex, gen):
        if not self.allow:
            return None  # skipped -> reported as N/A
        code = self._extract_code(gen, ex["func_prompt"])
        program = code + "\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n"
        return float(_run_in_subprocess(program))


def _run_in_subprocess(program: str, timeout: int = 10) -> bool:
    """Run untrusted generated code in a separate process with a timeout.

    WARNING: this executes model-generated code. Only enabled via
    RunConfig.allow_code_exec / --allow-code-exec. Run inside a container or VM.
    """
    import subprocess
    import sys
    import tempfile
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
            f.write(program)
            f.flush()
            r = subprocess.run([sys.executable, f.name],
                               capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Multiple-choice tasks (MMLU + ARC) -> extract a letter
# ---------------------------------------------------------------------------
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_letter(text: str):
    # Prefer an explicit "Answer: X"; otherwise take the last standalone letter.
    text = _strip_think(text)
    m = re.search(r"[Aa]nswer\s*[:\-]?\s*([A-D])", text)
    if m:
        return m.group(1)
    letters = _LETTER_RE.findall(text)
    return letters[-1] if letters else None


def _format_choices(choices):
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


class MMLU:
    name = "mmlu"

    def __init__(self, spec, tokenizer):
        self.tok = tokenizer
        self.ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)

    def examples(self, limit):
        out = []
        for row in self.ds.select(range(min(limit, len(self.ds)))):
            user = (f"{row['question']}\n\n{_format_choices(row['choices'])}\n\n"
                    "Reply with 'Answer: <letter>'.")
            gold = chr(65 + int(row["answer"]))  # answer is an index 0-3
            out.append({"prompt": build_chat_prompt(self.tok, user), "gold": gold})
        return out

    def score(self, ex, gen):
        return float(_extract_letter(gen) == ex["gold"])


class ARC:
    name = "arc"

    def __init__(self, spec, tokenizer):
        self.tok = tokenizer
        self.ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)

    def examples(self, limit):
        out = []
        for row in self.ds.select(range(min(limit, len(self.ds)))):
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            # ARC labels can be A-D or 1-4; normalise to letters A.. in order.
            user = (f"{row['question']}\n\n{_format_choices(texts)}\n\n"
                    "Reply with 'Answer: <letter>'.")
            gold_idx = labels.index(row["answerKey"])
            gold = chr(65 + gold_idx)
            out.append({"prompt": build_chat_prompt(self.tok, user), "gold": gold})
        return out

    def score(self, ex, gen):
        return float(_extract_letter(gen) == ex["gold"])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_task(spec, tokenizer, allow_code_exec):
    if spec.name == "gsm8k":
        return GSM8K(spec, tokenizer)
    if spec.name == "humaneval":
        return HumanEval(spec, tokenizer, allow_code_exec)
    if spec.name == "mmlu":
        return MMLU(spec, tokenizer)
    if spec.name == "arc":
        return ARC(spec, tokenizer)
    raise ValueError(f"unknown task: {spec.name}")
