"""Central configuration for the MTP decoding benchmark.

Everything you might want to tweak (which models, which datasets, how many
tokens, how many examples) lives here so the rest of the code stays readable.
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# The original study named:
#   - facebook/multi-token-prediction (7B)   -> gated + custom (non-transformers) code
#   - unsloth/Qwen3.5-9B-MTP-GGUF             -> GGUF (llama.cpp), not a transformers model
#
# We substitute two openly-loadable, transformers-format models so the whole
# benchmark runs without license gates or a separate llama.cpp build. They are
# the same architecture family on purpose: a shared tokenizer lets a single tiny
# "draft" model stand in for the MTP heads for BOTH targets.
#
# >>> Verify the exact repo names for your date with `huggingface-cli search`,
#     and edit these strings if needed. To compare two *distinct* architectures
#     (the study's intent), point MODEL_B at e.g. a DeepSeek/Llama MTP model that
#     shares no tokenizer with MODEL_A and give it its own DRAFT_MODEL_B.

@dataclass
class ModelSpec:
    name: str            # short label used in result tables
    repo: str            # Hugging Face repo id
    draft_repo: str      # small model that emulates the MTP heads (shares tokenizer)


MODELS = [
    ModelSpec(
        name="Qwen3.5-9B",
        repo="Qwen/Qwen3.5-9B",
        draft_repo="Qwen/Qwen3.5-0.6B",
    ),
    ModelSpec(
        name="Qwen3.5-4B",
        repo="Qwen/Qwen3.5-4B",
        draft_repo="Qwen/Qwen3.5-0.6B",
    ),
]


# ---------------------------------------------------------------------------
# Decoding configurations (the three execution modes under study)
# ---------------------------------------------------------------------------
#   standard    : plain greedy decoding, one token per target forward pass.
#   speculative : propose K tokens with the cheap draft, then VERIFY them with a
#                 single target pass and keep only the matching prefix. Output is
#                 token-for-token identical to `standard` (lossless) but faster.
#   parallel    : propose K tokens and accept ALL of them with NO verification.
#                 Fastest, but errors compound -> this is the mode whose accuracy
#                 we expect to fall off.
DECODE_MODES = ["standard", "speculative", "parallel"]

# How many tokens the draft proposes per block (the "lookahead" / number of
# extra MTP heads). 4 matches the facebook MTP paper's n=4.
SPECULATION_K = 4


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    name: str            # short label
    hf_path: str         # datasets.load_dataset path
    hf_config: str       # dataset config / subset
    split: str           # which split to evaluate


TASKS = [
    TaskSpec("gsm8k",     "gsm8k",          "main",       "test"),
    TaskSpec("humaneval", "openai_humaneval", "openai_humaneval", "test"),
    TaskSpec("mmlu",      "cais/mmlu",      "all",        "test"),
    TaskSpec("arc",       "allenai/ai2_arc", "ARC-Challenge", "test"),
]


# ---------------------------------------------------------------------------
# Generation / run settings
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    limit: int = 50                 # examples per dataset (small => fast smoke run)
    max_new_tokens: int = 256       # generation budget per example
    code_max_new_tokens: int = 384  # HumanEval needs a bit more room
    dtype: str = "bfloat16"         # bfloat16 is ideal on modern NVIDIA cards
    device: str = "cuda"
    seed: int = 0
    allow_code_exec: bool = False   # HumanEval executes model output -> opt-in only
    results_dir: str = "results"
    models: list = field(default_factory=lambda: MODELS)
    tasks: list = field(default_factory=lambda: TASKS)
    modes: list = field(default_factory=lambda: DECODE_MODES)
