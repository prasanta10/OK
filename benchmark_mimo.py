"""
Step 1: Environment Setup & Baseline Verification
==================================================

Before running this script, install dependencies:

    pip install vllm transformers torch

Hardware requirement: 1x GPU with >= 16GB VRAM (A100, A6000, RTX 4090, etc.)
MiMo-7B is ~14GB in fp16, so 24GB+ is comfortable.

This script does three things:
  1. Loads MiMo-7B-RL in standard autoregressive mode (Mode A)
  2. Loads MiMo-7B-RL with MTP speculative decoding (Mode C)
  3. Runs the same prompt through both and compares outputs

If both produce reasonable text, your setup is correct and we can move on.
"""

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "XiaomiMiMo/MiMo-7B-RL"

# We use greedy decoding (temperature=0) so outputs are deterministic.
# This matters because Mode C (speculative) must produce IDENTICAL output
# to Mode A (standard AR) under greedy decoding — that's our sanity check.
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,       # greedy — deterministic output
    max_tokens=256,        # enough for a short math solution
    stop=["Question:"],    # GSM8K-style stop token
)

# A simple GSM8K-style math problem to test with
TEST_PROMPT = (
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. She "
    "sells every duck egg at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market?\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Part 1: Mode A — Standard Autoregressive (no speculative decoding)
# ---------------------------------------------------------------------------
print("=" * 60)
print("MODE A: Standard autoregressive decoding")
print("=" * 60)

llm_mode_a = LLM(
    model=MODEL_ID,
    trust_remote_code=True,       # MiMo uses custom model code
    tensor_parallel_size=1,       # single GPU
    # No speculative_config → standard 1-token-per-step decoding
)

output_a = llm_mode_a.generate([TEST_PROMPT], SAMPLING_PARAMS)
text_a = output_a[0].outputs[0].text

print(f"\nPrompt:\n{TEST_PROMPT}\n")
print(f"Mode A output:\n{text_a}")
print(f"\nTokens generated: {len(output_a[0].outputs[0].token_ids)}")

# Free GPU memory before loading Mode C
del llm_mode_a


# ---------------------------------------------------------------------------
# Part 2: Mode C — MTP with speculative verification
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODE C: MTP speculative decoding (with verification)")
print("=" * 60)

llm_mode_c = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",              # use native MTP layer
        "num_speculative_tokens": 1,   # draft 1 extra token per step
    },
)

output_c = llm_mode_c.generate([TEST_PROMPT], SAMPLING_PARAMS)
text_c = output_c[0].outputs[0].text

print(f"\nPrompt:\n{TEST_PROMPT}\n")
print(f"Mode C output:\n{text_c}")
print(f"\nTokens generated: {len(output_c[0].outputs[0].token_ids)}")

del llm_mode_c


# ---------------------------------------------------------------------------
# Part 3: Compare outputs — they MUST match under greedy decoding
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

if text_a == text_c:
    print("✅ PASS: Mode A and Mode C produced identical output.")
    print("   This confirms speculative decoding is lossless under greedy.")
else:
    print("❌ MISMATCH: Mode A and Mode C differ!")
    print("   This can happen due to floating point precision.")
    print("   Check if the final numeric answer is the same.")
    print(f"\n   Mode A: ...{text_a[-100:]}")
    print(f"   Mode C: ...{text_c[-100:]}")

print("\n" + "=" * 60)
print("EXPECTED ANSWER: $18")
print("(16 eggs - 3 breakfast - 4 muffins = 9 eggs × $2 = $18)")
print("=" * 60)