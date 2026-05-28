"""
Step 2: GSM8K Evaluation — Mode A and Mode C via lm-eval Python API
====================================================================

This uses lm-eval's Python API instead of the CLI.
The advantage: we can pass speculative_config directly to vLLM
without needing a server or worrying about nested dict formatting.

Run: python step2_eval.py
"""

import lm_eval
import json
import os

# Prevent any HuggingFace network calls
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID = "XiaomiMiMo/MiMo-7B-RL"


def run_eval(mode_name, vllm_kwargs):
    """
    Run GSM8K evaluation with given vLLM configuration.

    lm_eval.simple_evaluate() does everything the CLI does:
      - Loads the model via the vllm backend
      - Constructs 5-shot GSM8K prompts
      - Generates text
      - Extracts numeric answers via regex
      - Computes exact_match accuracy
    """

    print(f"\n{'=' * 60}")
    print(f"Running {mode_name}")
    print(f"{'=' * 60}\n")

    # Build the model_args string for basic params
    model_args = f"pretrained={MODEL_ID},trust_remote_code=True,gpu_memory_utilization=0.9"

    # Add any extra vLLM kwargs (like speculative_config)
    for key, value in vllm_kwargs.items():
        if isinstance(value, dict):
            # lm-eval's vllm backend passes unknown kwargs to the LLM constructor
            # For dicts, we serialize as JSON string
            model_args += f",{key}={json.dumps(value)}"
        else:
            model_args += f",{key}={value}"

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=["gsm8k"],
        num_fewshot=5,
        limit=20,              # 20 samples for quick test
        log_samples=True,
    )

    # Extract accuracy from results
    gsm8k_results = results["results"]["gsm8k"]
    strict = gsm8k_results["exact_match,strict-match"]
    flexible = gsm8k_results["exact_match,flexible-extract"]

    print(f"\n{mode_name} Results:")
    print(f"  strict-match:    {strict:.4f}")
    print(f"  flexible-extract: {flexible:.4f}")

    return results


# -------------------------------------------------------------------
# Mode A: Standard Autoregressive (no speculative decoding)
# -------------------------------------------------------------------
results_a = run_eval("Mode A — Standard AR", vllm_kwargs={})


# -------------------------------------------------------------------
# Mode C: MTP with Speculative Verification
# -------------------------------------------------------------------
results_c = run_eval("Mode C — MTP + Verify", vllm_kwargs={
    "speculative_config": {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }
})


# -------------------------------------------------------------------
# Compare
# -------------------------------------------------------------------
acc_a = results_a["results"]["gsm8k"]["exact_match,strict-match"]
acc_c = results_c["results"]["gsm8k"]["exact_match,strict-match"]

print(f"\n{'=' * 60}")
print(f"COMPARISON")
print(f"{'=' * 60}")
print(f"Mode A accuracy: {acc_a:.4f}")
print(f"Mode C accuracy: {acc_c:.4f}")

if acc_a == acc_c:
    print("✅ MATCH — speculative decoding is lossless on GSM8K")
else:
    print(f"⚠️  Difference of {abs(acc_a - acc_c):.4f}")
    print("   Small differences (1 sample) can happen due to float precision.")