"""
Step 2: GSM8K Evaluation — Single Mode per Run
================================================

Usage:
    python step2_eval.py --mode a       # Standard AR
    python step2_eval.py --mode c       # MTP with verification

Run each mode as a separate invocation so the GPU is fully
released between runs.
"""

import argparse
import lm_eval
import os

os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID = "XiaomiMiMo/MiMo-7B-RL"


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["a", "c"], required=True,
                    help="a = standard AR, c = MTP with verification")
parser.add_argument("--limit", type=int, default=20,
                    help="Number of GSM8K samples to evaluate")
args = parser.parse_args()


# -------------------------------------------------------------------
# Build model_args based on mode
# -------------------------------------------------------------------
base_args = f"pretrained={MODEL_ID},trust_remote_code=True,gpu_memory_utilization=0.9"

if args.mode == "a":
    model_args = base_args
    mode_name = "Mode A — Standard AR"
    output_path = "results/mode_a"

elif args.mode == "c":
    # Pass speculative config as individual kwargs that vLLM understands
    model_args = (
        f"{base_args}"
        f",speculative_model={MODEL_ID}"    # MTP uses the same model
        f",num_speculative_tokens=1"
        f",speculative_method=mtp"          # explicitly set MTP method
    )
    mode_name = "Mode C — MTP + Verify"
    output_path = "results/mode_c"


# -------------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"Running: {mode_name}")
print(f"Samples: {args.limit}")
print(f"{'=' * 60}\n")

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args=model_args,
    tasks=["gsm8k"],
    num_fewshot=5,
    limit=args.limit,
    log_samples=True,
)

# -------------------------------------------------------------------
# Print results
# -------------------------------------------------------------------
gsm8k = results["results"]["gsm8k"]
strict = gsm8k["exact_match,strict-match"]
flexible = gsm8k["exact_match,flexible-extract"]

print(f"\n{'=' * 60}")
print(f"{mode_name} Results:")
print(f"  strict-match:     {strict:.4f}")
print(f"  flexible-extract: {flexible:.4f}")
print(f"  Samples: {args.limit}")
print(f"  Output saved to: {output_path}/")
print(f"{'=' * 60}")