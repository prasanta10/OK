"""
Step 2: GSM8K Evaluation — Custom Script for All Modes
======================================================

This replaces lm-eval with a direct evaluation script.
We load GSM8K, format prompts, run vLLM, extract answers, compute accuracy.

This gives us full control over vLLM's config — including speculative_config —
and will be the same script we extend for Mode B.

Usage:
    python step2_eval.py --mode a          # Standard AR
    python step2_eval.py --mode c          # MTP with verification
    python step2_eval.py --limit 50        # Run more samples
    python step2_eval.py --mode a --all    # Run full test set (1319 samples)
"""

import argparse
import json
import os
import re
import time

os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL_ID = "XiaomiMiMo/MiMo-7B-RL"


# ===================================================================
# 1. GSM8K PROMPT FORMATTING
# ===================================================================
# GSM8K uses 5-shot prompting. The few-shot examples are drawn from
# the training split. Each example shows the question, a step-by-step
# solution, and a final answer after "####".

def get_fewshot_examples(train_data, n=5):
    """Pick the first n training examples as few-shot demonstrations."""
    examples = []
    for i in range(n):
        q = train_data[i]["question"]
        a = train_data[i]["answer"]
        examples.append(f"Question: {q}\nAnswer: {a}")
    return "\n\n".join(examples)


def format_prompt(question, fewshot_prefix):
    """
    Construct the full prompt:
      [5 few-shot examples]

      Question: <test question>
      Answer:

    The model continues from "Answer:" and generates the solution.
    """
    return f"{fewshot_prefix}\n\nQuestion: {question}\nAnswer:"


# ===================================================================
# 2. ANSWER EXTRACTION
# ===================================================================
# GSM8K answers are formatted as "#### <number>" at the end of the
# solution. We extract this with regex, matching what lm-eval does.

def extract_answer(generated_text):
    """
    Extract the numeric answer from generated text.
    Tries two strategies (same as lm-eval's strict + flexible filters):
      1. strict: look for "#### <number>"
      2. flexible: take the last number in the text
    """
    # Strategy 1: strict — look for #### pattern
    match = re.search(r"####\s*(-?[\d,\.]+)", generated_text)
    if match:
        return match.group(1).replace(",", "").strip().rstrip(".")

    # Strategy 2: flexible — last number in the text
    numbers = re.findall(r"(-?[\d,\.]+)", generated_text)
    if numbers:
        return numbers[-1].replace(",", "").strip().rstrip(".")

    return None


def extract_gold_answer(answer_text):
    """Extract the gold answer from GSM8K's answer field."""
    match = re.search(r"####\s*(-?[\d,\.]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip().rstrip(".")
    return None


# ===================================================================
# 3. MAIN EVALUATION LOOP
# ===================================================================

def run_evaluation(mode, limit):
    # --- Load dataset ---
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]
    test_data = dataset["test"]

    if limit is not None:
        test_data = test_data.select(range(min(limit, len(test_data))))

    print(f"Test samples: {len(test_data)}")

    # --- Build few-shot prefix ---
    fewshot_prefix = get_fewshot_examples(train_data, n=5)

    # --- Format all prompts ---
    prompts = [
        format_prompt(sample["question"], fewshot_prefix)
        for sample in test_data
    ]

    # --- Gold answers ---
    gold_answers = [
        extract_gold_answer(sample["answer"])
        for sample in test_data
    ]

    # --- Initialize vLLM ---
    print(f"\nInitializing vLLM — {mode}...")

    llm_kwargs = {
        "model": MODEL_ID,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
    }

    if mode == "c":
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 1,
        }
    # Mode B will be added in Step 3

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["Question:"],
    )

    # --- Generate ---
    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    # --- Score ---
    correct_strict = 0
    correct_flexible = 0
    total_tokens = 0
    results_log = []

    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        total_tokens += num_tokens

        pred_answer = extract_answer(generated)
        gold = gold_answers[i]
        is_correct = pred_answer == gold

        if is_correct:
            correct_strict += 1
            correct_flexible += 1
        else:
            # Flexible: try matching just the number without #### prefix
            numbers = re.findall(r"(-?[\d,\.]+)", generated)
            if numbers:
                flexible_pred = numbers[-1].replace(",", "").strip().rstrip(".")
                if flexible_pred == gold:
                    correct_flexible += 1

        results_log.append({
            "index": i,
            "question": test_data[i]["question"],
            "gold_answer": gold,
            "predicted_answer": pred_answer,
            "correct": is_correct,
            "generated_text": generated[:500],  # truncate for readability
            "num_tokens": num_tokens,
        })

    # --- Report ---
    n = len(test_data)
    accuracy_strict = correct_strict / n
    accuracy_flexible = correct_flexible / n
    tokens_per_second = total_tokens / elapsed

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {mode}")
    print(f"{'=' * 60}")
    print(f"  Samples:           {n}")
    print(f"  Strict accuracy:   {accuracy_strict:.4f} ({correct_strict}/{n})")
    print(f"  Flexible accuracy: {accuracy_flexible:.4f} ({correct_flexible}/{n})")
    print(f"  Total tokens:      {total_tokens}")
    print(f"  Wall time:         {elapsed:.2f}s")
    print(f"  Tokens/sec:        {tokens_per_second:.1f}")
    print(f"{'=' * 60}")

    # --- Save results ---
    output_dir = f"results/mode_{mode[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "mode": mode,
        "samples": n,
        "strict_accuracy": accuracy_strict,
        "flexible_accuracy": accuracy_flexible,
        "total_tokens": total_tokens,
        "wall_time_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "samples.jsonl"), "w") as f:
        for r in results_log:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {output_dir}/")
    return summary


# ===================================================================
# 4. CLI
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["a", "c"], default="a",
                        help="a = standard AR, c = MTP with verification")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of test samples (default 20)")
    parser.add_argument("--all", action="store_true",
                        help="Run on full test set (1319 samples)")
    args = parser.parse_args()

    limit = None if args.all else args.limit

    mode_names = {
        "a": "Mode A — Standard AR",
        "c": "Mode C — MTP + Verify",
    }

    run_evaluation(mode_names[args.mode], limit)