"""Run the MTP decoding benchmark.

Sweeps every (model x decode-mode x dataset) combination, measuring:
  - accuracy  : task-specific correctness (see datasets_tasks.py)
  - throughput: generated tokens per second on the expensive target model

Examples
--------
Quick smoke test (10 examples per dataset, no code execution):
    python benchmark.py --limit 10

Full-ish run including HumanEval code execution (run in a sandbox!):
    python benchmark.py --limit 100 --allow-code-exec

Subset of the matrix:
    python benchmark.py --models Qwen3.5-9B --tasks gsm8k arc --modes standard parallel
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from tabulate import tabulate

import config
import datasets_tasks as dt
import mtp_decoding as mtp


def parse_args() -> config.RunConfig:
    rc = config.RunConfig()
    p = argparse.ArgumentParser(description="MTP decoding benchmark")
    p.add_argument("--limit", type=int, default=rc.limit)
    p.add_argument("--max-new-tokens", type=int, default=rc.max_new_tokens)
    p.add_argument("--allow-code-exec", action="store_true",
                   help="execute model-generated code for HumanEval (sandbox!)")
    p.add_argument("--models", nargs="*", help="subset by model name")
    p.add_argument("--tasks", nargs="*", help="subset by task name")
    p.add_argument("--modes", nargs="*", help="subset of decode modes")
    p.add_argument("--results-dir", default=rc.results_dir)
    a = p.parse_args()

    rc.limit = a.limit
    rc.max_new_tokens = a.max_new_tokens
    rc.allow_code_exec = a.allow_code_exec
    rc.results_dir = a.results_dir
    if a.models:
        rc.models = [m for m in config.MODELS if m.name in a.models]
    if a.tasks:
        rc.tasks = [t for t in config.TASKS if t.name in a.tasks]
    if a.modes:
        rc.modes = [m for m in config.DECODE_MODES if m in a.modes]
    return rc


def run_one_combo(lm, proposer, task, mode, rc, is_code) -> dict:
    """Evaluate one (model, task, mode) cell of the matrix."""
    examples = task.examples(rc.limit)
    budget = rc.code_max_new_tokens if is_code else rc.max_new_tokens

    correct, scored = 0.0, 0
    total_tokens, total_seconds = 0, 0.0

    for ex in examples:
        ids = lm.tokenizer(ex["prompt"], return_tensors="pt").input_ids.to(lm.device)
        res = mtp.generate(lm, proposer, ids, mode, budget, config.SPECULATION_K)

        total_tokens += res.new_tokens
        total_seconds += res.seconds

        s = task.score(ex, res.text)
        if s is not None:          # None => task skipped (e.g. code exec off)
            correct += s
            scored += 1

    accuracy = (correct / scored) if scored else None
    throughput = (total_tokens / total_seconds) if total_seconds else 0.0
    return {
        "accuracy": accuracy,
        "throughput_tok_s": round(throughput, 1),
        "n_examples": len(examples),
        "n_scored": scored,
    }


def main():
    rc = parse_args()
    torch.manual_seed(rc.seed)
    os.makedirs(rc.results_dir, exist_ok=True)

    results = []  # flat list of result rows for easy saving / tabulating

    for mspec in rc.models:
        print(f"\n=== Loading {mspec.name} ({mspec.repo}) ===")
        lm = mtp.load_model(mspec.repo, mspec.draft_repo, rc.dtype, rc.device)
        proposer = mtp.DraftProposer(lm.draft, lm.device)

        for tspec in rc.tasks:
            task = dt.build_task(tspec, lm.tokenizer, rc.allow_code_exec)
            is_code = tspec.name == "humaneval"
            for mode in rc.modes:
                print(f"  [{mspec.name} | {tspec.name:9s} | {mode:11s}] running...",
                      flush=True)
                t0 = time.perf_counter()
                stats = run_one_combo(lm, proposer, task, mode, rc, is_code)
                stats.update(model=mspec.name, task=tspec.name, mode=mode,
                             wall_seconds=round(time.perf_counter() - t0, 1))
                results.append(stats)
                acc = "N/A" if stats["accuracy"] is None else f"{stats['accuracy']:.3f}"
                print(f"      acc={acc}  throughput={stats['throughput_tok_s']} tok/s")

        # free GPU memory before the next model
        del lm, proposer
        torch.cuda.empty_cache()

    save_and_report(results, rc)


def save_and_report(results, rc):
    out_path = os.path.join(rc.results_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n================ SUMMARY ================")
    rows = [[r["model"], r["task"], r["mode"],
             "N/A" if r["accuracy"] is None else f"{r['accuracy']:.3f}",
             r["throughput_tok_s"]] for r in results]
    print(tabulate(rows,
                   headers=["model", "task", "mode", "accuracy", "tok/s"],
                   tablefmt="github"))
    print(f"\nSaved detailed results -> {out_path}")


if __name__ == "__main__":
    main()
