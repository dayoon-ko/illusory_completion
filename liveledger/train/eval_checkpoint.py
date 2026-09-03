"""
Downstream eval for a trained checkpoint using hybrid 1_run.py.

SEARCH phase uses gpt-oss-120b (user-provided server).
EXTRACT/UPDATE phases use the trained model (user-provided server).

python 5_eval_checkpoint.py --search_base_url http://localhost:8000/v1 --ledger_base_url http://localhost:8100/v1  --ledger_model_name sft_output/Qwen3.5-4B_lr3e-05_ep10_ml8192/checkpoint-52-merged  -d frames

Prerequisites:
    1. gpt-oss-120b vLLM server running (for SEARCH)
    2. Trained model vLLM server running (for EXTRACT/UPDATE)

Usage:
    # Start trained model server (user does this manually)
    vllm serve sft_output/Qwen3.5-4B_lr3e-05_ep10_ml8192/checkpoint-52-merged  --port 8100 --tensor-parallel-size 1 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 --enforce-eager

    # Run eval
    python 5_eval_checkpoint.py \
        --ledger_base_url http://localhost:8100/v1 \
        --ledger_model_name sft_output/Qwen3.5-4B_lr3e-05_ep10_ml8192/checkpoint-52-merged  \
        --search_base_url http://localhost:9100/v1
"""

import argparse
import json
import os
import subprocess
import sys
from glob import glob


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIVELEDGER_DIR = os.path.dirname(SCRIPT_DIR)


def compute_accuracy(eval_output_dir: str) -> dict:
    """Compute accuracy from 1_run.py output files."""
    files = [f for f in glob(os.path.join(eval_output_dir, "*.json"))
             if "summary" not in os.path.basename(f)]

    total, correct, errors = 0, 0, 0

    for fp in sorted(files):
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            errors += 1
            continue

        if data.get("status") != "success":
            errors += 1
            continue

        total += 1
        prediction = data.get("prediction", "").strip().lower()
        answer = data.get("answer", "")

        if isinstance(answer, dict):
            ground_truths = answer.get("ground_truths", [])
            is_correct = any(prediction == gt.strip().lower() for gt in ground_truths)
        else:
            is_correct = prediction == str(answer).strip().lower()

        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return {"total": total, "correct": correct, "errors": errors, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Downstream eval using hybrid 1_run.py")

    # Server URLs (both must be running already). Comma-separated for multi-instance fan-out.
    parser.add_argument("--search_base_url", type=str, required=True,
                        help="vLLM server(s) for SEARCH phase. Comma-separated for multiple instances; agents stick to one URL each so prefix cache stays warm.")
    parser.add_argument("--search_model_name", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--ledger_base_url", type=str, required=True,
                        help="vLLM server(s) for EXTRACT/UPDATE phases. Comma-separated for multiple instances.")
    parser.add_argument("--ledger_model_name", type=str, required=True,
                        help="Model name for EXTRACT/UPDATE phases")

    # Eval settings
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "..", "datasets"))
    parser.add_argument("--dataset_names", "-d", nargs="+", type=str,
                        default=["validation"])
    parser.add_argument("--output_dir", "-o", type=str, default="react_20B_liveledger_4B/run_2")
    parser.add_argument("--num_workers", "-w", type=int, default=16)
    parser.add_argument("--max_turns", type=int, default=30)
    parser.add_argument("--start_index", "-s", type=int, default=0,
                        help="Start example index (inclusive). Forwarded to 1_run.py as --start_idx.")
    parser.add_argument("--end_index", "-e", type=int, default=None,
                        help="End example index (exclusive). Forwarded to 1_run.py as --end_idx.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    runner = os.path.join(LIVELEDGER_DIR, "run.py")

    cmd = [
        sys.executable, runner,
        "--model_name", args.search_model_name,
        "--base_url", args.search_base_url,
        "--ledger_base_url", args.ledger_base_url,
        "--ledger_model_name", args.ledger_model_name,
        "--dataset_dir", args.dataset_dir,
        "-d", *args.dataset_names,
        "-o", args.output_dir,
        "-w", str(args.num_workers),
        "--max_turns", str(args.max_turns),
    ]
    if args.start_index:
        cmd.extend(["--start_idx", str(args.start_index)])
    if args.end_index is not None:
        cmd.extend(["--end_idx", str(args.end_index)])

    print(f"{'='*60}")
    print(f"Downstream Eval")
    print(f"  SEARCH:         {args.search_model_name} @ {args.search_base_url}")
    print(f"  EXTRACT/UPDATE: {args.ledger_model_name} @ {args.ledger_base_url}")
    print(f"  Datasets:       {args.dataset_names}")
    print(f"  Output:         {args.output_dir}")
    print(f"  Runner:         {runner}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=LIVELEDGER_DIR)
    if result.returncode != 0:
        print(f"\nWARNING: 1_run.py exited with code {result.returncode}")

    # Compute accuracy per dataset
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    all_results = {}
    for ds_name in args.dataset_names:
        ds_dir = os.path.join(args.output_dir, ds_name)
        if os.path.isdir(ds_dir):
            acc = compute_accuracy(ds_dir)
            all_results[ds_name] = acc
            print(f"  {ds_name}: {acc['correct']}/{acc['total']} ({acc['accuracy']:.1%}) [errors: {acc['errors']}]")
        else:
            print(f"  {ds_name}: no output found")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
