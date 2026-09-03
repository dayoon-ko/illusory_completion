"""Evaluation pipeline: build epistemic ledger (if needed) + judge accuracy + analyze failure modes.

Usage:
  # Evaluate LiveLedger outputs (ledger already built inline)
  python run_evaluation.py --output_dir liveledger/outputs --has_ledger

  # Evaluate baseline outputs (build ledger post-hoc)
  python run_evaluation.py --output_dir baselines/outputs/search-r1 \
      --baseline_name search-r1 --base_url http://localhost:8000/v1

  # Only run accuracy/failure-mode analysis on already-annotated outputs
  python run_evaluation.py --output_dir liveledger/outputs --has_ledger --skip_judge
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation pipeline")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory with inference outputs")
    parser.add_argument("--has_ledger", action="store_true",
                        help="Outputs already contain ledger (LiveLedger runs)")
    parser.add_argument("--baseline_name", default=None,
                        help="Baseline name for post-hoc ledger building (e.g., search-r1, react)")
    parser.add_argument("--dataset_dir", default="datasets", help="Dataset root directory")
    parser.add_argument("--datasets", "-d", nargs="+",
                        default=["frames", "browsecomp", "deepsearchqa", "livedrbench", "webwalkerqa", "bioasq"])
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="LLM judge server URL")
    parser.add_argument("--model_name", default="openai/gpt-oss-120b", help="LLM judge model")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--skip_ledger", action="store_true", help="Skip ledger building step")
    parser.add_argument("--skip_judge", action="store_true", help="Skip answer correctness judging")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip accuracy/failure-mode analysis")
    parser.add_argument("--num_workers", "-w", type=int, default=16)
    args = parser.parse_args()

    epistemic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epistemic_ledger")
    sys.path.insert(0, epistemic_dir)

    # Step 1: Build epistemic ledger post-hoc (only for baselines without inline ledger)
    if not args.has_ledger and not args.skip_ledger:
        if not args.baseline_name:
            parser.error("--baseline_name required when outputs don't have ledger (no --has_ledger)")
        print("=" * 60)
        print("Step 1: Building epistemic ledger post-hoc")
        print("=" * 60)
        build_ledger = importlib.import_module("build_ledger")
        for ds in args.datasets:
            ds_output = os.path.join(args.output_dir, ds)
            if not os.path.isdir(ds_output):
                print(f"  Skipping {ds} (no output directory)")
                continue
            print(f"  Processing {ds}...")
            build_ledger.JudgeAgent(
                base_url=args.base_url,
                model_name=args.model_name,
                api_key=args.api_key,
            ).process_baseline(
                baseline_name=args.baseline_name,
                dataset_name=ds,
                output_dir=ds_output,
                num_workers=args.num_workers,
            )
    else:
        print("Step 1: Skipped (ledger already present or --skip_ledger)")

    # Step 2: Judge answer correctness
    if not args.skip_judge:
        print("\n" + "=" * 60)
        print("Step 2: Judging answer correctness")
        print("=" * 60)
        evaluate = importlib.import_module("evaluate")
        evaluate.JUDGE_BASE_URL = args.base_url
        evaluate.JUDGE_MODEL = args.model_name
        evaluate.JUDGE_API_KEY = args.api_key
        for ds in args.datasets:
            ds_output = os.path.join(args.output_dir, ds)
            if not os.path.isdir(ds_output):
                continue
            print(f"  Judging {ds}...")
            evaluate.evaluate_outputs(ds_output, ds, num_workers=args.num_workers)
    else:
        print("Step 2: Skipped (--skip_judge)")

    # Step 3: Accuracy + failure mode analysis
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print("Step 3: Accuracy & failure mode analysis")
        print("=" * 60)
        accuracy = importlib.import_module("accuracy")
        failure_modes = importlib.import_module("failure_modes")

        acc_evaluator = accuracy.EpistemicLedgerAccuracyEvaluator(args.output_dir)
        acc_evaluator.evaluate()

        fm_evaluator = failure_modes.FailureModeEvaluator(args.output_dir)
        fm_evaluator.evaluate()
    else:
        print("Step 3: Skipped (--skip_analysis)")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
