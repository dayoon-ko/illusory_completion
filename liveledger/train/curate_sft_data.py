"""
Curate SFT training data from LiveLedger inference outputs.

Extracts two types of training data:
1. Constraint splitting (Phase 1): question -> constraints
2. Ledger updating (Phase 3): (ledger + thinking + query + results) -> updated entries

Filters by correctness from epistemic ledger evaluation.
Outputs ms-swift compatible JSONL format.

Usage:
    python 3_curate_sft_data.py \
        --input_dir outputs \
        --output_dir training_data \
        --datasets browsecomp deepsearchqa frames livedrbench webwalkerqa \
        --task both
"""

import argparse
import json
import os
import sys
import random
from collections import Counter
from glob import glob
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import TOOLS_EXTRACT, TOOLS_UPDATE


def check_verified_correct(data: Dict) -> bool:
    """Check if an output is both correct and verified.

    When verification fields are present (from run_verification.py),
    requires both is_correct=True AND verified=True.
    When not present, falls back to status=success only.
    """
    if "is_correct" in data and "verified" in data:
        return data.get("is_correct", False) and data.get("verified", False)
    # Fallback: no verification data, accept all successful outputs
    return True


def format_extract_example(extract_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format a constraint splitting example for ms-swift SFT."""
    if not extract_response or not extract_response.get("constraints"):
        return None

    messages = []

    # System message
    if extract_response.get("messages"):
        for msg in extract_response["messages"]:
            if msg["role"] == "system":
                messages.append({"role": "system", "content": msg["content"]})
                break

    # User message
    if extract_response.get("messages"):
        for msg in extract_response["messages"]:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
                break

    if len(messages) < 2:
        return None

    # Assistant response with tool call
    assistant_content = ""
    reasoning = extract_response.get("reasoning_content", "")
    content = extract_response.get("content", "")
    if reasoning:
        assistant_content = f"<think>{reasoning}</think>\n"
    if content:
        assistant_content += content

    tool_calls = extract_response.get("tool_calls", [])
    if not tool_calls:
        return None

    assistant_msg = {"role": "assistant", "content": assistant_content}
    # Normalize tool_calls to OpenAI format
    formatted_tool_calls = []
    for tc in tool_calls:
        formatted_tc = {
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name", "extract_constraints"),
                "arguments": tc.get("function", {}).get("arguments", ""),
            },
        }
        if "id" in tc:
            formatted_tc["id"] = tc["id"]
        formatted_tool_calls.append(formatted_tc)
    assistant_msg["tool_calls"] = formatted_tool_calls

    messages.append(assistant_msg)

    return {
        "messages": messages,
        "tools": TOOLS_EXTRACT,
    }


def format_update_example(update_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format a ledger update example for ms-swift SFT."""
    if not update_response:
        return None

    messages = []

    # System message
    if update_response.get("messages"):
        for msg in update_response["messages"]:
            if msg["role"] == "system":
                messages.append({"role": "system", "content": msg["content"]})
                break

    # User message
    if update_response.get("messages"):
        for msg in update_response["messages"]:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
                break

    if len(messages) < 2:
        return None

    # Assistant response with tool call
    assistant_content = ""
    reasoning = update_response.get("reasoning_content", "")
    content = update_response.get("content", "")
    if reasoning:
        assistant_content = f"<think>{reasoning}</think>\n"
    if content:
        assistant_content += content

    tool_calls = update_response.get("tool_calls", [])
    if not tool_calls:
        return None

    assistant_msg = {"role": "assistant", "content": assistant_content}
    formatted_tool_calls = []
    for tc in tool_calls:
        formatted_tc = {
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name", "update_ledger"),
                "arguments": tc.get("function", {}).get("arguments", ""),
            },
        }
        if "id" in tc:
            formatted_tc["id"] = tc["id"]
        formatted_tool_calls.append(formatted_tc)
    assistant_msg["tool_calls"] = formatted_tool_calls

    messages.append(assistant_msg)

    return {
        "messages": messages,
        "tools": TOOLS_UPDATE,
    }


def process_dataset(
    input_dir: str,
    dataset: str,
    task: str,
) -> tuple:
    """Process a single dataset and return (extract_examples, update_examples)."""
    extract_examples = []
    update_examples = []

    dataset_input_dir = os.path.join(input_dir, dataset)
    if not os.path.exists(dataset_input_dir):
        print(f"  Warning: input dir not found: {dataset_input_dir}")
        return extract_examples, update_examples

    # Find all output files
    output_files = glob(os.path.join(dataset_input_dir, "*.json"))
    output_files = [f for f in output_files if "summary" not in os.path.basename(f)]
    output_files = sorted(output_files, key=lambda x: int(os.path.basename(x).split(".")[0]))

    skipped_unverified = 0
    skipped_error = 0

    for filepath in output_files:
        idx = int(os.path.basename(filepath).split(".")[0])

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Warning: failed to load {filepath}: {e}")
            skipped_error += 1
            continue

        if data.get("status") != "success":
            skipped_error += 1
            continue

        # Filter: is_correct=True AND verified=True
        if not check_verified_correct(data):
            skipped_unverified += 1
            continue

        # Type 1: Constraint splitting
        if task in ("extract", "both"):
            extract_response = data.get("extract_response", {})
            example = format_extract_example(extract_response)
            if example:
                example["metadata"] = {
                    "dataset": dataset,
                    "idx": idx,
                    "question": data.get("question", ""),
                }
                extract_examples.append(example)

        # Type 2: Ledger updating
        if task in ("update", "both"):
            update_responses = data.get("update_responses", [])
            for ur in update_responses:
                example = format_update_example(ur)
                if example:
                    ledger_after = ur.get("ledger_after", {})
                    n_entities = len(ledger_after) if ledger_after else 0
                    example["metadata"] = {
                        "dataset": dataset,
                        "idx": idx,
                        "turn": ur.get("turn", -1),
                        "question": data.get("question", ""),
                        "n_entities": n_entities,
                    }
                    update_examples.append(example)

    print(f"  Files: {len(output_files)}, Skipped (unverified/incorrect): {skipped_unverified}, Skipped (error): {skipped_error}")

    return extract_examples, update_examples


def main():
    parser = argparse.ArgumentParser(description="Curate SFT training data from LiveLedger outputs")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Directory with verified inference outputs (from run_verification.py)")
    parser.add_argument("--output_dir", "-o", type=str, default="training_data",
                        help="Output directory for SFT JSONL files")
    parser.add_argument("--datasets", "-d", nargs="+", type=str,
                        default=["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa"],
                        help="Datasets to process")
    parser.add_argument("--task", "-t", type=str, default="both",
                        choices=["extract", "update", "both"],
                        help="Which training data to generate")
    parser.add_argument("--entity1_ratio", type=float, default=0.2,
                        help="Max ratio of entity=1 examples in update data (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_extract = []
    all_update = []

    for dataset in args.datasets:
        print(f"\nProcessing {dataset}...")
        extract_examples, update_examples = process_dataset(
            args.input_dir, dataset, args.task
        )
        all_extract.extend(extract_examples)
        all_update.extend(update_examples)
        print(f"  Extract examples: {len(extract_examples)}, Update examples: {len(update_examples)}")

    # Balance update examples: cap entity=1 ratio
    if args.task in ("update", "both") and all_update:
        ent_before = Counter(ex["metadata"]["n_entities"] for ex in all_update)
        print(f"\nUpdate examples entity distribution (before balancing):")
        for k in sorted(ent_before):
            pct = ent_before[k] / len(all_update) * 100
            print(f"  entity={k}: {ent_before[k]} ({pct:.1f}%)")
        print(f"  Total: {len(all_update)}")

        single = [ex for ex in all_update if ex["metadata"]["n_entities"] == 1]
        multi = [ex for ex in all_update if ex["metadata"]["n_entities"] >= 2]
        max_single = int(len(multi) * args.entity1_ratio / (1 - args.entity1_ratio))

        if len(single) > max_single:
            rng = random.Random(args.seed)
            rng.shuffle(single)
            single = single[:max_single]

        all_update = single + multi
        rng = random.Random(args.seed)
        rng.shuffle(all_update)

        ent_after = Counter(ex["metadata"]["n_entities"] for ex in all_update)
        print(f"\nUpdate examples entity distribution (after balancing):")
        for k in sorted(ent_after):
            pct = ent_after[k] / len(all_update) * 100
            print(f"  entity={k}: {ent_after[k]} ({pct:.1f}%)")
        print(f"  Total: {len(all_update)}")

    # Write output files
    if args.task in ("extract", "both") and all_extract:
        extract_path = os.path.join(args.output_dir, "sft_constraint_split.jsonl")
        with open(extract_path, "w") as f:
            for example in all_extract:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(all_extract)} constraint splitting examples to {extract_path}")

    if args.task in ("update", "both") and all_update:
        update_path = os.path.join(args.output_dir, "sft_ledger_update.jsonl")
        with open(update_path, "w") as f:
            for example in all_update:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_update)} ledger update examples to {update_path}")

    # Write stats
    stats = {
        "total_extract_examples": len(all_extract),
        "total_update_examples": len(all_update),
        "datasets": args.datasets,
        "task": args.task,
        "input_dir": args.input_dir,
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
