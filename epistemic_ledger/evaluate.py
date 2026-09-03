import json
import argparse
import os
import re
import time
import hashlib
import base64
from tqdm import tqdm 
from glob import glob
from openai import OpenAI
from multiprocessing.pool import ThreadPool


# =============================================================================
# Constants
# =============================================================================

# Cap bioasq evaluation to the first N items (by item index) to keep its
# weight comparable to the other datasets. Must match accuracy.py.

PROMPT = """
### Instruction
You are an impartial evaluator.

You will be given:
- a **question**
- a **gold (reference) answer**, which may contain **one or more valid entities**
- a **predicted answer**

Your task is to determine whether the **predicted answer is correct**.

### Rules (follow strictly)
1. If the gold answer contains **multiple valid entities**, the predicted answer(s) is **correct if and only if** it clearly matches **any one** of the gold entities.
2. If the gold answer contains a **single entity**, the predicted answer(s) must match **that exact entity**.
3. Minor surface differences (e.g., capitalization, abbreviations, aliases, name order) are allowed **only if** they unambiguously refer to the same entity.
4. If the predicted answer(s) refers to a **different entity**, the verdict must be **false**, even if it partially satisfies the question.
5. If the predicted answer(s) is **more general, more specific, or a different category** than the gold answer(s), the verdict must be **false**.
6. Do **not** use outside knowledge beyond comparing the gold and predicted answers.
7. If equivalence or membership is **ambiguous**, default to **false**.
8. Do not reward partial correctness.

### Output Format (exactly)
```json
{{
  "verdict": true or false,
  "justification": "One-sentence explanation."
}}

### Inputs
- question: {question}
- gold answer: {answer}
- predicted answer: {predicted_answer}
"""


# =============================================================================
# Utility Functions (Decryption)
# =============================================================================

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256.
    Code derived from https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
    """
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR.
    Code derived from https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
    """
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


# =============================================================================
# LLM Call
# =============================================================================

REASONING_EFFORT = None  # set from CLI; "low" | "medium" | "high" | None
JUDGE_BASE_URL = "http://localhost:8000/v1"
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_API_KEY = "EMPTY"


def call_llm(question, answer, predicted_answer, dataset_name=None):
    client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    while True:
        try:
            kwargs = {
                "model": JUDGE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": PROMPT.format(
                        question=question,
                        answer=answer,
                        predicted_answer=predicted_answer.split("<tool_call>")[0].strip()
                    )
                }],
            }
            if REASONING_EFFORT:
                kwargs["reasoning_effort"] = REASONING_EFFORT
                kwargs["max_tokens"] = 8192
            response = client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content or ""
            output = output.replace("```json", "").replace("```", "")
            m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", output, re.DOTALL)
            if m:
                output = m.group(0)

            output = json.loads(output)["verdict"]
            if type(output) == bool:
                return output

        except Exception as e:
            print(f"Error calling LLM: {e}")
            time.sleep(1)
            continue


# =============================================================================
# Answer Extraction Functions (per baseline)
# =============================================================================

def check_search_r1(data, answer, dataset_name=None):
    """Extract and check answer for search-r1, rag-r1 baselines."""
    question = data["question"]
    predicted_answer = re.findall(
        r"<answer>(.*?)</answer>",
        data["output"].split("</think>")[-1],
        re.DOTALL
    )

    if len(predicted_answer) == 0:
        return False, 0

    predicted_answer = predicted_answer[-1]
    is_correct = call_llm(question, answer, predicted_answer, dataset_name)
    return True, is_correct


def check_asearcher(data, answer, dataset_name=None):
    """Extract and check answer for asearcher baseline."""
    question = data["question"]
    output = data["output"]["thinking_blocks"][-1]

    if "prediction of the answer:" not in output:
        return False, 0

    predicted_answer = output.split("prediction of the answer: ")[-1].strip()
    if predicted_answer == "":
        return False, 0

    is_correct = call_llm(question, answer, predicted_answer, dataset_name)
    return True, is_correct


def check_webexplorer(data, answer, dataset_name=None):
    """Extract and check answer for webexplorer, tongyidr baselines."""
    question = data["question"]

    if len(data["messages"]) == 0:
        return False, 0

    last_message = data["messages"][-1]
    if last_message["role"] != "assistant":
        return False, 0
    if last_message["content"].split("</think>")[-1].strip() == "":
        return False, 0

    predicted_answer = last_message["content"].split("</think>")[-1].strip()
    is_correct = call_llm(question, answer, predicted_answer, dataset_name)
    return True, is_correct


def check_dr_tulu(data, answer, dataset_name=None):
    """Extract and check answer for dr-tulu baseline."""
    question = data["question"]

    if len(data["final_response"]) == 0:
        return False, 0

    is_correct = call_llm(question, answer, data["final_response"], dataset_name)
    return True, is_correct


def check_hds(data, answer, dataset_name=None):
    """Extract and check answer for hds, hds-grpo baselines."""
    question = data["question"]

    if not data["finished"]:
        return False, 0

    predicted_answer = data["output"].split("</think>")[-1].strip()
    is_correct = call_llm(question, answer, predicted_answer, dataset_name)
    return True, is_correct


def check_react(data, answer, dataset_name=None):
    """Extract and check answer for react baselines."""
    question = data["question"]
    # predicted_answer = data["prediction"]
    predicted_answer = data["content"] if len(data["content"]) > 0 else data["prediction"]

    # Fallback logic for empty or timeout predictions
    if len(predicted_answer) == 0 or predicted_answer == "No answer found after 2h30mins":
        if data["content"] != "":
            predicted_answer = data["content"]
        elif len(data["messages"]) > 0:
            last_message = data["messages"][-1]
            if last_message["role"] == "user" and data["messages"][-2]["role"] == "assistant":
                predicted_answer = data["messages"][-2]["content"]
            elif last_message["role"] != "assistant":
                return False, 0
            elif last_message["content"].split("</think>")[-1].strip() == "":
                return False, 0
            else:
                predicted_answer = last_message["content"].split("</think>")[-1].strip()

    is_correct = call_llm(question, answer, predicted_answer, dataset_name)
    return True, is_correct


def check_search_o1(data, answer, dataset_name=None):
    """Extract and check answer for search_o1 baselines."""
    question = data["question"]
    predicted_answer = data["history"][-1]

    if "boxed" in predicted_answer:
        predicted_answer = predicted_answer.split("boxed")[-1].strip()
    elif "assistantfinal" in predicted_answer:
        predicted_answer = predicted_answer.split("assistantfinal")[-1].strip()

    if len(predicted_answer) == 0:
        return True, 0

    is_correct = call_llm(question, answer, predicted_answer, dataset_name)
    return True, is_correct


def check_standard_tao(data, answer, dataset_name=None):
    """Extract and check answer for baselines with standard TAO format.
    
    Expected format in data["output"]:
    - thinking_blocks: List of thinking strings (last one should contain the answer)
    - query_blocks: List of query strings
    - results_blocks: List of result strings
    
    Also checks for 'prediction' field directly in the output.
    """
    question = data["question"]
    output = data.get("output", {})
    
    # Try to get prediction from various sources
    predicted_answer = None
    
    # 1. Check for 'content' field
    if isinstance(output, dict) and "content" in output and len(output["content"]) > 0:
        predicted_answer = output["content"]
        
    # 2. Check for explicit 'prediction' field
    elif isinstance(output, dict) and "prediction" in output:
        predicted_answer = output["prediction"]
    
    # 3. Check for 'prediction' in data root
    elif "prediction" in data and data["prediction"]:
        predicted_answer = data["prediction"]
    
    # 4. Extract from last thinking block
    elif isinstance(output, dict) and "thinking_blocks" in output:
        thinking_blocks = output.get("thinking_blocks", [])
        if thinking_blocks:
            last_thinking = thinking_blocks[-1]
            # Try common patterns
            if "prediction of the answer:" in last_thinking.lower():
                predicted_answer = last_thinking.lower().split("prediction of the answer:")[-1].strip()
            elif "final answer:" in last_thinking.lower():
                predicted_answer = last_thinking.lower().split("final answer:")[-1].strip()
            elif "answer:" in last_thinking.lower():
                predicted_answer = last_thinking.lower().split("answer:")[-1].strip()
            else:
                # Use the entire last thinking block as answer
                predicted_answer = last_thinking.strip()
    
    # 5. Check for 'content' field (common fallback)
    elif "content" in data and data["content"]:
        predicted_answer = data["content"]
    
    if not predicted_answer or predicted_answer == "":
        return False, 0

    is_correct = call_llm(question, answer, predicted_answer, dataset_name)
    return True, is_correct


def has_standard_tao_format(data):
    """Check if data has standard TAO format."""
    output = data.get("output", {})
    if not isinstance(output, dict):
        return False
    required_keys = ["thinking_blocks", "query_blocks", "results_blocks"]
    return all(key in output for key in required_keys)


def get_num_turns(data):
    """Extract the number of search turns from data by counting assistant messages."""
    messages = data.get("messages", [])
    return sum(1 for m in messages if m.get("role") == "assistant")


# Baseline to checker function mapping
BASELINE_CHECKERS = {
    "search-r1": check_search_r1,
    "rag-r1": check_search_r1,
    "asearcher": check_asearcher,
    "webexplorer": check_webexplorer,
    "tongyidr": check_webexplorer,
    "tongyidr-liveledger-20b": check_webexplorer,
    "react": check_react, 
    "react_20B": check_react,
    "react_s1": check_react,
    "react_120B_liveledger_4B": check_react,
    "react_20B_liveledger_4B": check_react,
    "react_7B_liveledger_4B": check_react,
    "react_7B_liveledger_4B_tts": check_react,
    "search_o1_gpt-oss-20b": check_search_o1,
    "search_o1_gpt-oss-120b": check_search_o1,
    "hds": check_hds,
    "hds-grpo": check_hds,
    "dr-tulu": check_dr_tulu,
}


def get_checker(baseline_name, data):
    """Get appropriate checker function for baseline.
    
    Falls back to standard TAO checker for unknown baselines with TAO format.
    """
    # Try known baseline checker first
    if baseline_name.endswith("_2") or baseline_name.endswith("_3"):
        checker = BASELINE_CHECKERS.get(baseline_name.strip("_2").strip("_3"))
    else:
        checker = BASELINE_CHECKERS.get(baseline_name)
    if checker is not None:
        return checker
    
    # Fall back to standard TAO checker if data has the format
    if has_standard_tao_format(data):
        return check_standard_tao
    
    raise ValueError(
        f"Unknown baseline '{baseline_name}' and data does not have standard TAO format. "
        f"Expected 'thinking_blocks', 'query_blocks', 'results_blocks' in output."
    )


# =============================================================================
# Argument Parser
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../baselines/results",
                        help="Directory containing baseline JSONL results")
    parser.add_argument("--ledger_dir", type=str, default=None,
                        help="If set, read from ledger JSON files instead of baseline JSONL")
    parser.add_argument("--output_dir", type=str, default="annotations")
    parser.add_argument("--update_turns", action="store_true",
                        help="Force update num_turns and is_correct_penalized in existing annotations")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run the LLM judge and overwrite existing annotations (e.g., after changing the prompt)")
    parser.add_argument(
        "--baseline_name", "-b",
        nargs="+", type=str,
        default=[
            "search-r1", "search-r1_2", "search-r1_3",
            "rag-r1", "rag-r1_2", "rag-r1_3", 
            "asearcher", "asearcher_2", "asearcher_3",
            "dr-tulu", "dr-tulu_2", "dr-tulu_3",
            "hiprag", "hiprag_2", "hiprag_3",
            "reseek", "reseek_2", "reseek_3",
            "webexplorer", "webexplorer_2", "webexplorer_3",
            "search_o1_gpt-oss-20b", "search_o1_gpt-oss-20b_2", "search_o1_gpt-oss-20b_3", 
            "search_o1_gpt-oss-120b", "search_o1_gpt-oss-120b_2", "search_o1_gpt-oss-120b_3",   
            "tongyidr", "tongyidr_2", "tongyidr_3", 
            "tongyidr-liveledger-4B", "tongyidr-liveledger-4B_2", "tongyidr-liveledger-4B_3", 
            "react_20B", "react_20B_2", "react_20B_3",
            "react_20B_liveledger_4B", "react_20B_liveledger_4B_2", "react_20B_liveledger_4B_3",
            "react", "react_2", "react_3",
            "react_s1", "react_s1_2", "react_s1_3",
            "react_120B_liveledger_4B", "react_120B_liveledger_4B_2", "react_120B_liveledger_4B_3"
        ]
    )
    parser.add_argument(
        "--dataset_name", "-d",
        nargs="+", type=str,
        default=["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa", "bioasq"]
    )
    parser.add_argument("--num_workers", "-w", type=int, default=64,
                        help="Number of parallel workers for processing items")
    parser.add_argument("--reasoning_effort", default=None,
                        choices=[None, "low", "medium", "high"],
                        help="Reasoning effort to pass to the judge model")
    parser.add_argument("--base_url", default="http://localhost:8000/v1",
                        help="Base URL for the judge LLM server")
    parser.add_argument("--model_name", default="openai/gpt-oss-120b",
                        help="Model name for the judge LLM")
    parser.add_argument("--api_key", default="EMPTY")
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Accumulate per-baseline stats across datasets: {baseline: {dataset: (correct, total, finished, turns_list)}}
    baseline_stats = {}

    for dataset_name in args.dataset_name:
        print("=" * 31)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 31)

        # Load livedrbench answers if needed
        livedrbench_answer_dict = {}
        if dataset_name == "livedrbench":
            with open("../datasets/livedrbench/test_mcqa.jsonl", "r") as f:
                for line in f:
                    line = json.loads(line)
                    livedrbench_answer_dict[line["question"]] = decrypt(
                        line.get("ground_truths", ""),
                        line.get("canary", "")
                    )

        for baseline_name in args.baseline_name:
            # Load input data items
            items = []
            if args.ledger_dir:
                # Read from ledger JSON files
                input_dir = os.path.join(args.ledger_dir, baseline_name, dataset_name)
                input_files = glob(os.path.join(input_dir, "*.json"))
                input_files = [fn for fn in input_files if not fn.endswith("_.json")]
                input_files = sorted(
                    input_files,
                    key=lambda x: int(x.split("/")[-1].split("_")[-1].replace(".json", ""))
                )
                for fn in input_files:
                    item_id = fn.split("/")[-1]  # e.g. item_0.json
                    items.append((item_id, fn))
            else:
                # Read from baseline JSONL
                jsonl_path = os.path.join(args.input_dir, baseline_name, f"{dataset_name}.jsonl")
                if not os.path.exists(jsonl_path):
                    print(f"  Skipping {baseline_name}/{dataset_name}: {jsonl_path} not found")
                    continue
                with open(jsonl_path, "r") as f:
                    for idx, line in enumerate(f):
                        items.append((f"item_{idx}.json", line))

            os.makedirs(os.path.join(args.output_dir, baseline_name, dataset_name), exist_ok=True)

            def process_item(item):
                item_id, source = item
                save_fn = os.path.join(
                    args.output_dir, baseline_name, dataset_name, item_id
                )
                # Load data
                if args.ledger_dir:
                    with open(source, "r") as f:
                        data = json.load(f)
                else:
                    data = json.loads(source)

                if data.get("question", "") == "":
                    return

                # Get answer
                if dataset_name == "livedrbench":
                    answer = livedrbench_answer_dict[data["question"]]
                    data["answer"] = answer
                else:
                    answer = data.get("answer", data.get("Answer", ""))

                # Check answer using appropriate checker (supports unknown baselines with TAO format)
                checker = get_checker(baseline_name, data)
                finished, is_correct = checker(data, answer, dataset_name)

                # Save annotation with turn info
                num_turns = get_num_turns(data)
                annotation = {
                    "finished": finished,
                    "is_correct": is_correct,
                    "num_turns": num_turns,
                }
                with open(save_fn, "w") as f:
                    json.dump(annotation, f)

            with ThreadPool(processes=args.num_workers) as pool:
                list(tqdm(pool.imap_unordered(process_item, items), total=len(items)))

            # Print accuracy summary
            total = 0
            correct = 0
            finished_count = 0
            turns_list = []
            annotation_files = glob(os.path.join(args.output_dir, baseline_name, dataset_name, "*.json"))
            if dataset_name == "bioasq":
                annotation_files = [
                    f for f in annotation_files
                    if int(f.split("/")[-1].split("_")[-1].replace(".json", "")) 
                ]
            for af in annotation_files:
                if os.path.getsize(af) == 0:
                    continue
                with open(af, "r") as f:
                    ann = json.load(f)
                total += 1
                if ann.get("finished", False):
                    finished_count += 1
                if ann.get("is_correct", False):
                    correct += 1
                if "num_turns" in ann:
                    turns_list.append(ann["num_turns"])

            if total > 0:
                turns_str = ""
                if turns_list:
                    avg_t = sum(turns_list) / len(turns_list)
                    turns_str = (f", Turns: min={min(turns_list)}, max={max(turns_list)}, "
                                 f"mean={avg_t:.1f}")
                print(f"  {baseline_name}/{dataset_name}: "
                      f"Accuracy={correct}/{total} ({100*correct/total:.1f}%), "
                      f"Finished={finished_count}/{total} ({100*finished_count/total:.1f}%)"
                      f"{turns_str}")

                # Accumulate stats
                if baseline_name not in baseline_stats:
                    baseline_stats[baseline_name] = {}
                baseline_stats[baseline_name][dataset_name] = (correct, total, finished_count, turns_list)

    # Print per-baseline average across all datasets
    if baseline_stats:
        print("\n" + "=" * 60)
        print("Overall Average (per baseline, across all datasets)")
        print("=" * 60)
        for baseline_name in args.baseline_name:
            if baseline_name not in baseline_stats:
                continue
            ds_stats = baseline_stats[baseline_name]
            total_correct = sum(s[0] for s in ds_stats.values())
            total_items = sum(s[1] for s in ds_stats.values())
            total_finished = sum(s[2] for s in ds_stats.values())
            all_turns = [t for s in ds_stats.values() for t in s[3]]
            turns_str = ""
            if all_turns:
                avg_t = sum(all_turns) / len(all_turns)
                turns_str = f", Turns: mean={avg_t:.1f}"
            print(f"  {baseline_name}: "
                  f"Accuracy={total_correct}/{total_items} ({100*total_correct/total_items:.1f}%), "
                  f"Finished={total_finished}/{total_items} ({100*total_finished/total_items:.1f}%)"
                  f"{turns_str}")

    # Group baselines by prefix (strip _2, _3 suffixes) and compute mean/variance across runs
    import re as _re
    from collections import defaultdict

    def get_prefix(name):
        return _re.sub(r'_\d+$', '', name)

    # Build prefix groups: {prefix: [baseline_name, ...]}
    prefix_groups = defaultdict(list)
    for baseline_name in args.baseline_name:
        if baseline_name in baseline_stats:
            prefix_groups[get_prefix(baseline_name)].append(baseline_name)

    # Only print groups with multiple runs
    multi_run_groups = {p: runs for p, runs in prefix_groups.items() if len(runs) > 1}
    if multi_run_groups:
        print("\n" + "=" * 60)
        print("Multi-run Average (grouped by prefix, mean ± std)")
        print("=" * 60)
        for prefix, runs in multi_run_groups.items():
            # Compute accuracy (all correct / num all) for each run
            run_accs = []
            run_turns = []
            for run in runs:
                ds_stats = baseline_stats[run]
                total_correct = sum(s[0] for s in ds_stats.values())
                total_items = sum(s[1] for s in ds_stats.values())
                run_accs.append(total_correct / total_items if total_items > 0 else 0.0)
                all_turns = [t for s in ds_stats.values() for t in s[3]]
                run_turns.append(sum(all_turns) / len(all_turns) if all_turns else 0.0)

            n = len(run_accs)
            acc_mean = sum(run_accs) / n
            acc_var = sum((x - acc_mean) ** 2 for x in run_accs) / n
            turns_mean = sum(run_turns) / n
            turns_var = sum((x - turns_mean) ** 2 for x in run_turns) / n

            print(f"  {prefix} ({n} runs): "
                  f"Accuracy={100*acc_mean:.1f}% ± {100*acc_var**0.5:.1f}%, "
                  f"Turns: mean={turns_mean:.1f} ± {turns_var**0.5:.1f}")

            # Also per-dataset breakdown across runs
            all_datasets = sorted(set(ds for run in runs for ds in baseline_stats[run]))
            for ds in all_datasets:
                ds_accs = []
                for run in runs:
                    if ds in baseline_stats[run]:
                        s = baseline_stats[run][ds]
                        ds_accs.append(s[0] / s[1] if s[1] > 0 else 0.0)
                if len(ds_accs) > 1:
                    ds_mean = sum(ds_accs) / len(ds_accs)
                    ds_var = sum((x - ds_mean) ** 2 for x in ds_accs) / len(ds_accs)
                    print(f"    {ds}: {100*ds_mean:.1f}% ± {100*ds_var**0.5:.1f}%")


if __name__ == "__main__":
    args = get_args()
    if args.reasoning_effort:
        globals()["REASONING_EFFORT"] = args.reasoning_effort
    globals()["JUDGE_BASE_URL"] = args.base_url
    globals()["JUDGE_MODEL"] = args.model_name
    globals()["JUDGE_API_KEY"] = args.api_key
    main(args)
