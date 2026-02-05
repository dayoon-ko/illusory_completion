import json
import argparse
import os
import re
import time
import hashlib
import base64
from glob import glob
from openai import OpenAI
from multiprocessing.pool import ThreadPool


# =============================================================================
# Constants
# =============================================================================

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

def call_llm(question, answer, predicted_answer):
    """Call LLM to evaluate if predicted answer is correct."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{
                    "role": "user",
                    "content": PROMPT.format(
                        question=question,
                        answer=answer,
                        predicted_answer=predicted_answer
                    )
                }]
            )
            output = response.choices[0].message.content
            output = output.replace("```json", "").replace("```", "")            
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

def check_search_r1(data, answer):
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
    is_correct = call_llm(question, answer, predicted_answer)
    return True, is_correct


def check_asearcher(data, answer):
    """Extract and check answer for asearcher baseline."""
    question = data["question"]
    output = data["output"]["thinking_blocks"][-1]
    
    if "prediction of the answer:" not in output:
        return False, 0
    
    predicted_answer = output.split("prediction of the answer: ")[-1].strip()
    if predicted_answer == "":
        return False, 0
    
    is_correct = call_llm(question, answer, predicted_answer)
    return True, is_correct


def check_webexplorer(data, answer):
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
    is_correct = call_llm(question, answer, predicted_answer)
    return True, is_correct


def check_dr_tulu(data, answer):
    """Extract and check answer for dr-tulu baseline."""
    question = data["question"]
    
    if len(data["final_response"]) == 0:
        return False, 0
    
    is_correct = call_llm(question, answer, data["final_response"])
    return True, is_correct


def check_hds(data, answer):
    """Extract and check answer for hds, hds-grpo baselines."""
    question = data["question"]
    
    if not data["finished"]:
        return False, 0
    
    predicted_answer = data["output"].split("</think>")[-1].strip()
    is_correct = call_llm(question, answer, predicted_answer)
    return True, is_correct


def check_react(data, answer):
    """Extract and check answer for react baselines."""
    question = data["question"]
    predicted_answer = data["prediction"]
    
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
    
    is_correct = call_llm(question, answer, predicted_answer)
    return True, is_correct


def check_search_o1(data, answer):
    """Extract and check answer for search_o1 baselines."""
    question = data["question"]
    predicted_answer = data["history"][-1]
    
    if "boxed" in predicted_answer:
        predicted_answer = predicted_answer.split("boxed")[-1].strip()
    elif "assistantfinal" in predicted_answer:
        predicted_answer = predicted_answer.split("assistantfinal")[-1].strip()
    
    if len(predicted_answer) == 0:
        return True, 0
    
    is_correct = call_llm(question, answer, predicted_answer)
    return True, is_correct


def check_standard_tao(data, answer):
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
    
    # 1. Check for explicit 'prediction' field
    if isinstance(output, dict) and "prediction" in output:
        predicted_answer = output["prediction"]
    
    # 2. Check for 'prediction' in data root
    elif "prediction" in data and data["prediction"]:
        predicted_answer = data["prediction"]
    
    # 3. Extract from last thinking block
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
    
    # 4. Check for 'content' field (common fallback)
    elif "content" in data and data["content"]:
        predicted_answer = data["content"]
    
    if not predicted_answer or predicted_answer == "":
        return False, 0
    
    is_correct = call_llm(question, answer, predicted_answer)
    return True, is_correct


def has_standard_tao_format(data):
    """Check if data has standard TAO format."""
    output = data.get("output", {})
    if not isinstance(output, dict):
        return False
    required_keys = ["thinking_blocks", "query_blocks", "results_blocks"]
    return all(key in output for key in required_keys)


# Baseline to checker function mapping
BASELINE_CHECKERS = {
    "search-r1": check_search_r1,
    "rag-r1": check_search_r1,
    "asearcher": check_asearcher,
    "webexplorer": check_webexplorer,
    "hds": check_hds,
    "hds-grpo": check_hds,
    "dr-tulu": check_dr_tulu,
    "tongyidr": check_webexplorer,
    "tongyidr-liveledger-20b": check_webexplorer,
    "search_o1_gpt-oss-20b": check_search_o1,
    "search_o1_gpt-oss-120b": check_search_o1,
    "react_20b": check_react,
    "react_liveledger_20b": check_react,
    "react": check_react,
    "react_s1": check_react,
    "react_liveledger": check_react,
}


def get_checker(baseline_name, data):
    """Get appropriate checker function for baseline.
    
    Falls back to standard TAO checker for unknown baselines with TAO format.
    """
    # Try known baseline checker first
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
    parser.add_argument("--ledger_dir", type=str, default="output/epistemic_ledger")
    parser.add_argument("--output_dir", type=str, default="output/epistemic_ledger_finished")
    parser.add_argument(
        "--baseline_name", "-b",
        nargs="+", type=str,
        default=[
            "search-r1", "hds", "rag-r1", "asearcher", "dr-tulu", "webexplorer",
            "tongyidr", "tongyidr-liveledger-20b", 
            "search_o1_gpt-oss-20b", "search_o1_gpt-oss-120b",
            "react_20b", "react_liveledger_20b",
            "react", "react_s1", "react_liveledger",
        ]
    )
    parser.add_argument(
        "--dataset_name", "-d",
        nargs="+", type=str,
        default=["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa"]
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main(args):
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
            # Get input files
            input_dir = os.path.join(args.ledger_dir, baseline_name, dataset_name)
            output_files = glob(os.path.join(input_dir, "*.json"))
            output_files = [fn for fn in output_files if not fn.endswith("_.json")]
            output_files = sorted(
                output_files,
                key=lambda x: int(x.split("/")[-1].split("_")[-1].replace(".json", ""))
            )
            
            os.makedirs(os.path.join(args.output_dir, baseline_name, dataset_name), exist_ok=True)
            
            def process_file(file):
                save_fn = os.path.join(
                    args.output_dir, baseline_name, dataset_name,
                    file.split("/")[-1]
                )
                if os.path.exists(save_fn):
                    return
                
                # Load ledger data
                with open(file, "r") as f:
                    data = json.load(f)
                
                # Get answer
                if dataset_name == "livedrbench":
                    answer = livedrbench_answer_dict[data["question"]]
                    data["answer"] = answer
                else:
                    answer = data.get("answer", data.get("Answer", ""))
                
                # Check answer using appropriate checker (supports unknown baselines with TAO format)
                checker = get_checker(baseline_name, data)
                finished, is_correct = checker(data, answer)
                data["finished"] = finished
                data["is_correct"] = is_correct
                
                # Save result
                with open(save_fn, "w") as f:
                    json.dump(data, f, indent=2)
            
            with ThreadPool(processes=10) as pool:
                pool.map(process_file, output_files)


if __name__ == "__main__":
    args = get_args()
    main(args)
