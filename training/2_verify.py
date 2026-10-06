"""Step 2 (optional): label each rollout as correct / verified, so step 3 can keep only correct + verified runs.

For every rollout: build the Epistemic Ledger from its (thinking, query, result, next thinking) blocks with the
evaluator prompts in epistemic_ledger/prompts.py (constraints = the ones the rollout extracted), judge the answer with
the correctness prompt in epistemic_ledger/judge_prompts.py, and mark it verified when its final ledger has an active
candidate with every constraint established by evidence. Adds is_correct, verified, epistemic_ledger and
verification_result to each file.

NOTE: the released dayoon/LiveLedger-4B checkpoint was trained WITHOUT this step (step 3 keeps every successful
rollout when these fields are absent). Run it to apply the correct + verified filter.

    python training/2_verify.py -i rollouts -o rollouts_verified --base_url http://localhost:8000/v1
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "epistemic_ledger"))
from prompts import prompt_obj_ledger_update, prompt_per_ledger_update  # noqa: E402
from judge_prompts import PROMPT as CORRECTNESS_PROMPT  # noqa: E402
from metrics import LocalMinimaAccuracyEvaluator  # noqa: E402


def strip_reasoning(text: str) -> str:
    """Drop a reasoning preamble + code fences, leaving the JSON answer.

    Some reasoning models emit their chain-of-thought ending with a closing </think> tag (even
    when no opening tag is present) followed by the clean answer. Take everything
    after the last </think>, then strip markdown fences. Works for nested JSON too.
    """
    return text.split("</think>")[-1].replace("```json", "").replace("```", "").strip()


# =============================================================================
# Trajectory Block Extraction
# =============================================================================

def extract_blocks_from_messages(data: Dict) -> List[Tuple[str, str, str, str]]:
    """Extract (prev_thinking, query, result, next_thinking) blocks from messages.

    The assistant message content holds the turn's reasoning; the final block is the answer content.
    """
    messages = data["messages"]

    thinking_blocks = []
    query_blocks = []
    results_blocks = []

    # Skip system and first user message
    if messages and messages[0]["role"] == "system":
        messages = messages[1:]
    if messages and messages[0]["role"] == "user":
        messages = messages[1:]

    for idx, message in enumerate(messages):
        role = message["role"]
        content = message.get("content", "") or ""
        if role == "assistant":
            thinking_blocks.append(content)
            if "tool_calls" in message:
                tool_call = message["tool_calls"][0]
                func_name = tool_call["function"]["name"]
                if func_name == "search":
                    arguments = json.loads(tool_call["function"]["arguments"])
                    if "query" in arguments:
                        query = json.dumps(arguments["query"])
                    else:
                        query = json.dumps(arguments)
                    query_blocks.append("Search: " + query)
                elif func_name == "browse":
                    arguments = json.dumps(tool_call["function"]["arguments"])
                    query_blocks.append("Browse: " + arguments)
                else:
                    query_blocks.append("Invalid tool call")
                # If next message is not a tool response, add placeholder
                next_is_tool = (idx + 1 < len(messages) and messages[idx + 1]["role"] == "tool")
                if not next_is_tool:
                    results_blocks.append("No tool response found.")
        elif role == "tool":
            results_blocks.append(content)
        elif role == "user":
            continue

    if len(thinking_blocks) == len(query_blocks):
        thinking_blocks.append(data.get("content", ""))

    # Trim if needed
    min_len = min(len(query_blocks), len(results_blocks))
    query_blocks = query_blocks[:min_len]
    results_blocks = results_blocks[:min_len]
    thinking_blocks = thinking_blocks[:min_len + 1]

    # Build (prev_thinking, query, result, next_thinking) tuples
    prev_thinking_blocks = thinking_blocks[:-1]
    next_thinking_blocks = thinking_blocks[1:]

    return list(zip(prev_thinking_blocks, query_blocks, results_blocks, next_thinking_blocks))


# =============================================================================
# Verification Pipeline
# =============================================================================

class VerificationPipeline:
    """Epistemic Ledger + correctness + verified label for one rollout."""

    def __init__(self, model_name: str, base_url: str, api_key: str = "EMPTY",
                 enable_thinking: bool = True, show_progress: bool = False):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # Thinking stays ON by default (preserves judgment quality). Optionally
        # disable via --no_thinking when comparing speed/quality.
        self.extra_body = {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        # Per-item inner progress bars (only safe single-threaded, -w 1).
        self.show_progress = show_progress

    def call_llm(self, prompt: str, max_retries: int = 5) -> Dict:
        """Call LLM and parse JSON response (tolerates a reasoning preamble)."""
        for attempt in range(max_retries):
            output = ""
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    extra_body=self.extra_body,
                )
                output = response.choices[0].message.content
                parsed = json.loads(strip_reasoning(output))
                return parsed
            except Exception as e:
                tqdm.write(f"  [call_llm] PARSE FAIL: {e} (attempt {attempt + 1}/{max_retries})")
                if output:
                    tqdm.write(f"  [call_llm] stripped tail was: {strip_reasoning(output)[-300:]!r}")
                time.sleep(1)
        raise RuntimeError(f"call_llm failed after {max_retries} attempts")

    def call_llm_verdict(self, question: str, answer: str, predicted_answer: str,
                         max_retries: int = 5) -> bool:
        """Call LLM to judge answer correctness (tolerates reasoning preamble)."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": CORRECTNESS_PROMPT.format(
                            question=question,
                            answer=answer,
                            predicted_answer=predicted_answer
                        )
                    }],
                    max_tokens=8192,
                    extra_body=self.extra_body,
                )
                output = response.choices[0].message.content
                verdict = json.loads(strip_reasoning(output))["verdict"]
                if isinstance(verdict, bool):
                    return verdict
            except Exception as e:
                tqdm.write(f"  Verdict call error: {e} (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
        return False

    def build_epistemic_ledger(
        self,
        question: str,
        constraints_formatted: str,
        blocks: List[Tuple[str, str, str, str]],
    ) -> List[Dict]:
        """Build epistemic ledger with obj/per/status for each turn."""
        ledger_history = []
        current_ledger = {}

        turn_iter = blocks
        if self.show_progress:
            turn_iter = tqdm(blocks, desc="    ledger turns", unit="turn",
                             leave=False, position=1)
        for prev_thinking, query, result, next_thinking in turn_iter:
            # Step 1: Objective ledger update
            prompt = prompt_obj_ledger_update.format(
                current_ledger=json.dumps(current_ledger),
                prev_thinking=prev_thinking,
                constraints=constraints_formatted,
                query=query,
                result=result,
                next_thinking=next_thinking,
                question=question,
            )
            updated_obj_ledger = self.call_llm(prompt)

            # Step 2: Perception ledger update
            prompt = prompt_per_ledger_update.format(
                current_ledger=json.dumps(updated_obj_ledger),
                prev_thinking=prev_thinking,
                constraints=constraints_formatted,
                query=query,
                result=result,
                next_thinking=next_thinking,
                question=question,
            )
            updated_per_ledger = self.call_llm(prompt)

            current_ledger = updated_per_ledger
            ledger_history.append(updated_per_ledger)

        return ledger_history

    def check_correctness(self, data: Dict) -> Tuple[bool, bool]:
        """Check if prediction is correct. Returns (finished, is_correct)."""
        question = data["question"]
        answer = data.get("answer", "")
        prediction = data.get("prediction", "")

        # Fallback for empty prediction
        if not prediction or prediction == "exceeded max turns":
            content = data.get("content", "")
            if content:
                prediction = content
            else:
                return False, False

        is_correct = self.call_llm_verdict(question, str(answer), prediction)
        return True, is_correct

    def classify_verification(
        self,
        ledger_history: List[Dict],
        is_correct: bool,
        finished: bool,
        dataset_name: str,
    ) -> Dict[str, bool]:
        """Classify verified vs underverified using LocalMinimaAccuracyEvaluator."""
        evaluator = LocalMinimaAccuracyEvaluator(dataset_name, "training")
        result = evaluator.evaluate(
            ledgers=ledger_history,
            checklist=None,
            is_correct=is_correct,
        )
        return result

    def format_constraints_from_ledger(self, data: Dict) -> str:
        """Format constraints from the liveledger output's ledger constraints."""
        # Try to get from extract_response
        extract_resp = data.get("extract_response", {})
        constraints = extract_resp.get("constraints", [])

        if not constraints:
            # Fallback: extract from ledger keys
            ledger = data.get("ledger", {})
            if ledger:
                first_candidate = next(iter(ledger.values()), {})
                constraint_keys = first_candidate.get("constraints", {}).keys()
                constraints = list(constraint_keys)

        if not constraints:
            return ""

        # Format as "- C1: constraint_text" (use constraint index as name)
        lines = []
        for idx, c in enumerate(constraints):
            lines.append(f"- C{idx+1}: {c}")
        return "\n".join(lines)

# =============================================================================
# Processing
# =============================================================================

progress_lock = Lock()
correct_count = 0
verified_count = 0
error_count = 0
skip_count = 0


def process_single_item(
    input_path: str,
    output_path: str,
    pipeline: VerificationPipeline,
    dataset_name: str,
    pbar: tqdm,
):
    """Process a single inference output through verification pipeline."""
    global correct_count, verified_count, error_count, skip_count

    tqdm.write(f"[ITEM] start {os.path.basename(input_path)}")
    if os.path.exists(output_path):
        tqdm.write(f"[ITEM] {os.path.basename(input_path)} -> SKIP (output exists)")
        with progress_lock:
            skip_count += 1
            pbar.set_postfix(correct=correct_count, verified=verified_count, skip=skip_count, err=error_count)
            pbar.update(1)
        return

    try:
        with open(input_path, "r") as f:
            data = json.load(f)

        if data.get("status") != "success":
            tqdm.write(f"[ITEM] {os.path.basename(input_path)} -> SKIP (status={data.get('status')!r})")
            with progress_lock:
                skip_count += 1
                pbar.set_postfix(correct=correct_count, verified=verified_count, skip=skip_count, err=error_count)
                pbar.update(1)
            return

        # Step 1: Extract trajectory blocks
        blocks = extract_blocks_from_messages(data)
        tqdm.write(f"[ITEM] {os.path.basename(input_path)} -> {len(blocks)} blocks, calling LLM...")
        if not blocks:
            tqdm.write(f"[ITEM] {os.path.basename(input_path)} -> SKIP (no blocks)")
            with progress_lock:
                skip_count += 1
                pbar.set_postfix(correct=correct_count, verified=verified_count, skip=skip_count, err=error_count)
                pbar.update(1)
            return

        # Step 2: Build epistemic ledger
        constraints_formatted = pipeline.format_constraints_from_ledger(data)
        ledger_history = pipeline.build_epistemic_ledger(
            question=data["question"],
            constraints_formatted=constraints_formatted,
            blocks=blocks,
        )

        # Step 3: Check correctness
        finished, is_correct = pipeline.check_correctness(data)

        # Step 4: Classify verification
        verification_result = pipeline.classify_verification(
            ledger_history=ledger_history,
            is_correct=is_correct,
            finished=finished,
            dataset_name=dataset_name,
        )

        # Add results to data
        data["is_correct"] = is_correct
        data["verified"] = verification_result.get("Verified", False)
        data["epistemic_ledger"] = ledger_history
        data["verification_result"] = verification_result

        # Save
        temp_path = output_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.rename(temp_path, output_path)
        tqdm.write(f"[ITEM] {os.path.basename(input_path)} -> SAVED (is_correct={is_correct}, verified={data['verified']})")

        with progress_lock:
            if is_correct:
                correct_count += 1
            if verification_result.get("Verified", False):
                verified_count += 1
            pbar.set_postfix(correct=correct_count, verified=verified_count, skip=skip_count, err=error_count)
            pbar.update(1)

    except Exception as e:
        import traceback
        tqdm.write(f"[ITEM] {os.path.basename(input_path)} -> ERROR: {e}\n{traceback.format_exc()}")
        with progress_lock:
            error_count += 1
            pbar.set_postfix(correct=correct_count, verified=verified_count, skip=skip_count, err=error_count)
            pbar.update(1)


def main():
    global correct_count, verified_count, error_count, skip_count

    parser = argparse.ArgumentParser(description="Run verification pipeline on inference outputs")
    parser.add_argument("--input_dir", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--datasets", "-d", nargs="+", type=str, default=["hds-qa", "multiconir_books"])
    parser.add_argument("--num_workers", "-w", type=int, default=8)
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable model reasoning (default: thinking ON). Only for "
                             "speed/quality comparison; thinking stays on normally.")
    args = parser.parse_args()

    # Collect all tasks
    all_tasks = []
    for dataset_name in args.datasets:
        dataset_input_dir = os.path.join(args.input_dir, dataset_name)
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)

        if not os.path.exists(dataset_input_dir):
            print(f"Skipping {dataset_name}: input dir not found")
            continue

        os.makedirs(dataset_output_dir, exist_ok=True)

        input_files = glob(os.path.join(dataset_input_dir, "*.json"))
        input_files = [f for f in input_files if "summary" not in os.path.basename(f)]

        for input_path in input_files:
            idx = os.path.basename(input_path).replace(".json", "")
            output_path = os.path.join(dataset_output_dir, f"{idx}.json")
            all_tasks.append((input_path, output_path, dataset_name))

    total_count = len(all_tasks)
    correct_count = 0
    verified_count = 0
    error_count = 0
    skip_count = 0
    print(f"Total tasks: {total_count}")

    # Process with thread pool
    pbar = tqdm(total=total_count, desc="Verification", unit="item")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for input_path, output_path, dataset_name in all_tasks:
            # Each worker gets its own pipeline (own OpenAI client)
            pipeline = VerificationPipeline(
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key,
                enable_thinking=not args.no_thinking,
                show_progress=(args.num_workers == 1),
            )
            future = executor.submit(
                process_single_item,
                input_path, output_path, pipeline, dataset_name, pbar
            )
            futures[future] = input_path

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"Task failed: {futures[future]}: {e}")

    pbar.close()

    # Print summary
    print(f"\nCompleted: {total_count}")
    print(f"  Correct: {correct_count}, Verified: {verified_count}, Skipped: {skip_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
