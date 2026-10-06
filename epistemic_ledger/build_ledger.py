"""Build the Epistemic Ledger for agent trajectories (paper setting: gpt-5-nano, reasoning effort medium, first 30 turns).

Input  (--input_dir), either layout per agent:
  <input_dir>/<agent>/<dataset>/<index>.json   output of liveledger/run.py (converted on the fly)
  <input_dir>/<agent>/<dataset>.jsonl          one trajectory per line, "output" = {thinking_blocks, query_blocks,
                                               results_blocks} (or a raw Search-R1 / RAG-R1 string, see below)
Output: <output_dir>/<agent>/<dataset>/item_<index>.json   (input fields + checklist, ledger, ledger_pairs).
Finished items are skipped, so the same command resumes.

    export OPENAI_API_KEY=...
    python epistemic_ledger/build_ledger.py -i outputs -a gpt-oss-120b_liveledger -o ledgers
"""
import copy
import json
import os
import re
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI
from tqdm import tqdm

from prompts import prompt_checklist_generation, prompt_obj_ledger_update, prompt_per_ledger_update

DATASETS = ["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa", "bioasq"]

# Agents whose saved trajectories need special block extraction (matched on the agent name, run suffix _2/_3 ignored)
BASELINES_RAW_TRAJECTORY = ["search-r1", "rag-r1", "hds", "hds-grpo"]   # one raw string with <think>/<search>/... tags
BASELINES_TAO_S1 = ["react_s1"]  # filters invalid tool calls
BASELINES_TAOT = ["dr-tulu"]  # pre-separated prev/next thinking


def parse_trailing_json(output: str):
    """Some servers put commentary prose in front of the final JSON in `content`. Accept a JSON object only if it
    runs to the END of the content, else raise so the call is retried."""
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        s = output.rstrip()
        dec = json.JSONDecoder()
        for i, ch in enumerate(s):
            if ch == "{":
                try:
                    obj, end = dec.raw_decode(s, i)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and s[end:].strip() == "":
                    PARSE_STATS["trailing_json_recovered"] += 1
                    return obj
        PARSE_STATS["unparseable"] += 1
        raise


PARSE_STATS = {"trailing_json_recovered": 0, "unparseable": 0}


# =============================================================================
# Ledger evaluator
# =============================================================================

class LedgerEvaluator:
    """Builds the Epistemic Ledger of one trajectory: a constraint checklist from the question, then, for every
    (thinking, query, result, next thinking) block, an obj pass (what the retrieved text establishes) and a per pass
    (what the agent states it believes)."""
    
    def __init__(
        self,
        model_name: str = "gpt-5-nano",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        reasoning_effort: Optional[str] = None,
        max_retries: int = 0
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        # 0 = retry forever (fine for a local vLLM). Keep > 0 for a paid API, so an
        # unparseable-response loop cannot bill forever.
        self.max_retries = max_retries
        self._client = None
        
        # State
        self.question = None
        self.checklist = None
        self.ledger = []
        self.update_failures = []
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM with retry logic."""
        extra_body = {}
        if self.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.reasoning_effort
        attempt = 0
        rate_waits = 0
        while True:
            response = None
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body=extra_body if extra_body else None
                )
                output = response.choices[0].message.content
                output = output.replace("```json", "").replace("```", "")
                parsed = parse_trailing_json(output)
                if not isinstance(parsed, dict):
                    # checklist and ledgers must be JSON objects (a list breaks the next turn's ledger)
                    raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
                return parsed
            except openai.RateLimitError as e:
                # rate limits (HTTP 429): back off, do not count as a retry
                rate_waits += 1
                print(f"RateLimitError (wait {rate_waits}): {str(e)[:160]}")
                time.sleep(min(5 * rate_waits, 60))
                continue
            except Exception as e:
                attempt += 1
                try:
                    _c = response.choices[0].message.content
                    print(f"{type(e).__name__}: {e} | finish={response.choices[0].finish_reason} | content[:160]={(_c or '')[:160]!r} | tail={(_c or '')[-80:]!r}")
                except Exception:
                    print(e)
                if self.max_retries and attempt >= self.max_retries:
                    raise RuntimeError(
                        f"call_llm failed {attempt}x (last: {type(e).__name__}: {e})") from e
                time.sleep(min(1 * attempt, 10))
    
    def generate_checklist(self, item: Dict, is_call_llm: bool = True) -> Dict[str, Dict[str, str]]:
        """Generate a constraint checklist from a question."""
        self.question = item["question"]
        
        if is_call_llm:
            prompt = prompt_checklist_generation.format(question=self.question)
            attempt = 0
            while True:
                checklist = self.call_llm(prompt)
                if all(isinstance(c, dict) and "constraint" in c for c in checklist.values()):
                    break
                attempt += 1
                if self.max_retries and attempt >= self.max_retries:
                    raise RuntimeError(
                        f"checklist generation malformed {attempt}x: {checklist}")
                print(f"Error: Checklist generation failed. Retrying... {checklist}")
            self.checklist = {"candidate": None, "checklist": checklist}
        else:
            constraints = list(item["constraints"]["entities"].values())[0]["constraints"]
            checklist = {f"constraint_{idx+1}": {"constraint": c} for idx, c in enumerate(constraints)}
            self.checklist = {"candidate": None, "checklist": checklist}
        
        return self.checklist
    
    def format_constraints(self) -> str:
        """Format constraints for prompt injection."""
        if not self.checklist:
            return ""
        constraints = self.checklist["checklist"].values()
        return "\n".join([
            f"- C{idx+1}: {constraint['constraint']}" 
            for idx, constraint in enumerate(constraints)
        ])
    
    def update_ledger(
        self,
        prev_thinking: str,
        query: str,
        result: str,
        next_thinking: str
    ) -> Dict:
        """Update the epistemic ledger with a new trajectory segment."""
        current_ledger = self.ledger[-1] if len(self.ledger) > 0 else {}
        
        # Step 1: Update objective ledger
        prompt = prompt_obj_ledger_update.format(
            current_ledger=json.dumps(current_ledger),
            prev_thinking=prev_thinking,
            constraints=self.format_constraints(),
            query=query,
            result=result,
            next_thinking=next_thinking,
            question=self.question
        )
        try:
            updated_obj_ledger = self.call_llm(prompt)
        except RuntimeError as e:
            # retries exhausted (e.g. the model answers in prose instead of JSON): keep the previous ledger for this
            # block instead of dropping the whole item; recorded in item["ledger_update_failures"]
            self.update_failures.append({"block": len(self.ledger), "step": "obj", "error": str(e)[:300]})
            updated_obj_ledger = copy.deepcopy(current_ledger)
        
        # Step 2: Update perceptual ledger
        prompt = prompt_per_ledger_update.format(
            current_ledger=json.dumps(updated_obj_ledger),
            prev_thinking=prev_thinking,
            constraints=self.format_constraints(),
            query=query,
            result=result,
            next_thinking=next_thinking,
            question=self.question
        )
        try:
            updated_per_ledger = self.call_llm(prompt)
        except RuntimeError as e:
            self.update_failures.append({"block": len(self.ledger), "step": "per", "error": str(e)[:300]})
            updated_per_ledger = copy.deepcopy(updated_obj_ledger)
        
        self.ledger.append(updated_per_ledger)
        self.ledger_pairs.append((updated_obj_ledger, updated_per_ledger))
    
    def process_trajectory(self, blocks: List[tuple]) -> Tuple[List[Dict], List[Tuple[Dict, Dict]]]:
        """Process a full trajectory and build the ledger."""
        self.ledger = []
        self.ledger_pairs = []
        self.update_failures = []
        
        for idx, (prev_thinking, query, result, next_thinking) in enumerate(blocks):
            self.update_ledger(
                prev_thinking=prev_thinking,
                query=query,
                result=result,
                next_thinking=next_thinking
            )
    
    def get_active_candidates(self) -> Optional[str]:
        """Get the current active candidate from the ledger."""
        if not self.ledger:
            return None
        return {k: v for k, v in self.ledger[-1].items() if v.get("status") == "active"}
    
    def get_candidate_status(self, candidate: str) -> Optional[Dict]:
        """Get the constraint status for a specific candidate."""
        if not self.ledger:
            return None
        ledger_data = self.ledger[-1]
        return ledger_data.get(candidate).get("status")
    
    def get_all_candidates(self) -> Dict[str, str]:
        """Get all candidates and their statuses."""
        if not self.ledger:
            return {}
        ledger_data = self.ledger[-1]
        return {k: v.get("status") for k, v in ledger_data.items()}
    
    def reset(self):
        """Reset the agent state."""
        self.question = None
        self.checklist = None
        self.ledger = []


# =============================================================================
# Block Extraction Functions
# =============================================================================

def get_blocks(trajectory, baseline_name):
    """Extract blocks from raw trajectory string (search-r1, rag-r1, hds, hds-grpo)."""
    
    # Strip prefix if present
    if "<|im_start|>assistant" in trajectory:
        start_idx = trajectory.index("<|im_start|>assistant")
        trajectory = trajectory[start_idx:].strip()
    
    # Define patterns based on baseline
    if baseline_name in ["search-r1", "rag-r1"]:
        think_pattern = r"<think>(.*?)</think>"
        query_pattern = r"<search>(.*?)</search>"
        result_pattern = r"<information>(.*?)</information>"
        answer_pattern = r"<answer>(.*?)</answer>"
    elif baseline_name in ["hds", "hds-grpo"]:
        think_pattern = r"<think>(.*?)</think>"
        query_pattern = r"<\|begin_search_queries\|>(.*?)<\|end_search_queries\|>"
        result_pattern = r"<\|begin_search_results\|>(.*?)<\|end_search_results\|>"
        answer_pattern = r"\\boxed{(.*?)}"
    else:
        raise ValueError(f"Invalid baseline name: {baseline_name}")
    
    # Find all matches with positions
    all_blocks = []
    
    for match in re.finditer(think_pattern, trajectory, re.DOTALL):
        all_blocks.append((match.start(), "think", match.group(1).strip()))
    
    for match in re.finditer(query_pattern, trajectory, re.DOTALL):
        all_blocks.append((match.start(), "query", match.group(1).strip()))
    
    for match in re.finditer(result_pattern, trajectory, re.DOTALL):
        if baseline_name == "rag-r1":
            result = json.loads(match.group(1).strip())
            result = [i for i in result.values() if len(i) > 0]
            result = "\n".join(result) if len(result) > 0 else "No results found"
        all_blocks.append((match.start(), "result", match.group(1).strip()))
    
    # Find answer block
    answer_blocks = re.findall(answer_pattern, trajectory, re.DOTALL)
    answer_block = f"\nprediction: {answer_blocks[-1]}" if answer_blocks else ""
    
    # Sort by position
    all_blocks.sort(key=lambda x: x[0])
    
    # Build the sequence with proper ordering
    sequence = []
    expected_order = ["think", "query", "result"]
    expected_idx = 0
    
    for pos, block_type, content in all_blocks:
        expected_type = expected_order[expected_idx]
        
        if block_type == expected_type:
            sequence.append((block_type, content))
            expected_idx = (expected_idx + 1) % 3
        elif block_type in expected_order:
            block_idx = expected_order.index(block_type)
            while expected_idx != block_idx:
                sequence.append((expected_order[expected_idx], ""))
                expected_idx = (expected_idx + 1) % 3
            sequence.append((block_type, content))
            expected_idx = (expected_idx + 1) % 3
    
    # Extract blocks from sequence
    thinking_blocks = [content for btype, content in sequence if btype == "think"]
    query_blocks = [content for btype, content in sequence if btype == "query"]
    results_blocks = [content for btype, content in sequence if btype == "result"]
    
    # Handle answer block
    if answer_block:
        if len(thinking_blocks) == len(results_blocks):
            thinking_blocks.append(answer_block)
        elif len(thinking_blocks) == len(results_blocks) + 1:
            thinking_blocks[-1] += answer_block
        else:
            while len(thinking_blocks) < len(results_blocks) + 1:
                thinking_blocks.append("")
            thinking_blocks[-1] += answer_block
    
    prev_thinking_blocks = thinking_blocks[:-1]
    next_thinking_blocks = thinking_blocks[1:]
    
    return list(zip(prev_thinking_blocks, query_blocks, results_blocks, next_thinking_blocks))


def get_tao_blocks(blocks):
    """Extract TAO (Thinking-Action-Observation) blocks from structured output."""
    thinking_blocks = blocks["thinking_blocks"]
    query_blocks = blocks["query_blocks"]
    result_blocks = blocks["results_blocks"]
    
    prev_thinking_blocks = thinking_blocks[:-1]
    next_thinking_blocks = thinking_blocks[1:]
    return list(zip(prev_thinking_blocks, query_blocks, result_blocks, next_thinking_blocks))


def get_truncated_tao_blocks(blocks, max_para_length=3):
    """Extract TAO blocks with truncated thinking (first + last paragraphs)."""
    thinking_blocks = blocks["thinking_blocks"]
    thinking_blocks = [
        "\n\n".join(t.split("\n\n")[:max_para_length] + t.split("\n\n")[-2:])
        for t in thinking_blocks
    ]
    query_blocks = blocks["query_blocks"]
    result_blocks = blocks["results_blocks"]
    
    prev_thinking_blocks = thinking_blocks[:-1]
    next_thinking_blocks = thinking_blocks[1:]
    return list(zip(prev_thinking_blocks, query_blocks, result_blocks, next_thinking_blocks))


def get_taot_blocks(blocks):
    """Extract TAOT blocks (pre-separated prev/next thinking)."""
    prev_thinking_blocks = blocks["prev_thinking_blocks"]
    query_blocks = blocks["query_blocks"]
    result_blocks = blocks["results_blocks"]
    next_thinking_blocks = blocks["next_thinking_blocks"]
    return list(zip(prev_thinking_blocks, query_blocks, result_blocks, next_thinking_blocks))


def get_tao_blocks_s1(blocks):
    """Extract TAO blocks, filtering out invalid tool calls."""
    thinking_blocks = blocks["thinking_blocks"]
    query_blocks = blocks["query_blocks"]
    result_blocks = blocks["results_blocks"]
    
    prev_thinking_blocks = thinking_blocks[:-1]
    next_thinking_blocks = thinking_blocks[1:]
    blocks_to_return = list(zip(prev_thinking_blocks, query_blocks, result_blocks, next_thinking_blocks))
    blocks_to_return = [i for i in blocks_to_return if i[1] != "Invalid tool call"]
    return blocks_to_return


def save_blocks(item, blocks):
    """Save extracted blocks back to item."""
    item["thinking_blocks"] = []
    item["query_blocks"] = []
    item["result_blocks"] = []
    
    for prev_thinking, query, result, next_thinking in blocks:
        item["thinking_blocks"].append(prev_thinking)
        item["query_blocks"].append(query)
        item["result_blocks"].append(result)
    item["thinking_blocks"].append(blocks[-1][-1])
    
    return item


def has_standard_tao_format(output):
    """Check if output has standard TAO format (thinking_blocks, query_blocks, results_blocks)."""
    if not isinstance(output, dict):
        return False
    required_keys = ["thinking_blocks", "query_blocks", "results_blocks"]
    return all(key in output for key in required_keys)


def has_taot_format(output):
    """Check if output has TAOT format (prev/next thinking pre-separated)."""
    if not isinstance(output, dict):
        return False
    required_keys = ["prev_thinking_blocks", "query_blocks", "results_blocks", "next_thinking_blocks"]
    return all(key in output for key in required_keys)


def extract_blocks_for_baseline(item, baseline_name):
    """Extract blocks based on baseline type or data format.
    
    Priority:
    1. Baseline-specific handling for known baselines with special requirements
    2. Standard TAO format detection (thinking_blocks, query_blocks, results_blocks)
    3. TAOT format detection (prev/next thinking pre-separated)
    4. Raw trajectory parsing for legacy baselines
    """
    output = item["output"]
    base_name = re.sub(r"_[23]$", "", baseline_name)

    # 1. Handle baselines with special requirements
    if base_name in BASELINES_RAW_TRAJECTORY:
        # Raw trajectory string that needs parsing
        return get_blocks(output, base_name)

    if base_name in BASELINES_TAO_S1:
        # TAO format but filters invalid tool calls
        return get_tao_blocks_s1(output)

    if base_name in BASELINES_TAOT:
        # Pre-separated prev/next thinking blocks
        return get_taot_blocks(output)
    
    # 2. Auto-detect format for other baselines (including unknown ones)
    if has_taot_format(output):
        return get_taot_blocks(output)
    
    if has_standard_tao_format(output):
        return get_tao_blocks(output)
    
    # 3. If raw string, try to infer baseline type or fail gracefully
    if isinstance(output, str):
        raise ValueError(
            f"Baseline '{baseline_name}' has raw trajectory output but no parsing rules defined. "
            f"Please add it to BASELINES_RAW_TRAJECTORY or convert output to standard TAO format."
        )
    
    raise ValueError(
        f"Unable to extract blocks for baseline '{baseline_name}'. "
        f"Output must have 'thinking_blocks', 'query_blocks', 'results_blocks' keys, "
        f"or baseline must be in BASELINES_RAW_TRAJECTORY with parsing rules defined."
    )


# =============================================================================
# Processing Functions
# =============================================================================

def process_item(item, model_name, baseline_name, output_path, base_url="https://api.openai.com/v1", max_turns=None, is_save_blocks=False, reasoning_effort=None, api_key="EMPTY", max_retries=0):
    """Process a single item: generate checklist and build ledger."""
    
    # Validation
    if item.get("question", "") == "":
        print(f"Skipping {output_path} because question is empty")
        return
    
    if os.path.exists(output_path):
        print(f"Skipping {output_path} because it already exists")
        return
    
    print(f"Processing {output_path}")
    
    # Initialize judge agent
    judge = LedgerEvaluator(model_name=model_name, base_url=base_url, reasoning_effort=reasoning_effort,
                       api_key=api_key, max_retries=max_retries)
    
    # Step 1: Generate checklist
    judge.generate_checklist(item)
    
    # Step 2: Extract trajectory blocks
    blocks = extract_blocks_for_baseline(item, baseline_name)
    
    if max_turns is not None:
        blocks = blocks[:max_turns]
    
    # Step 3: Process trajectory with judge
    judge.process_trajectory(blocks)
    
    # Step 4: Save results
    if is_save_blocks:
        item = save_blocks(item, blocks)
    
    item["checklist"] = judge.checklist
    item["ledger"] = judge.ledger
    item["ledger_pairs"] = judge.ledger_pairs
    if judge.update_failures:
        item["ledger_update_failures"] = judge.update_failures
    
    with open(output_path, "w") as f:
        json.dump(item, f, indent=2)
    
    print(f"Saved {output_path}")
    
    return item


# =============================================================================
# liveledger/run.py output -> blocks
# =============================================================================

def _think(rec):
    parts = [rec.get("reasoning") or "", rec.get("content") or ""]
    return "\n\n".join(p for p in parts if p.strip())


def convert_runner_output(d):
    """One (thinking, query, result) block per tool call; the first tool call of an assistant turn carries that turn's
    reasoning + content, and the last thinking block is the final answer content."""
    thinking, queries, results = [], [], []
    final_think = ""
    for rec in d["turn_records"]:
        if rec["kind"] == "tool_call":
            for j, (tc, res) in enumerate(zip(rec["tool_calls"], rec["tool_results"])):
                thinking.append(_think(rec) if j == 0 else "")
                if tc["name"] == "search":
                    queries.append("Search: " + json.dumps(tc["arguments"].get("query", tc["arguments"])))
                elif tc["name"] == "browse":
                    queries.append("Browse: " + json.dumps(json.dumps(tc["arguments"])))
                else:
                    queries.append("Invalid tool call")
                results.append(res)
        elif rec["kind"] in ("answer", "forced_final"):
            final_think = rec.get("content") or ""
    thinking.append(final_think)
    return {"thinking_blocks": thinking, "query_blocks": queries, "results_blocks": results}


def runner_item(raw):
    return {"question": raw["question"], "answer": raw["answer"], "content": raw.get("content", "") or "",
            "prediction": raw.get("prediction", ""), "messages": raw.get("messages", []), "turns": raw.get("turns"),
            "termination": raw.get("termination"), "status": raw.get("status"), "dataset": raw.get("dataset"),
            "index": raw.get("index"), "output": convert_runner_output(raw)}


def load_items(input_dir, agent, dataset):
    """[(index, item)] from either input layout."""
    per_item_dir = os.path.join(input_dir, agent, dataset)
    jsonl = os.path.join(input_dir, agent, f"{dataset}.jsonl")
    if os.path.isdir(per_item_dir):
        out = []
        for p in glob(os.path.join(per_item_dir, "*.json")):
            name = os.path.basename(p)[:-5]
            if name.isdigit():
                out.append((int(name), runner_item(json.load(open(p)))))
        return sorted(out, key=lambda x: x[0])
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            return [(i, json.loads(line)) for i, line in enumerate(f.read().split("\n")) if line.strip()]
    return []


# =============================================================================
# Main
# =============================================================================

def get_args():
    p = ArgumentParser()
    p.add_argument("--input_dir", "-i", required=True)
    p.add_argument("--agents", "-a", nargs="+", required=True, help="sub-folder names under --input_dir")
    p.add_argument("--datasets", "-d", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--output_dir", "-o", default="ledgers")
    p.add_argument("--model_name", default="gpt-5-nano")
    p.add_argument("--base_url", default="https://api.openai.com/v1")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--reasoning_effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--max_retries", type=int, default=5,
                   help="per-call retry cap; 0 = unlimited (only for a local vLLM server)")
    p.add_argument("--max_turns", type=int, default=30, help="blocks evaluated per trajectory")
    p.add_argument("--max_workers", "-w", type=int, default=64)
    return p.parse_args()


def main(args):
    tasks = []
    for agent in args.agents:
        for ds in args.datasets:
            items = load_items(args.input_dir, agent, ds)
            if not items:
                print(f"no trajectories for {agent}/{ds}")
                continue
            out_dir = os.path.join(args.output_dir, agent, ds)
            os.makedirs(out_dir, exist_ok=True)
            tasks += [(item, agent, os.path.join(out_dir, f"item_{idx}.json")) for idx, item in items]
    print(f"{len(tasks)} trajectories")
    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(process_item, item, args.model_name, agent, path, base_url=args.base_url,
                             max_turns=args.max_turns, reasoning_effort=args.reasoning_effort,
                             api_key=args.api_key, max_retries=args.max_retries): path
                   for item, agent, path in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                fut.result()
            except Exception as e:  # one failed item must not stop the run
                n_fail += 1
                print(f"FAILED {futures[fut]}: {type(e).__name__}: {e}")
    if n_fail:
        print(f"{n_fail}/{len(futures)} items failed; re-run the same command to retry them.")
    print("PARSE_STATS", PARSE_STATS)


if __name__ == "__main__":
    main(get_args())
