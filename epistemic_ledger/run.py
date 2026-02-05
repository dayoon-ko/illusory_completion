import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Tuple

from openai import OpenAI
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from prompts import prompt_checklist_generation, prompt_obj_ledger_update, prompt_per_ledger_update


# =============================================================================
# Constants
# =============================================================================

# Known baselines with specific handling
KNOWN_BASELINES = [
    "search-r1", "rag-r1", "hds", "hds-grpo", "asearcher", "webexplorer", "dr-tulu",
    "tongyidr", "tongyidr-liveledger-20b", 
    "search_o1_gpt-oss-20b", "search_o1_gpt-oss-120b",
    "react_20b", "react_liveledger_20b",
    "react", "react_s1", "react_liveledger",
]

DATASET_CHOICES = ["all", "deepsearchqa", "browsecomp", "frames", "livedrbench", "webwalkerqa"]

# Baseline groups for block extraction (only for baselines requiring special handling)
BASELINES_RAW_TRAJECTORY = ["search-r1", "rag-r1", "hds", "hds-grpo"]
BASELINES_TAO_S1 = ["react_s1"]  # Filters invalid tool calls
BASELINES_TAOT = ["dr-tulu"]  # Pre-separated prev/next thinking


# =============================================================================
# JudgeAgent Class
# =============================================================================

class JudgeAgent:
    """
    A judge agent that evaluates answers based on constraint checklists
    and epistemic ledger tracking. Follows the same process as the existing
    checklist verification pipeline.
    """
    
    def __init__(
        self, 
        model_name: str = "openai/gpt-oss-120b",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY"
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        
        # State
        self.question = None
        self.checklist = None
        self.ledger = []
    
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
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                output = response.choices[0].message.content
                output = output.replace("```json", "").replace("```", "")
                print(output)
                return json.loads(output)
            except Exception as e:
                print(e)
                time.sleep(1)
    
    def generate_checklist(self, item: Dict, is_call_llm: bool = True) -> Dict[str, Dict[str, str]]:
        """Generate a constraint checklist from a question."""
        self.question = item["question"]
        
        if is_call_llm:
            prompt = prompt_checklist_generation.format(question=self.question)
            while True:
                checklist = self.call_llm(prompt)
                if all("constraint" in c.keys() for c in checklist.values()):
                    break 
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
        updated_obj_ledger = self.call_llm(prompt)
        
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
        updated_per_ledger = self.call_llm(prompt)
        
        self.ledger.append(updated_per_ledger)
        self.ledger_pairs.append((updated_obj_ledger, updated_per_ledger))
    
    def process_trajectory(self, blocks: List[tuple]) -> Tuple[List[Dict], List[Tuple[Dict, Dict]]]:
        """Process a full trajectory and build the ledger."""
        self.ledger = []
        self.ledger_pairs = []
        
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
    
    # 1. Handle baselines with special requirements
    if baseline_name in BASELINES_RAW_TRAJECTORY:
        # Raw trajectory string that needs parsing
        return get_blocks(output, baseline_name)
    
    if baseline_name in BASELINES_TAO_S1:
        # TAO format but filters invalid tool calls
        return get_tao_blocks_s1(output)
    
    if baseline_name in BASELINES_TAOT:
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

def process_item(item, model_name, baseline_name, output_path, max_turns=None, is_save_blocks=False):
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
    judge = JudgeAgent(model_name=model_name)
    
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
    
    with open(output_path, "w") as f:
        json.dump(item, f, indent=2)
    
    print(f"Saved {output_path}")
    
    return item


# =============================================================================
# Argument Parser
# =============================================================================

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--baseline_name", "-b",
        nargs="+", type=str,
        default=KNOWN_BASELINES,
        help=f"Baseline names. Known: {KNOWN_BASELINES}. Any baseline with standard TAO format is also supported."
    )
    parser.add_argument(
        "--dataset_name", "-d",
        nargs="+", type=str,
        default=["browsecomp", "frames", "livedrbench", "deepsearchqa", "webwalkerqa"],
        choices=DATASET_CHOICES
    )
    parser.add_argument("--num_try", type=int, default=0)
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--output_dir", "-o", type=str, default="output/epistemic_ledger")
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--max_turns", type=int, default=None)
    parser.add_argument("--is_save_blocks", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=8)
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Step 1: Collect all tasks from all baseline + dataset combinations
    all_tasks = []
    
    for baseline_name in args.baseline_name:
        for dataset_name in args.dataset_name:
            input_path = f"{args.input_dir}/{baseline_name}/{dataset_name}.jsonl"
            output_dir = f"{args.output_dir}/{baseline_name}/{dataset_name}"
            
            if args.num_try >= 1:
                input_path = input_path.replace(".jsonl", f"_{args.num_try}.jsonl")
                output_dir = output_dir + f"_{args.num_try}"
            
            if not os.path.exists(input_path):
                print(f"Skipping {input_path} because it does not exist")
                continue
            
            os.makedirs(output_dir, exist_ok=True)
            
            with open(input_path, "r") as f:
                data = [json.loads(line) for line in f.readlines()]
            
            for idx, item in enumerate(data):
                output_path = f"{output_dir}/item_{idx}.json"
                all_tasks.append({
                    "item": item,
                    "model_name": args.model_name,
                    "baseline_name": baseline_name,
                    "output_path": output_path,
                    "max_turns": args.max_turns,
                    "is_save_blocks": args.is_save_blocks
                })
    
    print(f"Total tasks collected: {len(all_tasks)}")
    
    # Step 2: Process all tasks with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_item,
                task["item"],
                task["model_name"],
                task["baseline_name"],
                task["output_path"],
                max_turns=task["max_turns"],
                is_save_blocks=task["is_save_blocks"]
            ): task for task in all_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing all datasets"):
            output = future.result()


if __name__ == "__main__":
    args = get_args()
    main(args)
