"""
Multi-turn agent with THREE-PHASE approach:
1. EXTRACT phase: Extract constraints from the question
2. SEARCH phase: Think and search for evidence
3. UPDATE phase: Update ledger based on search results

This architecture separates constraint extraction, reasoning/search, 
and ledger updates into distinct phases for better modularity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import ssl
import httpx
import urllib3
from openai import OpenAI

from tools import TOOLS_EXTRACT, TOOLS_SEARCH, TOOLS_UPDATE
from prompt import (
    SYSTEM_PROMPT_EXTRACT_CONSTRAINTS,
    SYSTEM_PROMPT_MAIN_W_LEDGER,
    SYSTEM_PROMPT_UPDATE_LEDGER
)
from search_engine import SerperSearchEngine, JinaBrowser
from utils import (
    parse_boxed, log_event,
    AgentStateMachine, EpistemicLedger,
    ANSI_PHASE, ANSI_THINK, ANSI_RESPONSE, ANSI_TOOL_CALL,
    ANSI_TOOL_MSG, ANSI_USER, ANSI_LEDGER
)

logger = logging.getLogger(__name__)

# Constants
MAX_RETRY_ATTEMPTS = 10
DEFAULT_TIMEOUT = 300.0
STAGNATION_THRESHOLD = 5
RESULT_PREVIEW_LENGTH = 500


class EpistemicAgentThreePhase:
    """
    Agent that uses THREE PHASES:
    1. EXTRACT phase: Extract constraints from question
    2. SEARCH phase: Think and search for evidence
    3. UPDATE phase: Update ledger based on latest search results
    """
    
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        search_engine: SerperSearchEngine,
        browser: JinaBrowser,
        max_turns: int = 50,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        reasoning_effort: str = "high",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.search_engine = search_engine
        self.browser = browser
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        
        # Initialize OpenAI client with SSL configuration
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=DEFAULT_TIMEOUT,
            http_client=httpx.Client(verify=ssl_context),
        )
    
    def _call_extract_constraints_phase(self, question: str) -> List[str]:
        """
        PHASE 1: Extract constraints from the question.
        
        Args:
            question: The question to extract constraints from
            
        Returns:
            List of constraint strings
        """
        log_event(
            "EXTRACT_CONSTRAINTS - PHASE",
            "=== EXTRACT PHASE: Parsing question into constraints ===",
            ANSI_PHASE
        )
        
        extract_prompt = SYSTEM_PROMPT_EXTRACT_CONSTRAINTS.format(question=question)
        
        messages = [
            {
                "role": "system",
                "content": "You are a constraint extraction assistant. Parse questions into atomic, verifiable constraints."
            },
            {"role": "user", "content": extract_prompt}
        ]
        
        # Retry loop until valid tool call
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=TOOLS_EXTRACT,
                )
                
                finish_reason = response.choices[0].finish_reason
                message = response.choices[0].message
                
                if finish_reason == "tool_calls":
                    tc = message.tool_calls[0]
                    if tc.function.name == "extract_constraints":
                        args = json.loads(tc.function.arguments)
                        constraints = args.get("constraints", [])
                        if constraints:
                            # Log extraction results
                            reasoning = getattr(message, 'reasoning_content', '') or ''
                            content = message.content or ''
                            
                            if reasoning:
                                log_event("EXTRACT_CONSTRAINTS - THINKING", reasoning, ANSI_THINK)
                            if content:
                                log_event("EXTRACT_CONSTRAINTS - RESPONSE", content, ANSI_RESPONSE)
                            
                            log_event(
                                "EXTRACT_CONSTRAINTS - TOOL_CALL",
                                f"extract_constraints(constraints={json.dumps(constraints, indent=2)})",
                                ANSI_TOOL_CALL
                            )
                            
                            return constraints
                            
                elif finish_reason == "stop":
                    content = message.content or ''
                    content = content.strip().strip("```json").strip("```").strip()
                    try:
                        tool_call = json.loads(content)
                        constraints = tool_call.get("constraints", [])
                        if constraints:
                            return constraints
                    except json.JSONDecodeError:
                        pass
                        
            except Exception as e:
                log_event(
                    "EXTRACT_CONSTRAINTS - RETRY",
                    f"Attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} failed: {e}",
                    ANSI_RESPONSE
                )
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    raise
        
        return []
    
    def _call_update_phase(
        self,
        question: str,
        ledger: EpistemicLedger,
        thinking: str,
        search_query: str,
        retrieval_results: str,
    ) -> List[Dict[str, Any]]:
        """
        PHASE 3: Update ledger based on search results.
        
        Args:
            question: Original question
            ledger: Current epistemic ledger
            thinking: Model's thinking from search phase
            search_query: The search query that was executed
            retrieval_results: Results from the search
            
        Returns:
            List of ledger entries to update
        """
        update_prompt = SYSTEM_PROMPT_UPDATE_LEDGER.format(
            question=question,
            constraints=ledger.format_constraints_for_update(),
            ledger=ledger.format_ledger_json(),
            thinking=thinking or "(No explicit thinking)",
            search_query=search_query,
            retrieval_results=retrieval_results
        )
        
        log_event(
            "UPDATE_LEDGER - PHASE",
            "=== UPDATE PHASE: Updating ledger based on search results ===",
            ANSI_PHASE,
            entire=True
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful and thorough ledger update assistant. "
                    "Given the constraints, the current ledger, the thinking, the search query, "
                    "and the retrieval results, analyze search results and update the epistemic "
                    "ledger of the candidates and constraints. Read the search results carefully, "
                    "and if any candidate has supported evidence for the constraints in the search "
                    "results, include the candidate and the evidence in the ledger accordingly.\n"
                    "Remember to output only the tool calls without any other text."
                )
            },
            {"role": "user", "content": update_prompt}
        ]
        
        # Retry loop until valid tool call
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=TOOLS_UPDATE,
                )
                
                finish_reason = response.choices[0].finish_reason
                message = response.choices[0].message
                
                if finish_reason == "tool_calls":
                    tc = message.tool_calls[0]
                    if tc.function.name == "update_ledger":
                        fn_args = json.loads(tc.function.arguments)
                        entries = fn_args.get("entries", [])
                        
                        # Log update results
                        reasoning = getattr(message, 'reasoning_content', '') or ''
                        content = message.content or ''
                        
                        if reasoning:
                            log_event("UPDATE_LEDGER - THINKING", reasoning, ANSI_THINK)
                        if content:
                            log_event("UPDATE_LEDGER - RESPONSE", content, ANSI_RESPONSE)
                        
                        log_event(
                            "UPDATE_LEDGER - TOOL_CALL",
                            f"Entries: {json.dumps(entries, indent=2)}",
                            ANSI_TOOL_CALL
                        )
                        
                        return entries
                        
                elif finish_reason == "stop":
                    content = message.content or ''
                    
                    # Try parsing as direct function call format
                    if 'update_ledger(entries=[' in content:
                        entries_str = content.strip().strip('update_ledger(entries=[').strip('])').strip()
                        entries = json.loads(entries_str)
                        if entries:
                            return entries
                    
                    # Try parsing as JSON
                    elif 'entries' in content:
                        content = content.strip().strip('```json').strip('```').strip()
                        entries = json.loads(content).get('entries', [])
                        if entries:
                            return entries
                
            except Exception as e:
                log_event(
                    "UPDATE_LEDGER - RETRY",
                    f"Attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} failed: {e}",
                    ANSI_RESPONSE
                )
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.error(f"Failed to update ledger after {MAX_RETRY_ATTEMPTS} attempts")
        
        return []
    
    def run(self, question: str) -> Tuple[str, List[Dict], str, int, float, Dict[str, Any]]:
        """
        Run agent on question with three-phase approach.
        
        Args:
            question: The question to answer
            
        Returns:
            Tuple of (content, messages, prediction, turns, latency, ledger)
        """
        ledger = EpistemicLedger()
        state_machine = AgentStateMachine()
        
        log_event("USER", question, ANSI_USER)
        
        start_time = time.time()
        
        # =====================================================================
        # PHASE 1: Extract constraints
        # =====================================================================
        constraints = self._call_extract_constraints_phase(question)
        ledger.set_constraints(constraints)
        state_machine.transition("extract_constraints")
        
        init_msg = f"✓ Initialized ledger with {len(ledger.constraints)} constraints:\n"
        init_msg += "\n".join(f"  {k}: {v}" for k, v in ledger.constraints.items())
        log_event("EXTRACT_CONSTRAINTS - RESULT", init_msg, ANSI_TOOL_MSG)
        log_event("STATE", state_machine.get_state_message(), ANSI_LEDGER)
        
        # =====================================================================
        # PHASE 2 & 3: Search and Update loop
        # =====================================================================
        latest_thinking = ""
        latest_query = ""
        latest_results = ""
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_MAIN_W_LEDGER},
            {"role": "user", "content": question},
        ]
        
        turn = 0
        
        while turn < self.max_turns:
            turn += 1
            
            log_event(
                "SEARCH - PHASE",
                f"=== SEARCH PHASE (Turn {turn}): Thinking and Searching ===",
                ANSI_PHASE
            )
            
            # Get model response
            response = self._get_search_response(messages)
            if response is None:
                break
            
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            reasoning = getattr(message, 'reasoning_content', '') or ''
            content = message.content or ''
            
            if reasoning:
                log_event("SEARCH - THINKING", reasoning, ANSI_THINK)
                latest_thinking = reasoning
            if content:
                log_event("SEARCH - RESPONSE", content, ANSI_RESPONSE)
            
            # Handle tool calls (search/browse)
            if finish_reason == "tool_calls" and message.tool_calls:
                tool_results = self._execute_tool_calls(
                    message.tool_calls,
                    state_machine,
                    ledger
                )
                
                # Check if we performed search/browse that needs ledger update
                needs_update = any(
                    tc.function.name in ["search", "browse"]
                    for tc in message.tool_calls
                )
                
                if needs_update:
                    # Extract latest query and results
                    for tc in message.tool_calls:
                        fn_name = tc.function.name
                        if fn_name in ["search", "browse"]:
                            fn_args = json.loads(tc.function.arguments)
                            if fn_name == "search":
                                queries = fn_args.get("query", [])
                                if isinstance(queries, str):
                                    queries = [queries]
                                latest_query = ", ".join(queries)
                            else:  # browse
                                urls = fn_args.get("urls", [])
                                if isinstance(urls, str):
                                    urls = [urls]
                                latest_query = f"browse: {', '.join(urls)}"
                            
                            # Get corresponding result
                            for result in tool_results:
                                if result["tool_call_id"] == tc.id:
                                    latest_results = result["content"]
                                    break
                
                # Update messages
                messages.append({
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls],
                })
                messages.extend(tool_results)
                
                # PHASE 3: Update ledger if we did a search/browse
                if needs_update and ledger.constraints:
                    entries = self._call_update_phase(
                        question=question,
                        ledger=ledger,
                        thinking=latest_thinking,
                        search_query=latest_query,
                        retrieval_results=latest_results,
                    )
                    
                    if entries:
                        ledger.reset_stagnation_count()
                        ledger.update(entries)
                        
                        is_complete, _ = ledger.check_completion()
                        if is_complete:
                            state_machine.set_complete()
                        
                        # Update messages with new ledger state
                        messages = [m for m in messages if m["role"] != "user"]
                        messages.append({
                            "role": "user",
                            "content": ledger.format_ledger()
                        })
                        
                        log_event("LEDGER - UPDATED", ledger.format_ledger(), ANSI_LEDGER)
                    else:
                        ledger.increase_stagnation_count()
                        log_event("LEDGER - NO UPDATES", ledger.format_ledger(), ANSI_LEDGER)
                    
                    # Handle stagnation
                    if ledger.get_stagnation_count() >= STAGNATION_THRESHOLD:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"No new candidates or evidence found in the last "
                                f"{STAGNATION_THRESHOLD} turns. If you are stuck, consider: "
                                "(1) using different keywords, (2) searching for specific facts, "
                                "or (3) trying alternative candidate answers."
                            )
                        })
                        ledger.reset_stagnation_count()
                        log_event(
                            "SEARCH - STAGNATION",
                            f"Resetting stagnation count after {STAGNATION_THRESHOLD} turns",
                            ANSI_LEDGER
                        )
                
                log_event("STATE", state_machine.get_state_message(), ANSI_LEDGER)
                continue
            
            # Check for final answer
            else:
                prediction = parse_boxed(content)
                latency = time.time() - start_time
                
                log_event("FINAL STATE", state_machine.get_state_message(), ANSI_LEDGER)
                log_event("FINAL LEDGER", ledger.format_ledger(), ANSI_LEDGER)
                
                return content, messages, prediction, turn, latency, ledger.ledger
        
        # Max turns exceeded
        latency = time.time() - start_time
        return content, messages, "exceeded max turns", turn, latency, ledger.ledger
    
    def _get_search_response(self, messages: List[Dict[str, Any]]) -> Optional[Any]:
        """Get response from model with retry logic."""
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=TOOLS_SEARCH,
                )
                
                message = response.choices[0].message
                content = message.content or ""
                finish_reason = response.choices[0].finish_reason
                
                if finish_reason == "tool_calls":
                    # Validate tool call format
                    tc = message.tool_calls[0]
                    json.loads(tc.function.arguments)
                    return response
                elif content and "query" not in content.lower():
                    # Valid final answer
                    return response
                    
            except Exception as e:
                log_event(
                    "SEARCH - RETRY",
                    f"Attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} failed: {e}",
                    ANSI_RESPONSE
                )
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.error(f"Failed to get search response after {MAX_RETRY_ATTEMPTS} attempts")
                    return None
        
        return None
    
    def _execute_tool_calls(
        self,
        tool_calls: List[Any],
        state_machine: AgentStateMachine,
        ledger: EpistemicLedger
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            
            log_event(
                "SEARCH - TOOL_CALL",
                f"{fn_name}({json.dumps(fn_args, indent=2)})",
                ANSI_TOOL_CALL
            )
            
            # Check state machine
            error = state_machine.transition(fn_name, ledger.check_completion()[0])
            if error:
                result = error
                log_event("SEARCH - STATE_ERROR", error, ANSI_LEDGER)
            
            # Execute tool
            elif fn_name == "search":
                queries = fn_args.get("query", [])
                if isinstance(queries, str):
                    queries = [queries]
                result = self.search_engine.search_batch(queries)
            
            elif fn_name == "browse":
                urls = fn_args.get("urls", [])
                if isinstance(urls, str):
                    urls = [urls]
                result = self.browser.browse_batch(urls)
            
            else:
                continue
            
            # Log result preview
            preview = result[:RESULT_PREVIEW_LENGTH]
            if len(result) > RESULT_PREVIEW_LENGTH:
                preview += "..."
            log_event("SEARCH - RESULT", preview, ANSI_TOOL_MSG)
            
            tool_results.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tc.id,
            })
        
        return tool_results


class DataLoader:
    """Loads and validates dataset from JSON/JSONL files."""
    
    def __init__(self, data_path: str, start_idx: int = 0, end_idx: Optional[int] = None):
        self.data_path = data_path
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and slice dataset."""
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r") as f:
                dataset = json.load(f)
        elif self.data_path.endswith(".jsonl"):
            with open(self.data_path, "r") as f:
                dataset = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file extension: {self.data_path}")
        
        # Slice dataset
        if self.start_idx is not None and self.end_idx is not None:
            dataset = dataset[self.start_idx:self.end_idx]
        elif self.start_idx is not None:
            dataset = dataset[self.start_idx:]
        elif self.end_idx is not None:
            dataset = dataset[:self.end_idx]
        
        # Validate each datapoint
        dataset = [self.validate_datapoint(item) for item in dataset]
        return dataset
    
    def validate_datapoint(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure datapoint has required fields."""
        # Normalize question field
        if "question" not in item:
            if "Question" in item:
                item["question"] = item["Question"]
                del item["Question"]
            else:
                raise ValueError(f"Question not found in item: {item}")
        
        # Normalize answer field
        if "answer" not in item:
            if "Answer" in item:
                item["answer"] = item["Answer"]
                del item["Answer"]
            elif "ground_truths" in item:
                item["answer"] = {
                    "ground_truths": item["ground_truths"],
                    "misc": item.get("misc"),
                    "canary": item.get("canary"),
                    "key": item.get("key")
                }
            else:
                raise ValueError(f"Answer not found in item: {item}")
        
        return item


# Thread-safe progress tracking
progress_lock = Lock()
completed_count = 0
total_count = 0


def process_item(
    idx: int,
    item: Dict[str, Any],
    agent: EpistemicAgentThreePhase,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    """
    Process a single dataset item (thread-safe worker function).
    
    Args:
        idx: Item index
        item: Dataset item
        agent: Agent instance
        output_dir: Output directory
        
    Returns:
        Result dictionary or None if skipped
    """
    global completed_count
    
    output_path = os.path.join(output_dir, f"{idx}.json")
    
    # Skip if already processed (resume capability)
    if os.path.exists(output_path):
        with progress_lock:
            completed_count += 1
            print(f"[{completed_count}/{total_count}] Skipping {idx} (already exists)")
        return None
    
    question = item["question"]
    answer = item["answer"]
    
    try:
        content, messages, prediction, turns, latency, ledger = agent.run(question)
        
        result = {
            "question": question,
            "answer": answer,
            "content": content,
            "messages": messages,
            "prediction": prediction,
            "turns": turns,
            "latency": latency,
            "ledger": ledger,
            "status": "success",
        }
        
        # Write output atomically (write to temp then rename)
        temp_path = output_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.rename(temp_path, output_path)
        
        with progress_lock:
            completed_count += 1
            print(f"\n{'='*60}")
            print(f"[{completed_count}/{total_count}] Completed item {idx}")
            print(f"Question: {question[:100]}...")
            print(f"Prediction: {prediction}")
            print(f"Turns: {turns}, Time: {latency:.1f}s")
            print(f"Saved to {output_path}")
        
        return result
        
    except Exception as e:
        logger.exception(f"Error processing item {idx}")
        
        error_result = {
            "question": question,
            "answer": answer,
            "error": str(e),
            "status": "error",
        }
        
        # Save error result
        error_path = output_path + ".error"
        with open(error_path, "w") as f:
            json.dump(error_result, f, indent=2)
        
        with progress_lock:
            completed_count += 1
            print(f"\n[{completed_count}/{total_count}] ERROR on item {idx}: {e}")
        
        return error_result


def main():
    """Main entry point."""
    global total_count, completed_count
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    urllib3.disable_warnings()
    
    parser = argparse.ArgumentParser(description="Run epistemic agent on datasets")
    parser.add_argument("--model_name", default="openai/gpt-oss-120b")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--serper_api_key", default=os.getenv("SERPER_API_KEY", ""))
    parser.add_argument("--jina_api_key", default=os.getenv("JINA_API_KEY", ""))
    parser.add_argument("--reasoning_effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--max_turns", type=int, default=100)
    
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument(
        "--dataset_names", "-d",
        nargs="+",
        type=str,
        default=["frames"],
        choices=["browsecomp", "frames", "deepsearchqa", "livedrbench", "webwalkerqa"]
    )
    parser.add_argument("--start_idx", "-s", type=int, default=0)
    parser.add_argument("--end_idx", "-e", type=int, default=None)
    
    parser.add_argument("--output_dir", "-o", type=str, default="outputs")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing"
    )
    args = parser.parse_args()
    
    # Process each dataset
    for dataset_name in args.dataset_names:
        dataset_dir = os.path.join(args.dataset_dir, dataset_name)
        data_path = os.path.join(dataset_dir, "test_mcqa.jsonl")
        
        dataset = DataLoader(data_path, args.start_idx, args.end_idx).load_data()
        
        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        total_count = len(dataset)
        completed_count = 0
        
        print(f"{'='*60}")
        print(f"Processing {dataset_name}: {total_count} items with {args.num_workers} worker(s)")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        # Create agent factory
        def create_agent():
            return EpistemicAgentThreePhase(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                search_engine=SerperSearchEngine(serper_api_key=args.serper_api_key),
                browser=JinaBrowser(jina_api_key=args.jina_api_key),
                max_turns=args.max_turns,
                reasoning_effort=args.reasoning_effort,
            )
        
        start_time = time.time()
        
        if args.num_workers == 1:
            # Sequential processing
            agent = create_agent()
            results = []
            for idx, item in enumerate(dataset):
                result = process_item(idx, item, agent, output_dir)
                results.append(result)
        else:
            # Multi-threaded processing
            agents = [create_agent() for _ in range(args.num_workers)]
            
            results = []
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                # Submit all tasks
                futures = {}
                for idx, item in enumerate(dataset):
                    agent = agents[idx % args.num_workers]
                    future = executor.submit(process_item, idx, item, agent, output_dir)
                    futures[future] = idx
                
                # Collect results as they complete
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.exception(f"Future for item {idx} raised exception")
                        results.append({"idx": idx, "status": "error", "error": str(e)})
        
        total_time = time.time() - start_time
        
        # Summary statistics
        successful = sum(1 for r in results if r and r.get("status") == "success")
        errors = sum(1 for r in results if r and r.get("status") == "error")
        skipped = sum(1 for r in results if r is None)
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: {dataset_name}")
        print(f"{'='*60}")
        print(f"Total items: {total_count}")
        print(f"Successful: {successful}")
        print(f"Errors: {errors}")
        print(f"Skipped (already existed): {skipped}")
        print(f"Total time: {total_time:.1f}s")
        if successful > 0:
            print(f"Average time per successful item: {total_time / successful:.1f}s")
        print()


if __name__ == "__main__":
    main()