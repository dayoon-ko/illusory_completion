"""Step 1: collect LiveLedger rollouts with gpt-oss-120b on the tracker-training questions (HDS-QA, MultiConIR-books).

The same model plays both roles: it extracts the constraints, searches (search / browse), and after every search
updates the ledger with an update_ledger tool call. Every extraction and every ledger update is saved, so step 3 can
turn them into SFT examples for the 4B tracker.

    vllm serve openai/gpt-oss-120b --port 8000 --tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser openai
    export SERPER_API_KEY=... JINA_API_KEY=...
    python training/1_collect_rollouts.py -o rollouts

Input : <dataset_dir>/<dataset>/test_mcqa.jsonl (fields: question, answer)
Output: <output_dir>/<dataset>/<index>.json (messages, extract_response, update_responses, prediction, ...)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ssl
import httpx
import requests
import urllib3
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "liveledger"))
from tools import TOOLS_EXTRACT, TOOLS_SEARCH, TOOLS_UPDATE  # noqa: E402
from prompts import SYSTEM_PROMPT_EXTRACT_CONSTRAINTS, SYSTEM_PROMPT_UPDATE_LEDGER  # noqa: E402
from search import SerperSearchEngine, JinaBrowser  # noqa: E402
from rollout_utils import *  # noqa: E402,F401,F403

# System prompt of the search model during rollout collection (kept verbatim, including the doubled braces).
SYSTEM_PROMPT_MAIN_W_LEDGER = r"""
You are a reasoning assistant that answers multi-constraint questions. You will work with a ledger system that tracks your verification progress.

## How This Works

1. You will receive a question that requires satisfying multiple constraints.
2. You search for information using `search` or `browse` tools.
3. After each search, an evaluator analyzes the results. If it has any updates, it will provide you with an updated **ledger**. 
4. The ledger shows which constraints have been verified, contradicted, or remain unknown for each candidate answer.
5. You review the ledger and decide your next action: search again or provide a final answer.

## Ledger

The ledger tracks the verification status for each candidate-constraint pair:
- **obj = true**: The constraint is satisfied
- **obj = false**: The constraint is false
- **obj = null**: The constraint is unknown

  ## Your Task

Based on the ledger:
- If any constraint has obj = null → Search for evidence to verify it
- If any constraint has obj = false → Reject that candidate and explore alternatives
- If all constraints have obj = true → Provide your final answer in \boxed{{...}}

You may ONLY provide a final answer when all constraints for a candidate are verified (obj = true).

Continue searching until verification is complete.
"""

logger = logging.getLogger(__name__)


def _coerce_list(value):
    """Recover a list when a tool-call arg (constraints/entries) came back as a
    JSON-encoded string instead of a parsed list. Without this, downstream code
    iterates the raw string char-by-char ("142 constraints: C1:[ C2:{ ...")."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                pass
        return [value] if value else []
    if value is None:
        return []
    return [value]


# ============================================================================
# MAIN AGENT - Three-Phase Approach
# ============================================================================

class EpistemicAgentThreePhase:
    """
    Agent that uses THREE PHASES:
    1. EXTRACT phase: Extract constraints from question
    2. SEARCH phase: Think and search
    3. UPDATE phase: Update ledger based on latest (t, q, r)
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
        reasoning_effort_search: str = "high",
        reasoning_effort_ledger: str = "high",
        ledger_base_url: str = None,
        ledger_api_key: str = None,
        ledger_model_name: str = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.search_engine = search_engine
        self.browser = browser
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort_search = reasoning_effort_search
        self.reasoning_effort_ledger = reasoning_effort_ledger

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=300.0,
            http_client=httpx.Client(verify=ssl_context),
        )

        # Hybrid mode: separate client for EXTRACT/UPDATE phases
        if ledger_base_url:
            self.ledger_client = OpenAI(
                base_url=ledger_base_url,
                api_key=ledger_api_key or "EMPTY",
                timeout=300.0,
                http_client=httpx.Client(verify=ssl_context),
            )
            self.ledger_model_name = ledger_model_name or model_name
        else:
            self.ledger_client = self.client
            self.ledger_model_name = self.model_name
    
    def _call_extract_constraints_phase(self, question: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        PHASE 1: Extract constraints from the question.
        Returns (list of constraint strings, extract_response dict).
        """
        log_event("EXTRACT_CONSTRAINTS - PHASE", "=== EXTRACT PHASE: Parsing question into constraints ===", ANSI_PHASE)

        extract_prompt = SYSTEM_PROMPT_EXTRACT_CONSTRAINTS.format(question=question)

        messages = [
            {"role": "system", "content": "You are a constraint extraction assistant. Parse questions into atomic, verifiable constraints."},
            {"role": "user", "content": extract_prompt}
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.ledger_client.chat.completions.create(
                    model=self.ledger_model_name,
                    messages=messages,
                    extra_body={"reasoning_effort": self.reasoning_effort_ledger},
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
                        constraints = _coerce_list(constraints)
                        if constraints:
                            # Log the extraction
                            reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None) or ''
                            content = message.content or ''

                            if reasoning:
                                log_event("EXTRACT_CONSTRAINTS - THINKING", reasoning, ANSI_THINK)
                            if content:
                                log_event("EXTRACT_CONSTRAINTS - RESPONSE", content, ANSI_RESPONSE)

                            log_event("EXTRACT_CONSTRAINTS - TOOL_CALL", f"extract_constraints(constraints={json.dumps(constraints, indent=2)})", ANSI_TOOL_CALL)

                            extract_response = {
                                "messages": messages,
                                "reasoning_content": reasoning,
                                "content": content,
                                "tool_calls": [tc.model_dump()],
                                "constraints": constraints,
                            }
                            return constraints, extract_response
                elif finish_reason == "stop":
                    content = message.content or ''
                    content = content.strip().strip("```json").strip("```").strip()
                    tool_call = json.loads(content)
                    constraints = tool_call.get("constraints", [])
                    constraints = _coerce_list(constraints)
                    if constraints:
                        # Log the extraction
                        reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None) or ''
                        content_log = message.content or ''

                        if reasoning:
                            log_event("EXTRACT_CONSTRAINTS - THINKING", reasoning, ANSI_THINK)
                        if content_log:
                            log_event("EXTRACT_CONSTRAINTS - RESPONSE", content_log, ANSI_RESPONSE)

                        log_event("EXTRACT_CONSTRAINTS - TOOL_CALL", f"extract_constraints(constraints={json.dumps(constraints, indent=2)})", ANSI_TOOL_CALL)

                        extract_response = {
                            "messages": messages,
                            "reasoning_content": reasoning,
                            "content": content_log,
                            "tool_calls": [{"function": {"name": "extract_constraints", "arguments": json.dumps({"constraints": constraints})}}],
                            "constraints": constraints,
                        }
                        return constraints, extract_response
            except Exception:
                log_event(f"EXTRACT_CONSTRAINTS - RETRY", f"Invalid tool call, retrying...", ANSI_RESPONSE)

        return [], {}
    
    def _call_update_phase(
        self,
        question: str,
        ledger: EpistemicLedger,
        thinking: str,
        search_query: str,
        retrieval_results: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        PHASE 3: Update ledger based on search results.
        Returns (list of entries to update, update_response dict).
        """

        update_prompt = SYSTEM_PROMPT_UPDATE_LEDGER.format(
            question=question,
            constraints=ledger.format_constraints_for_update(),
            ledger=ledger.format_ledger_json(),
            thinking=thinking or "(No explicit thinking)",
            search_query=search_query,
            retrieval_results=retrieval_results
        )

        log_event("UPDATE_LEDGER - PHASE", f"=== UPDATE PHASE: Updating ledger based on search results ===", ANSI_PHASE, entire=True)

        messages = [
            {"role": "system", "content": "You are a careful and thorough ledger update assistant. Given the constraints, the current ledger, the thinking, the search query, and the retrieval results, analyze search results and update the epistemic ledger of the candidates and constraints." \
                                           " Read the search results carefully, and if any candidate has supported evidence for the constraints in the search results, include the candidate and the evidence in the ledger accordingly.\nRemember to output only the tool calls without any other text."
            },
            {"role": "user", "content": update_prompt}
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.ledger_client.chat.completions.create(
                    model=self.ledger_model_name,
                    messages=messages,
                    extra_body={"reasoning_effort": self.reasoning_effort_ledger},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=TOOLS_UPDATE,
                )
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "tool_calls":
                    tc = response.choices[0].message.tool_calls[0]
                    if tc.function.name == "update_ledger":
                        fn_args = json.loads(tc.function.arguments)
                        entries = fn_args.get("entries", [])
                        entries = _coerce_list(entries)

                        message = response.choices[0].message
                        reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None) or ''
                        content = message.content or ''

                        if reasoning:
                            log_event("UPDATE_LEDGER - THINKING", reasoning, ANSI_THINK)
                        if content:
                            log_event("UPDATE_LEDGER - RESPONSE", content, ANSI_RESPONSE)
                        log_event("UPDATE_LEDGER - TOOL_CALL", f"Entries: {json.dumps(entries, indent=2)}", ANSI_TOOL_CALL)

                        update_response = {
                            "messages": messages,
                            "reasoning_content": reasoning,
                            "content": content,
                            "tool_calls": [tc.model_dump()],
                            "entries": entries,
                        }
                        return entries, update_response
                elif finish_reason == "stop":
                    content = response.choices[0].message.content or ''
                    reasoning = getattr(response.choices[0].message, 'reasoning', None) or getattr(response.choices[0].message, 'reasoning_content', None) or ''
                    if 'update_ledger(entries=[' in content:
                        entries = content.strip().strip('update_ledger(entries=[').strip('])').strip()
                        entries = json.loads(entries)
                        entries = _coerce_list(entries)
                        if entries:
                            log_event("UPDATE_LEDGER - TOOL_CALL", f"Entries: {json.dumps(entries, indent=2)}", ANSI_TOOL_CALL)
                            update_response = {
                                "messages": messages,
                                "reasoning_content": reasoning,
                                "content": content,
                                "tool_calls": [{"function": {"name": "update_ledger", "arguments": json.dumps({"entries": entries})}}],
                                "entries": entries,
                            }
                            return entries, update_response
                    elif 'entries' in content:
                        content_parsed = content.strip().strip('```json').strip('```').strip()
                        entries = json.loads(content_parsed).get('entries', [])
                        if entries:
                            log_event("UPDATE_LEDGER - TOOL_CALL", f"Entries: {json.dumps(entries, indent=2)}", ANSI_TOOL_CALL)
                            update_response = {
                                "messages": messages,
                                "reasoning_content": reasoning,
                                "content": content,
                                "tool_calls": [{"function": {"name": "update_ledger", "arguments": json.dumps({"entries": entries})}}],
                                "entries": entries,
                            }
                            return entries, update_response

            except Exception:
                log_event("UPDATE_LEDGER - RETRY", f"Invalid tool call, retrying...", ANSI_RESPONSE)

        return [], {}
    
    def run(self, question: str) -> Tuple[str, str, List, str, int, float, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run agent on question with three-phase approach.
        Returns (content, reasoning_content, messages, prediction, turns, latency, ledger, extract_response, update_responses).
        """

        ledger = EpistemicLedger()
        state_machine = AgentStateMachine()

        log_event("USER", question, ANSI_USER)

        start_time = time.time()

        # =====================================================================
        # PHASE 1: Extract constraints
        # =====================================================================
        constraints, extract_response = self._call_extract_constraints_phase(question)
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
        update_responses = []
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT_MAIN_W_LEDGER},
            {"role": "user", "content": question},
        ]
        
        turn = 0
        
        while turn < self.max_turns:
            turn += 1
            
            log_event("SEARCH - PHASE", f"=== SEARCH PHASE (Turn {turn}): Thinking and Searching ===", ANSI_PHASE)
            
            # print("-" * 100) 
            # print(json.dumps(messages, indent=2))
            # print("-" * 100, "\n")
            
            max_retries = 3
            response = None
            last_finish_reason = None
            last_content = ""
            last_exc = None
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        extra_body={"reasoning_effort": self.reasoning_effort_search},
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=TOOLS_SEARCH,
                    )
                    message = response.choices[0].message
                    content = message.content or ""
                    finish_reason = response.choices[0].finish_reason
                    last_content = content
                    last_finish_reason = finish_reason
                    if finish_reason == "tool_calls":
                        # Tool call format check
                        tc = message.tool_calls[0]
                        json.loads(tc.function.arguments)
                        break
                    elif content == "" or "query" in content.lower():
                        log_event("SEARCH - RETRY", f"attempt {attempt+1}/{max_retries} rejected (finish_reason={finish_reason}, content_preview={content[:200]!r})", ANSI_RESPONSE)
                        response = None
                        continue
                    else:
                        # No tool call
                        break
                except Exception as e:
                    last_exc = f"{type(e).__name__}: {e}"
                    log_event("SEARCH - RETRY", f"attempt {attempt+1}/{max_retries} failed: {last_exc}", ANSI_RESPONSE)
                    response = None

            if response is None:
                reason = last_exc or f"finish_reason={last_finish_reason}, content_preview={last_content[:200]!r}"
                log_event("SEARCH - FAILED", f"SEARCH phase failed after {max_retries} retries ({reason}); returning partial state", ANSI_RESPONSE)
                latency = time.time() - start_time
                return (
                    "",
                    "",
                    messages,
                    "[search_failed]",
                    turn,
                    latency,
                    ledger.ledger,
                    extract_response,
                    update_responses,
                )

            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason
            
            reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None) or ''
            content = message.content or ''
            
            if reasoning:
                log_event("SEARCH - THINKING", reasoning, ANSI_THINK)
                latest_thinking = reasoning
            if content:
                log_event("SEARCH - RESPONSE", content, ANSI_RESPONSE)
            
            # Handle tool calls (search/browse)
            if finish_reason == "tool_calls" and message.tool_calls:
                tool_results = []
                needs_ledger_update = False
                
                for tc in message.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    
                    log_event("SEARCH - TOOL_CALL", f"{fn_name}({json.dumps(fn_args, indent=2)})", ANSI_TOOL_CALL)
                    
                    error = state_machine.transition(fn_name, ledger.check_completion()[0])
                    if error:
                        result = error
                        log_event("SEARCH - STATE_ERROR", error, ANSI_LEDGER)
                    elif fn_name == "search":
                        if isinstance(fn_args, list):
                            queries = fn_args
                        elif isinstance(fn_args, dict):
                            queries = fn_args.get("query") or fn_args.get("queries") or []
                        else:
                            queries = []
                        if isinstance(queries, str):
                            queries = [queries]
                        result = self.search_engine.search_batch(queries)
                        latest_query = ", ".join(queries)
                        latest_results = result
                        needs_ledger_update = True
                    elif fn_name == "browse":
                        if isinstance(fn_args, list):
                            urls = fn_args
                        elif isinstance(fn_args, dict):
                            urls = fn_args.get("urls") or fn_args.get("url") or []
                        else:
                            urls = []
                        if isinstance(urls, str):
                            urls = [urls]
                        result = self.browser.browse_batch(urls)
                        latest_query = f"browse: {', '.join(urls)}"
                        latest_results = result
                        needs_ledger_update = True
                    else:
                        # result = f"Unknown tool: {fn_name}"
                        continue
                    
                    log_event("SEARCH - RESULT", result[:500] + ("..." if len(result) > 500 else ""), ANSI_TOOL_MSG)
                    
                    tool_results.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    })
                
                messages.append({
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls],
                })
                messages.extend(tool_results)
                
                # PHASE 3: Update ledger if we did a search/browse
                if needs_ledger_update and ledger.constraints:
                    ledger_before = copy.deepcopy(ledger.ledger)
                    entries, update_response = self._call_update_phase(
                        question=question,
                        ledger=ledger,
                        thinking=latest_thinking,
                        search_query=latest_query,
                        retrieval_results=latest_results,
                    )

                    if entries:
                        ledger.reset_stagnation_count()
                        updated_candidates = ledger.update(entries)

                        if update_response:
                            update_response["turn"] = turn
                            update_response["ledger_before"] = ledger_before
                            update_response["ledger_after"] = copy.deepcopy(ledger.ledger)
                            update_responses.append(update_response)
                        
                        is_complete, _ = ledger.check_completion()
                        if is_complete:
                            state_machine.set_complete()
                        
                        messages = [m for m in messages if m["role"] != "user"]
                        messages.append({
                            "role": "user", 
                            "content": ledger.format_ledger()
                        })
                        
                        log_event("LEDGER - UPDATED", ledger.format_ledger(), ANSI_LEDGER)
                
                    
                    else:
                        ledger.increase_stagnation_count()
                        log_event("LEDGER - NO UPDATES", ledger.format_ledger(), ANSI_LEDGER)
                        
                    if ledger.get_stagnation_count() == 5:
                        messages.append({
                            "role": "user", 
                            "content": (
                                "No new candidates or evidence found in the last 3 turns. If you are stuck, consider: (1) using different keywords, (2) searching for specific facts, or (3) trying alternative candidate answers."
                            )
                        })
                        ledger.reset_stagnation_count()
                        log_event("SEARCH - STAGNATION", "Resetting stagnation count after 3 turns of no progress", ANSI_LEDGER)
                    
                
                log_event("STATE", state_machine.get_state_message(), ANSI_LEDGER)
                continue
            
            # Check for final answer
            else:
                prediction = parse_boxed(content)
                
                # if prediction:
                #     is_complete, missing = ledger.check_completion()
                #     if not is_complete:
                #         warning = f"❌ There is no valid answer yet. You need to verify all constraints.\n"
                #         warning += f"Constraints to verify: {ledger.format_ledger()}\n"
                #         log_event("ANSWER_REJECTED", warning, ANSI_LEDGER)
                        
                #         messages.append({"role": "assistant", "content": content})
                #         messages.append({"role": "user", "content": warning})
                #         continue
                
                latency = time.time() - start_time
                
                log_event("FINAL STATE", state_machine.get_state_message(), ANSI_LEDGER)
                log_event("FINAL LEDGER", ledger.format_ledger(), ANSI_LEDGER)
                
                return content, reasoning, messages, prediction, turn, latency, ledger.ledger, extract_response, update_responses

        latency = time.time() - start_time

        return content, reasoning, messages, "exceeded max turns", turn, latency, ledger.ledger, extract_response, update_responses


class DataLoader:
    def __init__(self, data_path: str, start_idx: int = 0, end_idx: int = None):
        self.data_path = data_path
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def load_data(self) -> List[Dict[str, Any]]:
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r") as f:
                dataset = json.load(f)
        elif self.data_path.endswith(".jsonl"):
            with open(self.data_path, "r") as f:
                dataset = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file extension: {self.data_path}")
        
        if self.start_idx is not None or self.end_idx is not None:
            dataset = dataset[self.start_idx:self.end_idx]
        elif self.start_idx is not None:
            dataset = dataset[self.start_idx:]
        elif self.end_idx is not None:
            dataset = dataset[:self.end_idx]
            
        dataset = list(map(self.validate_datapoint, dataset))
        return dataset
        
    def validate_datapoint(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if "question" not in item:
            if "Question" in item:
                item["question"] = item["Question"]
                del item["Question"]
            else:
                raise ValueError(f"Question not found in item: {item}")
        if "answer" not in item:
            if "Answer" in item:
                item["answer"] = item["Answer"]
                del item["Answer"]
            elif "ground_truths" in item:
                item["answer"] = {
                    "ground_truths": item["ground_truths"],
                    "misc": item["misc"],
                    "canary": item["canary"],
                    "key": item["key"]
                }
            else:
                raise ValueError(f"Answer not found in item: {item}")
        return item
    
# ============================================================================
# MAIN
# ============================================================================

# Thread-safe progress tracking
progress_lock = Lock()
completed_count = 0
total_count = 0


def is_already_done(output_path: str) -> bool:
    """True iff output JSON exists and is parseable (success or error both count as done)."""
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path) as f:
            data = json.load(f)
        return "status" in data
    except Exception:
        return False


def process_item(
    idx: int,
    item: Dict[str, Any],
    agent: EpistemicAgentThreePhase,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Process a single dataset item. Thread-safe worker function.
    """
    global completed_count
    
    output_path = os.path.join(output_dir, f"{idx}.json")
    
    # Skip if already successfully processed; retry items previously saved as error.
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
        except Exception:
            existing = None
        if existing and existing.get("status") == "success":
            with progress_lock:
                completed_count += 1
                print(f"[{completed_count}/{total_count}] Skipping {idx} (already success)")
            return None
        # Otherwise fall through and retry (previous status was error or file unreadable)
    
    question = item["question"]
    answer = item["answer"]
    
    try:
        content, reasoning_content, messages, prediction, turns, latency, ledger, extract_response, update_responses = agent.run(question)

        is_soft_failure = isinstance(prediction, str) and prediction.startswith("[") and prediction.endswith("]")
        status = "error" if is_soft_failure else "success"

        result = {
            "question": question,
            "answer": answer,
            "content": content,
            "reasoning_content": reasoning_content,
            "messages": messages,
            "prediction": prediction,
            "turns": turns,
            "latency": latency,
            "elapsed_time": latency,
            "ledger": ledger,
            "status": status,
            "extract_response": extract_response,
            "update_responses": update_responses,
        }
        if is_soft_failure:
            result["error"] = prediction.strip("[]").replace("_", " ")

        # Write output atomically (write to temp then rename)
        temp_path = output_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.rename(temp_path, output_path)

        with progress_lock:
            completed_count += 1
            print(f"\n{'='*60}")
            tag = "Completed" if status == "success" else f"Soft-failed ({status})"
            print(f"[{completed_count}/{total_count}] {tag} item {idx}")
            print(f"Question: {question[:100]}...")
            print(f"Prediction: {prediction}")
            print(f"Turns: {turns}, Time: {latency:.1f}s")
            print(f"Saved to {output_path}")

        return result
        
    except Exception as e:
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
    global total_count, completed_count
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    urllib3.disable_warnings()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", default="openai/gpt-oss-120b")
    parser.add_argument("--base_url", default="http://localhost:8000/v1",
                        help="vLLM server(s) for SEARCH phase. Comma-separated for multiple instances; agents are assigned sticky round-robin to preserve prefix cache.")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--serper_api_key", default=os.getenv("SERPER_API_KEY", ""))
    parser.add_argument("--jina_api_key", default=os.getenv("JINA_API_KEY", ""))
    parser.add_argument("--reasoning_effort_search", default="high")
    parser.add_argument("--reasoning_effort_ledger", default="high")
    parser.add_argument("--ledger_base_url", default=None,
                        help="Separate vLLM server(s) for EXTRACT/UPDATE phases. Comma-separated for multiple instances.")
    parser.add_argument("--ledger_api_key", default="EMPTY")
    parser.add_argument("--ledger_model_name", default=None,
                        help="Model name for EXTRACT/UPDATE phases (defaults to --model_name)")
    parser.add_argument("--max_turns", type=int, default=30)
    
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(HERE, "data"))
    parser.add_argument("--dataset_names", "-d", nargs="+", type=str, default=["hds-qa", "multiconir_books"],
                        help="sub-folders of --dataset_dir, each with test_mcqa.jsonl")
    parser.add_argument("--start_idx", "-s", type=int, default=0)
    parser.add_argument("--end_idx", "-e", type=int, default=None)
    
    parser.add_argument("--output_dir", "-o", type=str, default="rollouts")
    parser.add_argument("--num_workers", "-w", type=int, default=24, help="Number of parallel workers for processing")
    args = parser.parse_args()

    # Parse comma-separated URL lists. Each agent is assigned ONE
    # (search_url, ledger_url) pair for its entire lifetime, so all turns
    # of a given task land on the same vLLM instance and prefix-cache hits.
    search_urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    if args.ledger_base_url:
        ledger_urls = [u.strip() for u in args.ledger_base_url.split(",") if u.strip()]
    else:
        ledger_urls = [None]

    def create_agent(worker_idx: int = 0):
        return EpistemicAgentThreePhase(
            base_url=search_urls[worker_idx % len(search_urls)],
            api_key=args.api_key,
            model_name=args.model_name,
            search_engine=SerperSearchEngine(serper_api_key=args.serper_api_key),
            browser=JinaBrowser(jina_api_key=args.jina_api_key),
            max_turns=args.max_turns,
            reasoning_effort_search=args.reasoning_effort_search,
            reasoning_effort_ledger=args.reasoning_effort_ledger,
            ledger_base_url=ledger_urls[worker_idx % len(ledger_urls)],
            ledger_api_key=args.ledger_api_key,
            ledger_model_name=args.ledger_model_name,
        )

    # Step A: build flat task list across all datasets, mkdir output dirs.
    tasks = []           # (dataset_name, idx, item, output_path, output_dir)
    per_ds_total = {}
    for dataset_name in args.dataset_names:
        data_path = os.path.join(args.dataset_dir, dataset_name, "test_mcqa.jsonl")
        dataset = DataLoader(data_path, args.start_idx, args.end_idx).load_data()
        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        per_ds_total[dataset_name] = len(dataset)
        for idx, item in enumerate(dataset):
            out_path = os.path.join(output_dir, f"{idx}.json")
            tasks.append((dataset_name, idx, item, out_path, output_dir))

    # Step B: pre-skip items whose output already exists with status=success.
    pending = [t for t in tasks if not is_already_done(t[3])]
    per_ds_skipped = {ds: 0 for ds in args.dataset_names}
    for ds, _, _, out_path, _ in tasks:
        if is_already_done(out_path):
            per_ds_skipped[ds] += 1
    skipped_pre = len(tasks) - len(pending)

    print(f"{'='*60}")
    print(f"[plan] {len(args.dataset_names)} datasets, {len(tasks)} items total, "
          f"{skipped_pre} already done, {len(pending)} to run with {args.num_workers} worker(s)")
    for ds in args.dataset_names:
        print(f"  {ds}: total={per_ds_total[ds]} done={per_ds_skipped[ds]} pending={per_ds_total[ds] - per_ds_skipped[ds]}")
    print(f"[routing] {len(search_urls)} SEARCH url(s), {len(ledger_urls)} LEDGER url(s) (sticky round-robin)")
    print(f"Output base: {args.output_dir}")
    print(f"{'='*60}\n")

    total_count = len(pending)
    completed_count = 0

    results_by_ds: Dict[str, List[Optional[Dict[str, Any]]]] = {ds: [] for ds in args.dataset_names}
    start_time = time.time()

    if args.num_workers == 1 or len(pending) == 0:
        agent = create_agent(0) if pending else None
        for ds, idx, item, _, output_dir in pending:
            try:
                results_by_ds[ds].append(process_item(idx, item, agent, output_dir))
            except Exception as e:
                print(f"[ERROR] {ds}/{idx}: {e}")
                results_by_ds[ds].append({"idx": idx, "status": "error", "error": str(e)})
    else:
        # One agent per worker — fixed (search_url, ledger_url) for sticky cache.
        agents = [create_agent(i) for i in range(args.num_workers)]
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for task_pos, (ds, idx, item, _, output_dir) in enumerate(pending):
                agent = agents[task_pos % args.num_workers]
                fut = executor.submit(process_item, idx, item, agent, output_dir)
                futures[fut] = (ds, idx)
            for fut in as_completed(futures):
                ds, idx = futures[fut]
                try:
                    results_by_ds[ds].append(fut.result())
                except Exception as e:
                    print(f"[ERROR] {ds}/{idx}: {e}")
                    results_by_ds[ds].append({"idx": idx, "status": "error", "error": str(e)})

    total_time = time.time() - start_time

    # Per-dataset summary files (preserved shape) + global tally.
    g_successful = g_errors = g_skipped = 0
    print(f"\n{'='*60}")
    print(f"COMPLETED in {total_time:.1f}s")
    print(f"{'='*60}")
    for ds in args.dataset_names:
        rs = results_by_ds[ds]
        successful = sum(1 for r in rs if r and r.get("status") == "success")
        errors = sum(1 for r in rs if r and r.get("status") == "error")
        skipped = per_ds_skipped[ds] + sum(1 for r in rs if r is None)
        g_successful += successful
        g_errors += errors
        g_skipped += skipped
        print(f"  {ds}: total={per_ds_total[ds]} success={successful} errors={errors} skipped={skipped}")
        summary = {
            "total_items": per_ds_total[ds],
            "successful": successful,
            "errors": errors,
            "skipped": skipped,
            "elapsed_time": total_time,
        }
        out_dir = os.path.join(args.output_dir, ds)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nTotal: success={g_successful} errors={g_errors} skipped={g_skipped} time={total_time:.1f}s")
    if g_successful > 0:
        print(f"Avg time per successful item: {total_time / g_successful:.1f}s")


if __name__ == "__main__":
    main()
