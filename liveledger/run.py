"""LiveLedger: Three-phase epistemic agent for multi-constraint QA."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ssl
import httpx
import requests
import urllib3
from openai import OpenAI

from tools import (
    TOOLS_EXTRACT, 
    TOOLS_SEARCH, 
    TOOLS_UPDATE
)
from prompt import (
    SYSTEM_PROMPT_EXTRACT_CONSTRAINTS, 
    SYSTEM_PROMPT_MAIN_W_LEDGER, 
    SYSTEM_PROMPT_UPDATE_LEDGER
)
from search_engine import (
    SerperSearchEngine, 
    JinaBrowser
)
from utils import * 

logger = logging.getLogger(__name__)

# Field name for the model's chain of thought moved between vLLM releases: older
# builds put it in `reasoning_content`, current ones (0.16.1 and 0.25.1, both
# verified against these servers) use `reasoning` and omit `reasoning_content`
# entirely. Reading only the old name silently yields "" on every call, which
# fed `(No explicit thinking)` into every ledger-update prompt.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def reasoning_of(message) -> str:
    """Return the model's reasoning text under whichever field this vLLM uses."""
    for attr in _REASONING_FIELDS:
        value = getattr(message, attr, None)
        if value:
            return value
    return ""


def api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip reasoning from the outgoing request body.

    `messages` doubles as the saved transcript, so it now carries real reasoning
    text. That must not be replayed to the model: until this fix the field was
    always empty, so sending it back would change what the agent sees between
    turns rather than restore prior behaviour.
    """
    return [{k: v for k, v in m.items() if k not in _REASONING_FIELDS}
            for m in messages]


class EpistemicAgentThreePhase:
    
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        ledger_base_url: str = None,
        ledger_api_key: str = None,
        ledger_model_name: str = None,
        search_engine: SerperSearchEngine,
        browser: JinaBrowser,
        max_turns: int = 50,
        temperature: float = 1.0,
        max_tokens: int = 8192,
        reasoning_effort_search: str = "high",
        reasoning_effort_ledger: str = "high",
        ledger_disable_thinking: bool = False,
        ledger_max_inflight: int = 1,
        ledger_force_toolcall: bool = True,
        ledger_entries_cap: int = 10,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.ledger_model_name = ledger_model_name or model_name
        self.search_engine = search_engine
        self.browser = browser
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort_search = reasoning_effort_search
        self.reasoning_effort_ledger = reasoning_effort_ledger
        self.ledger_extra_body = {"reasoning_effort": reasoning_effort_ledger}
        if ledger_disable_thinking:
            self.ledger_extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        self.ledger_max_inflight = max(1, int(ledger_max_inflight))
        self.ledger_force_toolcall = bool(ledger_force_toolcall)
        if self.ledger_force_toolcall:
            self.ledger_extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        self.ledger_tools_update = copy.deepcopy(TOOLS_UPDATE)
        if ledger_entries_cap:
            for _t in self.ledger_tools_update:
                _props = (_t.get("function", {}).get("parameters", {})
                            .get("properties", {}))
                if "entries" in _props:
                    _props["entries"]["maxItems"] = int(ledger_entries_cap)

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=300.0,
            http_client=httpx.Client(verify=ssl_context),
        )

        ledger_base_url = ledger_base_url or base_url
        ledger_api_key = ledger_api_key or api_key
        if ledger_base_url != base_url or ledger_api_key != api_key:
            ssl_context_ledger = ssl.create_default_context()
            ssl_context_ledger.check_hostname = False
            ssl_context_ledger.verify_mode = ssl.CERT_NONE
            self.ledger_client = OpenAI(
                base_url=ledger_base_url,
                api_key=ledger_api_key,
                timeout=300.0,
                http_client=httpx.Client(verify=ssl_context_ledger),
            )
        else:
            self.ledger_client = self.client
    
    def _call_extract_constraints_phase(self, question: str) -> List[str]:
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
                    extra_body=self.ledger_extra_body,
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
                            # Log the extraction
                            reasoning = reasoning_of(message)
                            content = message.content or ''

                            if reasoning:
                                log_event("EXTRACT_CONSTRAINTS - THINKING", reasoning, ANSI_THINK)
                            if content:
                                log_event("EXTRACT_CONSTRAINTS - RESPONSE", content, ANSI_RESPONSE)

                            log_event("EXTRACT_CONSTRAINTS - TOOL_CALL", f"extract_constraints(constraints={json.dumps(constraints, indent=2)})", ANSI_TOOL_CALL)

                            return constraints
                elif finish_reason == "stop":
                    content = message.content or ''
                    content = content.strip().strip("```json").strip("```").strip()
                    tool_call = json.loads(content)
                    constraints = tool_call.get("constraints", [])
                    if constraints:
                        # Log the extraction
                        reasoning = reasoning_of(message)
                        content_log = message.content or ''

                        if reasoning:
                            log_event("EXTRACT_CONSTRAINTS - THINKING", reasoning, ANSI_THINK)
                        if content_log:
                            log_event("EXTRACT_CONSTRAINTS - RESPONSE", content_log, ANSI_RESPONSE)

                        log_event("EXTRACT_CONSTRAINTS - TOOL_CALL", f"extract_constraints(constraints={json.dumps(constraints, indent=2)})", ANSI_TOOL_CALL)

                        return constraints
            except Exception:
                log_event(f"EXTRACT_CONSTRAINTS - RETRY", f"Invalid tool call, retrying...", ANSI_RESPONSE)
    
    def _call_update_phase(
        self,
        question: str,
        constraints: str,
        ledger_json: str,
        thinking: str,
        search_query: str,
        retrieval_results: str,
    ) -> List[Dict[str, Any]]:

        update_prompt = SYSTEM_PROMPT_UPDATE_LEDGER.format(
            question=question,
            constraints=constraints,
            ledger=ledger_json,
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
                _kwargs = dict(
                    model=self.ledger_model_name,
                    messages=messages,
                    extra_body=self.ledger_extra_body,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=self.ledger_tools_update,
                )
                if self.ledger_force_toolcall:
                    _kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": "update_ledger"},
                    }
                response = self.ledger_client.chat.completions.create(**_kwargs)
                finish_reason = response.choices[0].finish_reason
                if response.choices[0].message.tool_calls:
                    tc = response.choices[0].message.tool_calls[0]
                    if tc.function.name == "update_ledger":
                        fn_args = json.loads(tc.function.arguments)
                        entries = fn_args.get("entries", [])
                        # Validate entries is a list of dicts
                        if isinstance(entries, dict):
                            entries = [entries]
                        entries = [e for e in entries if isinstance(e, dict)]

                        message = response.choices[0].message
                        reasoning = reasoning_of(message)
                        content = message.content or ''

                        if reasoning:
                            log_event("UPDATE_LEDGER - THINKING", reasoning, ANSI_THINK)
                        if content:
                            log_event("UPDATE_LEDGER - RESPONSE", content, ANSI_RESPONSE)
                        log_event("UPDATE_LEDGER - TOOL_CALL", f"Entries: {json.dumps(entries, indent=2)}", ANSI_TOOL_CALL)

                        return entries
                elif finish_reason == "stop":
                    content = response.choices[0].message.content or ''
                    content_clean = content.strip().strip('```json').strip('```').strip()
                    try:
                        parsed = json.loads(content_clean)
                        if isinstance(parsed, dict):
                            entries = parsed.get('entries', parsed.get('candidates', []))
                            if isinstance(entries, dict):
                                entries = [entries]
                        elif isinstance(parsed, list):
                            entries = parsed
                        else:
                            entries = []
                        entries = [e for e in entries if isinstance(e, dict)]
                        if entries:
                            log_event("UPDATE_LEDGER - TOOL_CALL", f"Entries: {json.dumps(entries, indent=2)}", ANSI_TOOL_CALL)
                            return entries
                    except (json.JSONDecodeError, TypeError):
                        pass

            except Exception:
                log_event("UPDATE_LEDGER - RETRY", f"Invalid tool call, retrying...", ANSI_RESPONSE)

        return []
    
    # N-1 relaxed completion: auxiliary constraints (proximity, approximate numbers)
    # that snippet search rarely verifies don't block completion.
    _AUX_CONSTRAINT_RE = re.compile(
        r"(\bmeters?\b|\bmetres?\b|\bmiles?\b|\bkm\b|\bfeet\b|\byards?\b"
        r"|\bwithin\b|\bfrom (the |a )?(train |bus )?(station|restaurant|store|stop)"
        r"|\bapproximately\b|\baround\b|\broughly\b|\babout\b"
        r"|\bbetween \d|\b\d+\s*(to|-|–)\s*\d+"
        r"|\bas of\b|\bcirca\b|\bdecade\b|\bcentury\b)",
        re.IGNORECASE,
    )

    def _best_ledger_candidate(self, ledger):
        """Fallback candidate when turn budget runs out. Tiers: fully verified > N-1 auxiliary > N-1 any."""
        best = None  # (score, candidate)
        for candidate in ledger.ledger.keys():
            is_complete, is_false, missing = ledger.check_completion_of_candidate(candidate)
            if is_false:
                continue
            cons = ledger.ledger[candidate].get("constraints", {})
            n_true = sum(1 for c in cons.values() if c.get("obj") is True)
            if is_complete:
                score = n_true + 2000
            elif len(missing) == 1 and self._AUX_CONSTRAINT_RE.search(
                    ledger.constraints.get(missing[0], "")):
                score = n_true + 1000
            elif len(missing) == 1:
                score = n_true
            else:
                continue
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None and best[0] < 1000:
            miss = ledger.check_completion_of_candidate(best[1])[2]
            log_event("LEDGER - N-1 FALLBACK",
                      f"out of turns: committing '{best[1]}', which satisfies every "
                      f"constraint but {miss[0]} "
                      f"({ledger.constraints.get(miss[0], '')[:80]})",
                      ANSI_LEDGER)
        return best[1] if best else None

    def _relaxed_check_completion(self, ledger):
        """Returns (is_complete, missing) honoring the N-1 auxiliary rule."""
        is_complete, missing = ledger.check_completion()
        if is_complete:
            return True, missing
        for candidate in ledger.ledger.keys():
            _, is_false, miss = ledger.check_completion_of_candidate(candidate)
            if is_false or len(miss) != 1:
                continue
            ctext = ledger.constraints.get(miss[0], "")
            if self._AUX_CONSTRAINT_RE.search(ctext):
                log_event("LEDGER - RELAXED COMPLETE",
                          f"candidate '{candidate}' complete under N-1 rule "
                          f"(unverified auxiliary constraint {miss[0]}: {ctext[:80]})",
                          ANSI_LEDGER)
                return True, miss
        return False, missing

    def _apply_update_result(self, entries, ledger, state_machine, messages):
        if entries:
            ledger.reset_stagnation_count()
            ledger.update(entries)
            is_complete, _ = self._relaxed_check_completion(ledger)
            if is_complete:
                state_machine.set_complete()
            # Refresh the ledger view without losing the question. Dropping every
            # user message and re-adding only the ledger table deleted the
            # question itself (run() seeds it as messages[1]), so from the first
            # update onward the agent never saw it again: transcripts end up with
            # a single user message holding the ledger table, and some runs
            # answered "I don't see the question". Keep the first user message,
            # in its original slot right after the system prompt.
            question_msg = next((m for m in messages if m["role"] == "user"), None)
            messages[:] = [m for m in messages if m["role"] != "user"]
            if question_msg is not None:
                at = 1 if messages and messages[0].get("role") == "system" else 0
                messages.insert(at, question_msg)
            messages.append({"role": "user", "content": ledger.format_ledger()})
            log_event("LEDGER - UPDATED (async)", ledger.format_ledger(), ANSI_LEDGER)
        else:
            ledger.increase_stagnation_count()
            log_event("LEDGER - NO UPDATES (async)", ledger.format_ledger(), ANSI_LEDGER)
        if ledger.get_stagnation_count() == 5:
            messages.append({
                "role": "user",
                "content": (
                    "No new candidates or evidence found in the last 3 turns. If you are stuck, consider: (1) using different keywords, (2) searching for specific facts, or (3) trying alternative candidate answers."
                )
            })
            ledger.reset_stagnation_count()

    def _timed_update_call(self, timing, **kwargs):
        _t = time.time()
        try:
            return self._call_update_phase(**kwargs)
        finally:
            timing["update_calls"].append(round(time.time() - _t, 2))

    def _apply_future(self, fut, ledger, state_machine, messages):
        try:
            entries = fut.result()
        except Exception as exc:
            log_event("LEDGER - ASYNC ERROR", str(exc)[:200], ANSI_LEDGER)
            entries = []
        self._apply_update_result(entries, ledger, state_machine, messages)

    def _harvest_pending(self, pending, ledger, state_machine, messages, timing):
        while pending and pending[0].done():
            self._apply_future(pending.popleft(), ledger, state_machine, messages)
        while len(pending) >= self.ledger_max_inflight:
            _t = time.time()
            self._apply_future(pending.popleft(), ledger, state_machine, messages)
            timing["drain_wait"] += time.time() - _t
            while pending and pending[0].done():
                self._apply_future(pending.popleft(), ledger, state_machine, messages)

    def _drain_all_pending(self, pending, ledger, state_machine, messages, timing):
        while pending:
            _t = time.time()
            self._apply_future(pending.popleft(), ledger, state_machine, messages)
            timing["drain_wait"] += time.time() - _t

    def run(self, question: str) -> Tuple[str, str, int, float, Dict[str, Any]]:

        ledger = EpistemicLedger()
        state_machine = AgentStateMachine()
        update_executor = ThreadPoolExecutor(max_workers=self.ledger_max_inflight)
        pending_updates = deque()
        
        log_event("USER", question, ANSI_USER)

        start_time = time.time()
        timing = {"extract": 0.0, "search_llm": 0.0, "tools": 0.0,
                  "drain_wait": 0.0, "update_calls": []}

        _t = time.time()
        constraints = self._call_extract_constraints_phase(question)
        timing["extract"] = time.time() - _t
        ledger.set_constraints(constraints)
        state_machine.transition("extract_constraints")
        
        init_msg = f"✓ Initialized ledger with {len(ledger.constraints)} constraints:\n"
        init_msg += "\n".join(f"  {k}: {v}" for k, v in ledger.constraints.items())
        log_event("EXTRACT_CONSTRAINTS - RESULT", init_msg, ANSI_TOOL_MSG)
        log_event("STATE", state_machine.get_state_message(), ANSI_LEDGER)
        
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
            
            log_event("SEARCH - PHASE", f"=== SEARCH PHASE (Turn {turn}): Thinking and Searching ===", ANSI_PHASE)
            
            # print("-" * 100) 
            # print(json.dumps(messages, indent=2))
            # print("-" * 100, "\n")
            
            max_retries = 3
            _t_search = time.time()
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=api_messages(messages),
                        extra_body={"reasoning_effort": self.reasoning_effort_search},
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=TOOLS_SEARCH,
                    )
                    message = response.choices[0].message
                    content = message.content or ""
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == "tool_calls":
                        # Tool call format check
                        tc = message.tool_calls[0]
                        json.loads(tc.function.arguments)
                        break
                    elif content == "" or "query" in content.lower():
                        continue
                    else:
                        # No tool call
                        break
                except Exception:
                    log_event("SEARCH - RETRY", f"Invalid response, retrying...", ANSI_RESPONSE)
            
            timing["search_llm"] += time.time() - _t_search

            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            reasoning = reasoning_of(message)
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
                    
                    error = state_machine.transition(fn_name, self._relaxed_check_completion(ledger)[0])
                    if error:
                        result = error
                        log_event("SEARCH - STATE_ERROR", error, ANSI_LEDGER)
                    elif fn_name == "search":
                        queries = fn_args.get("query", [])
                        if type(queries) == str:
                            queries = [queries]
                        _t = time.time()
                        result = self.search_engine.search_batch(queries)
                        timing["tools"] += time.time() - _t
                        latest_query = ", ".join(queries)
                        latest_results = result
                        needs_ledger_update = True
                    elif fn_name == "browse":
                        urls = fn_args.get("urls", [])
                        if type(urls) == str:
                            urls = [urls]
                        _t = time.time()
                        result = self.browser.browse_batch(urls)
                        timing["tools"] += time.time() - _t
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
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls],
                })
                messages.extend(tool_results)
                
                if needs_ledger_update and ledger.constraints:
                    self._harvest_pending(
                        pending_updates, ledger, state_machine, messages, timing)
                    pending_updates.append(update_executor.submit(
                        self._timed_update_call,
                        timing,
                        question=question,
                        constraints=ledger.format_constraints_for_update(),
                        ledger_json=ledger.format_ledger_json(),
                        thinking=latest_thinking,
                        search_query=latest_query,
                        retrieval_results=latest_results,
                    ))

                log_event("STATE", state_machine.get_state_message(), ANSI_LEDGER)
                continue
            
            else:
                prediction = parse_boxed(content)

                messages.append({
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                })

                self._drain_all_pending(
                    pending_updates, ledger, state_machine, messages, timing)
                update_executor.shutdown(wait=False)
                latency = time.time() - start_time

                log_event("FINAL STATE", state_machine.get_state_message(), ANSI_LEDGER)
                log_event("FINAL LEDGER", ledger.format_ledger(), ANSI_LEDGER)

                return content, messages, prediction, turn, latency, ledger.ledger, timing

        # Store last message with reasoning on max turns
        messages.append({
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
        })

        self._drain_all_pending(
            pending_updates, ledger, state_machine, messages, timing)
        update_executor.shutdown(wait=False)
        latency = time.time() - start_time

        # Out of turns: if the ledger already verified a candidate (all
        # constraints True, or N-1 with only an auxiliary one unmet), commit it
        # instead of returning no answer at all. Without this a run can verify
        # the right entity and still score zero purely because the turn budget
        # ran out before the model chose to answer.
        fallback = self._best_ledger_candidate(ledger)
        if fallback:
            log_event("LEDGER - MAXTURN FALLBACK",
                      f"committing verified candidate '{fallback}' at max turns",
                      ANSI_LEDGER)
            return content, messages, fallback, turn, latency, ledger.ledger, timing

        return content, messages, "exceeded max turns", turn, latency, ledger.ledger, timing


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

progress_lock = Lock()
completed_count = 0
total_count = 0


def process_item(
    idx: int,
    item: Dict[str, Any],
    agent: EpistemicAgentThreePhase,
    output_dir: str,
) -> Dict[str, Any]:
    global completed_count
    output_path = os.path.join(output_dir, f"{idx}.json")
    if os.path.exists(output_path):
        with progress_lock:
            completed_count += 1
            print(f"[{completed_count}/{total_count}] Skipping {idx} (already exists)")
        return None
    
    question = item["question"]
    answer = item["answer"]
    
    try:
        content, messages, prediction, turns, latency, ledger, timing = agent.run(question)

        result = {
            "question": question,
            "answer": answer,
            "content": content,
            "messages": messages,
            "prediction": prediction,
            "turns": turns,
            "latency": latency,
            "elapsed_time": latency,
            "ledger": ledger,
            "timing": timing,
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
    urllib3.disable_warnings()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", default="openai/gpt-oss-120b")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--ledger_model_name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ledger_base_url", default="http://localhost:9000/v1")
    parser.add_argument("--ledger_api_key", default=None)
    parser.add_argument("--serper_api_key", default=os.getenv("SERPER_API_KEY", ""))
    parser.add_argument("--jina_api_key", default=os.getenv("JINA_API_KEY", ""))
    parser.add_argument("--reasoning_effort_search", default="high")
    parser.add_argument("--reasoning_effort_ledger", default="high")
    parser.add_argument("--ledger_disable_thinking", action="store_true")
    parser.add_argument("--enable_thinking", "--ledger_enable_thinking",
                        "--ledger_reasoning_ledger", dest="ledger_force_toolcall",
                        action="store_false")
    parser.add_argument("--ledger_force_toolcall", dest="ledger_force_toolcall",
                        action="store_true")
    parser.set_defaults(ledger_force_toolcall=True)
    parser.add_argument("--ledger_entries_cap", type=int, default=10)
    parser.add_argument("--ledger_max_inflight", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=100) 
    
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--dataset_names", "-d", nargs="+", type=str, 
        default=["frames", "browsecomp", "deepsearchqa", "livedrbench", "webwalkerqa", "bioasq"],
        choices=["frames", "browsecomp", "deepsearchqa", "livedrbench", "webwalkerqa", "bioasq"]
    )
    parser.add_argument("--start_idx", "-s", type=int, default=0)
    parser.add_argument("--end_idx", "-e", type=int, default=None)
    
    parser.add_argument("--output_dir", "-o", type=str, default="outputs")
    parser.add_argument("--num_workers", "-w", type=int, default=32)
    args = parser.parse_args()
    
    for dataset_name in args.dataset_names:
        args.dataset_name = dataset_name
        dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
        data_path = os.path.join(dataset_dir, "test_mcqa.jsonl")
        dataset = DataLoader(data_path, args.start_idx, args.end_idx).load_data()
        
        output_dir = os.path.join(args.output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        total_count = len(dataset)
        completed_count = 0
        
        print(f"{'='*60}")
        print(f"Processing {total_count} items with {args.num_workers} worker(s)")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        def create_agent():
            return EpistemicAgentThreePhase(
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model_name,
                ledger_base_url=args.ledger_base_url,
                ledger_api_key=args.ledger_api_key,
                ledger_model_name=args.ledger_model_name,
                search_engine=SerperSearchEngine(serper_api_key=args.serper_api_key),
                browser=JinaBrowser(jina_api_key=args.jina_api_key),
                max_turns=args.max_turns,
                reasoning_effort_search=args.reasoning_effort_search,
                reasoning_effort_ledger=args.reasoning_effort_ledger,
                ledger_disable_thinking=args.ledger_disable_thinking,
                ledger_max_inflight=args.ledger_max_inflight,
                ledger_force_toolcall=args.ledger_force_toolcall,
                ledger_entries_cap=args.ledger_entries_cap,
            )
        
        start_time = time.time()
        
        if args.num_workers == 1:
            agent = create_agent()
            results = []
            for idx, item in enumerate(dataset):
                result = process_item(idx, item, agent, output_dir)
                results.append(result)
        else:
            # Multi-threaded processing
            # Create a pool of agents (one per worker for thread safety)
            agents = [create_agent() for _ in range(args.num_workers)]
            
            results = []
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                # Submit all tasks
                futures = {}
                for idx, item in enumerate(dataset):
                    # Round-robin assign agents to tasks
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
                        print(f"[ERROR] Future for item {idx} raised exception: {e}")
                        results.append({"idx": idx, "status": "error", "error": str(e)})
        
        total_time = time.time() - start_time
        
        # Summary statistics
        successful = sum(1 for r in results if r and r.get("status") == "success")
        errors = sum(1 for r in results if r and r.get("status") == "error")
        skipped = sum(1 for r in results if r is None)
        
        print(f"\n{'='*60}")
        print(f"COMPLETED")
        print(f"{'='*60}")
        print(f"Total items: {total_count}")
        print(f"Successful: {successful}")
        print(f"Errors: {errors}")
        print(f"Skipped (already existed): {skipped}")
        print(f"Total time: {total_time:.1f}s")
        if successful > 0:
            print(f"Average time per successful item: {total_time / max(successful, 1):.1f}s")

        summary = {
            "total_items": total_count,
            "successful": successful,
            "errors": errors,
            "skipped": skipped,
            "elapsed_time": total_time,
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
