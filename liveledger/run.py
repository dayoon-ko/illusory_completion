"""Run a search agent (ReAct: search + browse) on the benchmark, with or without the LiveLedger tracker.

Both arms share everything except the ledger (--no_ledger = baseline):
  - one system prompt; the LiveLedger arm only adds the "Ledger" paragraph. No answer gate.
  - --max_turns 30 = 30 agent calls that make tool calls. At the cap the run ends with no answer (termination=turn_cap);
    --forced_final instead asks once more, without tools, for a final answer.
  - after every search/browse, the tracker (dayoon/LiveLedger-4B, served by vLLM) updates the ledger in the background;
    the newest ledger replaces the previous ledger message in the agent's context.
  - prior reasoning is not replayed to the agent (--replay_reasoning turns it on; used for Qwen-style models).

One JSON per item: <output_dir>/<dataset>/<index>.json (transcript, per-turn records, every ledger update, final answer).
Re-running the same command skips finished items.
"""
from __future__ import annotations

import argparse
import ast
import copy
import random
import json
import os
import ssl
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx
import requests
import urllib3
import openai
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import TOOLS_EXTRACT, TOOLS_SEARCH, TOOLS_UPDATE  # noqa: E402
from prompts import SYSTEM_PROMPT_EXTRACT_CONSTRAINTS, SYSTEM_PROMPT_UPDATE_LEDGER  # noqa: E402
from search import SerperSearchEngine, JinaBrowser, SEARCH_ERRORS  # noqa: E402
from ledger import EpistemicLedger, parse_boxed  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
RUNNER_VERSION = "illusory_completion v2"
DATASETS = ["browsecomp", "deepsearchqa", "frames", "livedrbench", "webwalkerqa", "bioasq"]


class InfraError(RuntimeError):
    """A server stayed unreachable for --server_wait seconds: the item is not a valid result."""


INFRA_STATUS = (502, 503, 504)
FATAL_STATUS = (401, 402, 403)  # bad key / out of credits: never a model failure; stop the arm
FATAL = {"reason": None}  # set on the first fatal error; no new items start afterwards


def is_infra_exc(exc) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", None) in INFRA_STATUS


def tolerant_loads(s):
    """Repair malformed tool arguments (OpenRouter models): strict json, then ast.literal_eval, then the first JSON value."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as orig:
        if not isinstance(s, str):
            raise
        try:
            v = ast.literal_eval(s.strip())
            if isinstance(v, dict):
                return v
        except (ValueError, SyntaxError):
            pass
        try:
            v = json.JSONDecoder().raw_decode(s.strip())[0]
            if isinstance(v, dict):
                return v
        except (json.JSONDecodeError, ValueError):
            pass
        raise orig


class OpenRouterPinningClient(httpx.Client):
    """Injects provider routing into OpenRouter chat bodies and absorbs 429s."""
    provider: Dict[str, Any] = {"quantizations": ["fp8"]}

    def send(self, request, **kwargs):
        is_or = (request.method == "POST" and request.url.host == "openrouter.ai"
                 and request.url.path.endswith("/chat/completions"))
        if is_or and self.provider:
            try:
                body = json.loads(request.content)
                body.setdefault("provider", dict(self.provider))
                headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
                new = self.build_request("POST", str(request.url), content=json.dumps(body).encode(), headers=headers)
                new.extensions = dict(request.extensions)
                request = new
            except Exception:
                pass
        response = super().send(request, **kwargs)
        if is_or:
            delay = 2.0
            for _ in range(8):
                if response.status_code != 429:
                    break
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 60)
                response = super().send(request, **kwargs)
        return response

# ---------------------------------------------------------------------------
# Prompts. Arms differ ONLY by LEDGER_PARAGRAPH.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_HEAD = r"""
You are a reasoning assistant that answers multi-constraint questions.

## Workflow

1. **SEARCH** → Call `search` or `browse` to find information
2. **REVIEW** → Think about the search results.
3. **REPEAT** → Continue until all constraints are verified
4. **ANSWER** → When complete, provide your answer in \boxed{...}
"""

LEDGER_PARAGRAPH = r"""
## Ledger

After each search or browse, an evaluator analyzes the results and, if it finds anything new, shows you an updated **ledger**. The ledger tracks the verification status of each candidate answer against each constraint:
- **obj = true**: the constraint is satisfied
- **obj = false**: the constraint is contradicted
- **obj = null**: the constraint is unknown

Use the ledger when deciding your next action: search for evidence on constraints that are still null, and move to alternative candidates when a constraint is false.
"""

SYSTEM_PROMPT_TAIL = r"""
Continue searching to verify constraints, or provide your final answer if complete with \boxed{...}.
"""

FORCED_FINAL_MSG = (r"You have reached the maximum number of tool calls. Do not call any more tools. "
                    r"Based on the information gathered so far, give your final answer now in \boxed{...}.")


TOOL_ARG_FIX = False  # --tool_arg_fix: accept url/link/queries/q and return an error message instead of ""
LEDGER_FEEDBACK = True  # --ledger_feedback off: drop the "Feedbacks:" lines under the ledger table


def build_system_prompt(use_ledger: bool) -> str:
    return SYSTEM_PROMPT_HEAD + (LEDGER_PARAGRAPH if use_ledger else "") + SYSTEM_PROMPT_TAIL


UPDATE_SYSTEM_MSG = (
    "You are a careful and thorough ledger update assistant. Given the constraints, the current ledger, the thinking, "
    "the search query, and the retrieval results, analyze search results and update the epistemic ledger of the "
    "candidates and constraints. Read the search results carefully, and if any candidate has supported evidence for "
    "the constraints in the search results, include the candidate and the evidence in the ledger accordingly.\n"
    "Remember to output only the tool calls without any other text.")

# gpt-oss sometimes emits a malformed harmony tool-call header. vLLM then either returns HTTP 500
# ("unexpected tokens remaining in message header") or leaks the raw tokens into `content`.
# Such content is a failed call, not an answer.
HARMONY_LEAK_MARKERS = ("<|call|>", "<|channel|>", "<|start|>", "<|constrain|>", "<|message|>", "to=functions.")


def is_harmony_leak(text: str) -> bool:
    return any(m in (text or "") for m in HARMONY_LEAK_MARKERS)


BAD_TOOL_STRINGS = ("No search results available", "Not enough credits", "Search failed", "Failed to read")

_REASONING_FIELDS = ("reasoning_content", "reasoning")


def reasoning_of(message) -> str:
    for attr in _REASONING_FIELDS:
        v = getattr(message, attr, None)
        if v:
            return v
    return ""


REPLAY_REASONING = False  # set from --replay_reasoning in main()
REPLAY_FIELD = "reasoning_content"  # --replay_reasoning_field ("reasoning" for gpt-oss via vLLM harmony)


def api_messages(messages):
    """Outgoing request body. Default: prior reasoning is NOT replayed (same in both arms).
    With --replay_reasoning, assistant messages keep `reasoning_content`, so a reasoning-aware chat template (Qwen3.5)
    renders the real thinking of the current tool-call chain instead of an empty <think></think> block (without it,
    Qwen copies the empty block and stops thinking after turn 1)."""
    if REPLAY_REASONING:
        if REPLAY_FIELD == "reasoning":
            # vLLM's gpt-oss (harmony) input parser reads the prior-turn analysis from `reasoning`
            # (harmony_utils.parse_chat_input), not `reasoning_content`; send it under that name.
            out = []
            for m in messages:
                m2 = {k: v for k, v in m.items() if k not in _REASONING_FIELDS}
                if m.get("role") == "assistant" and m.get("reasoning_content"):
                    m2["reasoning"] = m["reasoning_content"]
                out.append(m2)
            return out
        return [{k: v for k, v in m.items() if k != "reasoning"} for m in messages]
    return [{k: v for k, v in m.items() if k not in _REASONING_FIELDS} for m in messages]


def hf_revision(repo: str) -> Optional[str]:
    p = os.path.join(HF_HUB, "models--" + repo.replace("/", "--"), "refs", "main")
    try:
        return open(p).read().strip()
    except OSError:
        return None


def make_client(base_url, api_key, timeout=1800.0, openrouter_provider=None):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    if openrouter_provider is not None:
        hc = OpenRouterPinningClient(verify=ctx)
        hc.provider = openrouter_provider
    else:
        hc = httpx.Client(verify=ctx)
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0, http_client=hc)


class Agent:
    def __init__(self, args):
        self.args = args
        self.use_ledger = not args.no_ledger
        self.client = make_client(args.base_url, args.api_key,
                                  openrouter_provider=args.openrouter_provider if args.openrouter else None)
        self.ledger_client = make_client(args.ledger_base_url, "EMPTY") if self.use_ledger else None
        self.parse_args_fn = tolerant_loads if args.openrouter else json.loads
        self.n_repaired_args = 0
        self.search_engine = SerperSearchEngine(serper_api_key=args.serper_api_key)
        self.browser = JinaBrowser(jina_api_key=args.jina_api_key)
        self.system_prompt = build_system_prompt(self.use_ledger)
        # Tracker: forced tool call + no thinking + 10-entry cap
        self.ledger_extra_body = {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}
        self.tools_update = copy.deepcopy(TOOLS_UPDATE)
        for t in self.tools_update:
            props = t["function"]["parameters"]["properties"]
            if "entries" in props:
                props["entries"]["maxItems"] = args.ledger_entries_cap

    def create(self, client, **kw):
        """chat.completions.create that waits out a lost server (preemption/restart) instead of
        burning attempts; raises InfraError after --server_wait seconds."""
        t0, delay = time.time(), 10
        while True:
            try:
                return client.chat.completions.create(**kw)
            except Exception as exc:
                if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", None) in FATAL_STATUS:
                    FATAL["reason"] = f"HTTP {exc.status_code}: {str(exc)[:200]}"
                    print(f"[FATAL] {client.base_url}: {FATAL['reason']} -> stopping new items", flush=True)
                    raise InfraError(f"fatal API error {FATAL['reason']}") from exc
                if not is_infra_exc(exc):
                    raise
                waited = time.time() - t0
                if waited > self.args.server_wait:
                    raise InfraError(f"server {client.base_url} unreachable for {waited:.0f}s: "
                                     f"{type(exc).__name__}: {str(exc)[:200]}") from exc
                print(f"[INFRA WAIT] {client.base_url}: {type(exc).__name__} ({waited:.0f}s)", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)

    # ---------------------------------------------------------------- tracker
    def extract_constraints(self, question):
        msgs = [{"role": "system", "content": "You are a constraint extraction assistant. Parse questions into atomic, verifiable constraints."},
                {"role": "user", "content": SYSTEM_PROMPT_EXTRACT_CONSTRAINTS.format(question=question)}]
        errors = []
        for attempt in range(3):
            try:
                r = self.create(self.ledger_client,
                    model=self.args.ledger_model_name, messages=msgs, extra_body=self.ledger_extra_body,
                    temperature=self.args.temperature, max_tokens=self.args.max_tokens, tools=TOOLS_EXTRACT)
                m = r.choices[0].message
                if m.tool_calls:
                    tc = m.tool_calls[0]
                    if tc.function.name == "extract_constraints":
                        cons = json.loads(tc.function.arguments).get("constraints", [])
                        if cons:
                            return cons, {"status": "ok", "attempts": attempt + 1, "errors": errors}
                content = (m.content or "").strip().strip("```json").strip("```").strip()
                if content:
                    cons = json.loads(content).get("constraints", [])
                    if cons:
                        return cons, {"status": "ok", "attempts": attempt + 1, "errors": errors}
                errors.append(f"attempt {attempt+1}: no constraints (finish={r.choices[0].finish_reason})")
            except InfraError:
                raise
            except Exception as exc:
                errors.append(f"attempt {attempt+1}: {type(exc).__name__}: {str(exc)[:500]}")
        raise RuntimeError(f"extract_constraints failed 3x: {errors}")

    def update_call(self, question, constraints, ledger_json, thinking, search_query, retrieval_results):
        """Returns a record {status: ok|empty|error, entries, attempts, errors, truncated, seconds}."""
        t0 = time.time()
        rec = {"status": "error", "entries": [], "attempts": 0, "errors": [], "truncated": False}
        r_lim, t_lim = self.args.ledger_max_result_chars, self.args.ledger_max_thinking_chars
        for attempt in range(3):
            rec["attempts"] = attempt + 1
            rr = retrieval_results if len(retrieval_results) <= r_lim else retrieval_results[:r_lim] + "\n...[truncated]"
            th = thinking if len(thinking) <= t_lim else thinking[:t_lim // 2] + "\n...[truncated]...\n" + thinking[-t_lim // 2:]
            if rr is not retrieval_results or th is not thinking:
                rec["truncated"] = True
            prompt = SYSTEM_PROMPT_UPDATE_LEDGER.format(
                question=question, constraints=constraints, ledger=ledger_json,
                thinking=th or "(No explicit thinking)", search_query=search_query, retrieval_results=rr)
            try:
                r = self.create(self.ledger_client,
                    model=self.args.ledger_model_name,
                    messages=[{"role": "system", "content": UPDATE_SYSTEM_MSG}, {"role": "user", "content": prompt}],
                    extra_body=self.ledger_extra_body, temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens, tools=self.tools_update,
                    tool_choice={"type": "function", "function": {"name": "update_ledger"}})
                m = r.choices[0].message
                if m.tool_calls and m.tool_calls[0].function.name == "update_ledger":
                    entries = json.loads(m.tool_calls[0].function.arguments).get("entries", [])
                    if isinstance(entries, dict):
                        entries = [entries]
                    entries = [e for e in entries if isinstance(e, dict)]
                    rec.update(status="ok" if entries else "empty", entries=entries,
                               finish_reason=r.choices[0].finish_reason)
                    break
                rec["errors"].append(f"attempt {attempt+1}: no update_ledger tool call (finish={r.choices[0].finish_reason}, content={str(m.content)[:200]!r})")
            except InfraError:
                raise
            except Exception as exc:
                rec["errors"].append(f"attempt {attempt+1}: {type(exc).__name__}: {str(exc)[:500]}")
            # shrink inputs for the next attempt (covers context-length 400s)
            r_lim //= 2
            t_lim //= 2
        rec["seconds"] = round(time.time() - t0, 2)
        if rec["status"] == "error":
            print(f"[LEDGER ERROR] {rec['errors']}", flush=True)
        return rec

    # ---------------------------------------------------------------- agent
    def agent_call(self, messages, forced_final=False):
        """One agent LLM call with up to --agent_attempts attempts. Returns (message, finish_reason, attempt_log) or (None, None, log)."""
        log = []
        a = self.args
        n_att = a.agent_attempts
        for attempt in range(n_att):
            effort = a.reasoning_effort
            if forced_final and a.forced_final_fallback_effort and attempt >= n_att - 2:
                effort = a.forced_final_fallback_effort
            kwargs = dict(model=a.model_name, tools=TOOLS_SEARCH, extra_body=self.effort_body(effort),
                          max_tokens=a.forced_final_max_tokens if forced_final else a.max_tokens)
            if a.sampling == "explicit":
                kwargs["temperature"] = a.temperature
            if forced_final:
                kwargs["tool_choice"] = "none"
            try:
                r = self.create(self.client, messages=api_messages(messages), **kwargs)
                ch = r.choices[0]
                m = ch.message
                if m.tool_calls and not forced_final:
                    parsed = [self.parse_args_fn(tc.function.arguments) for tc in m.tool_calls]
                    for tc, pa in zip(m.tool_calls, parsed):
                        try:
                            json.loads(tc.function.arguments)
                        except Exception:  # repaired (openrouter path only): store valid JSON for replay
                            tc.function.arguments = json.dumps(pa, ensure_ascii=False)
                            self.n_repaired_args += 1
                            log.append(f"attempt {attempt+1}: repaired malformed tool arguments")
                    if forced_final is False and effort != a.reasoning_effort:
                        log.append(f"attempt {attempt+1}: effort={effort}")
                    return m, ch.finish_reason, log
                if is_harmony_leak(m.content):
                    log.append(f"attempt {attempt+1}: harmony tokens leaked into content (finish={ch.finish_reason}): {(m.content or '')[:200]!r}")
                    continue
                if (m.content or "").strip():
                    if effort != a.reasoning_effort:
                        log.append(f"attempt {attempt+1}: answered with fallback effort={effort}")
                    return m, ch.finish_reason, log
                log.append(f"attempt {attempt+1}: empty content, no tool call (finish={ch.finish_reason}, reasoning_chars={len(reasoning_of(m))}, effort={effort})")
            except InfraError:
                raise
            except Exception as exc:
                log.append(f"attempt {attempt+1}: {type(exc).__name__}: {str(exc)[:500]}")
        return None, None, log

    def effort_body(self, effort):
        """How reasoning effort is expressed per backend (identical for both arms of a backbone)."""
        mode = self.args.effort_control
        if mode == "reasoning_effort":      # vLLM gpt-oss (harmony)
            return {"reasoning_effort": effort}
        if mode == "openrouter":            # OpenRouter unified reasoning parameter
            return {"reasoning": {"effort": effort}}
        if mode == "qwen_thinking":         # Qwen3.5 on vLLM: no effort knob; "low" = thinking off
            return {"chat_template_kwargs": {"enable_thinking": effort not in ("low", "none")}}
        return {}

    def run(self, question):
        a = self.args
        t_start = time.time()
        timing = {"extract": 0.0, "agent_llm": 0.0, "tools": 0.0, "drain_wait": 0.0, "update_calls": []}
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": question}]
        question_msg = messages[1]

        def assert_question():
            assert messages[1] is question_msg and messages[1]["role"] == "user" and messages[1]["content"] == question, \
                "question dropped from agent context"

        ledger = EpistemicLedger()
        constraints, extract_rec = None, None
        executor = ThreadPoolExecutor(max_workers=1) if self.use_ledger else None
        pending = deque()  # (source_turn, future)
        ledger_updates: List[Dict[str, Any]] = []

        if self.use_ledger:
            t0 = time.time()
            constraints, extract_rec = self.extract_constraints(question)
            timing["extract"] = round(time.time() - t0, 2)
            ledger.set_constraints(constraints)

        def apply_one(source_turn, fut, applied_at, touch_messages=True):
            try:
                rec = fut.result()
            except Exception as exc:  # should not happen: update_call catches everything
                rec = {"status": "error", "entries": [], "errors": [f"future: {type(exc).__name__}: {exc}"]}
            timing["update_calls"].append(rec.get("seconds"))
            if rec["entries"]:
                ledger.update(rec["entries"])
            if rec["entries"] and touch_messages:
                messages[:] = [m for m in messages if m["role"] != "user" or m is question_msg]
                messages.append({"role": "user", "content": ledger.format_ledger(include_guidance=LEDGER_FEEDBACK)})
            assert_question()
            rec.update(source_turn=source_turn, applied_at_turn=applied_at,
                       ledger_after=copy.deepcopy(ledger.ledger))
            ledger_updates.append(rec)

        def harvest(applied_at, block_all=False, touch_messages=True):
            while pending and pending[0][1].done():
                st, f = pending.popleft(); apply_one(st, f, applied_at, touch_messages)
            while pending and (block_all or len(pending) >= a.ledger_max_inflight):
                t0 = time.time()
                st, f = pending.popleft(); apply_one(st, f, applied_at, touch_messages)
                timing["drain_wait"] += time.time() - t0

        def ledger_seen():
            if not self.use_ledger:
                return None
            um = [m for m in messages if m["role"] == "user" and m is not question_msg and m["content"] != FORCED_FINAL_MSG]
            return um[-1]["content"] if um else "(no ledger message yet)"

        turns: List[Dict[str, Any]] = []
        n_toolcall_turns = 0
        content, prediction, termination = "", "", None
        last_reasoning = ""
        latest_thinking = ""

        while True:
            forced = n_toolcall_turns >= a.max_turns
            if forced and not a.forced_final:
                # paper protocol: no forced answer at the cap -> the run ends with no answer
                content, prediction, termination = "", "", "turn_cap"
                break
            if forced:
                if self.use_ledger:
                    harvest(applied_at="forced_final", block_all=True)
                messages.append({"role": "user", "content": FORCED_FINAL_MSG})
            assert_question()
            seen = ledger_seen()
            snap = copy.deepcopy(ledger.ledger) if self.use_ledger else None
            t0 = time.time()
            msg, finish, attempt_log = self.agent_call(messages, forced_final=forced)
            timing["agent_llm"] += time.time() - t0
            call_idx = len(turns) + 1
            if msg is None:
                turns.append({"call": call_idx, "turn": None, "kind": "agent_error", "attempt_log": attempt_log,
                              "ledger_seen": seen, "ledger_state_seen": snap})
                termination = "agent_error_forced_final" if forced else "agent_error"
                break
            reasoning = reasoning_of(msg)
            content = msg.content or ""
            if reasoning:
                latest_thinking = reasoning
                last_reasoning = reasoning
            if msg.tool_calls and not forced:
                n_toolcall_turns += 1
                rec = {"call": call_idx, "turn": n_toolcall_turns, "kind": "tool_call", "finish_reason": finish,
                       "reasoning": reasoning, "content": content, "attempt_log": attempt_log,
                       "ledger_seen": seen, "ledger_state_seen": snap, "tool_calls": [], "tool_results": []}
                tool_msgs, latest_query, latest_results, need_update = [], "", "", False
                for tc in msg.tool_calls:
                    name = tc.function.name
                    fa = json.loads(tc.function.arguments)  # already validated / repaired in agent_call
                    if TOOL_ARG_FIX and not isinstance(fa, dict):
                        # some models (e.g. Qwen2.5-7B, hermes parser) send the arguments as a bare list
                        # (["q1", "q2"] or [{"query": ...}]) or a string; read them as the query / url list.
                        key = "query" if name == "search" else "urls"
                        flat = []
                        for x in (fa if isinstance(fa, list) else [fa]):
                            if isinstance(x, dict):
                                v = x.get(key) or x.get("query") or x.get("queries") or x.get("urls") or x.get("url") or []
                                flat += v if isinstance(v, list) else [v]
                            else:
                                flat.append(x)
                        fa = {key: [str(v) for v in flat if v not in (None, "")], "_raw_non_dict_args": True}
                    t1 = time.time()
                    if name == "search":
                        qs = fa.get("query", [])
                        if TOOL_ARG_FIX and not qs:
                            qs = fa.get("queries") or fa.get("q") or []
                        qs = [qs] if isinstance(qs, str) else qs
                        if TOOL_ARG_FIX and not qs:
                            result = ('Error: search needs {"query": ["<search query>", ...]}; got arguments '
                                      + json.dumps(fa)[:300])
                        else:
                            result = self.search_engine.search_batch(qs)
                            latest_query, latest_results, need_update = ", ".join(qs), result, True
                    elif name == "browse":
                        us = fa.get("urls", [])
                        if TOOL_ARG_FIX and not us:
                            # gpt-oss often calls browse in its native browser style ({"url": ...},
                            # {"id": 3}, {"cursor": 0, "loc": 2000}); without this the call silently returns "".
                            us = fa.get("url") or fa.get("link") or []
                        us = [us] if isinstance(us, str) else us
                        if TOOL_ARG_FIX and not us:
                            result = ('Error: browse needs {"urls": ["<full URL starting with http>", ...]}; got arguments '
                                      + json.dumps(fa)[:300] + '. Copy the URL from the search results.')
                        else:
                            result = self.browser.browse_batch(us)
                            latest_query, latest_results, need_update = f"browse: {', '.join(us)}", result, True
                    else:
                        result = f"Unknown tool: {name}. Available tools: search, browse."
                    timing["tools"] += time.time() - t1
                    rec["tool_calls"].append({"name": name, "arguments": fa})
                    rec["tool_results"].append(result)
                    tool_msgs.append({"role": "tool", "content": result, "tool_call_id": tc.id})
                rec["query"] = latest_query
                rec["bad_tool_msgs"] = sum(r.count(s) for r in rec["tool_results"] for s in BAD_TOOL_STRINGS)
                messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning,
                                 "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
                messages.extend(tool_msgs)
                if self.use_ledger and need_update and ledger.constraints:
                    harvest(applied_at=n_toolcall_turns)
                    pending.append((n_toolcall_turns, executor.submit(
                        self.update_call, question=question, constraints=ledger.format_constraints_for_update(),
                        ledger_json=ledger.format_ledger_json(), thinking=latest_thinking,
                        search_query=latest_query, retrieval_results=latest_results)))
                turns.append(rec)
                continue
            # final answer (natural or forced)
            messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
            prediction = parse_boxed(content)
            turns.append({"call": call_idx, "turn": None, "kind": "forced_final" if forced else "answer",
                          "finish_reason": finish, "reasoning": reasoning, "content": content,
                          "attempt_log": attempt_log, "ledger_seen": seen, "ledger_state_seen": snap,
                          "has_boxed": "\\boxed" in content})
            termination = "forced_final" if forced else "answered"
            break

        if self.use_ledger:
            # after the answer: record the last updates, but leave the transcript as the agent saw it
            harvest(applied_at="end", block_all=True, touch_messages=False)
            executor.shutdown(wait=False)
        assert_question()
        latency = time.time() - t_start
        st = {"ok": 0, "empty": 0, "error": 0}
        for u in ledger_updates:
            st[u["status"]] = st.get(u["status"], 0) + 1
        return {
            "content": content, "prediction": prediction, "termination": termination,
            "turns": n_toolcall_turns, "n_agent_calls": len(turns),
            "n_search_calls": sum(1 for t in turns for c in t.get("tool_calls", []) if c["name"] == "search"),
            "n_browse_calls": sum(1 for t in turns for c in t.get("tool_calls", []) if c["name"] == "browse"),
            "n_queries": sum(len(c["arguments"].get("query", []) if isinstance(c["arguments"].get("query", []), list) else [1])
                             for t in turns for c in t.get("tool_calls", []) if c["name"] == "search"),
            "bad_tool_msgs": sum(t.get("bad_tool_msgs", 0) for t in turns),
            "latency": latency, "timing": timing,
            "messages": messages, "api_messages_final": api_messages(messages),
            "turn_records": turns, "ledger_updates": ledger_updates,
            "ledger": ledger.ledger if self.use_ledger else None,
            "ledger_constraints": ledger.constraints if self.use_ledger else None,
            "extract_record": extract_rec, "ledger_status_counts": st if self.use_ledger else None,
            "question_kept": True, "last_reasoning": last_reasoning,
            "n_repaired_tool_args": self.n_repaired_args, "runner_version": RUNNER_VERSION,
        }


def dataset_len(ds_dir, ds):
    with open(os.path.join(ds_dir, ds, "test_mcqa.jsonl")) as f:
        return sum(1 for _ in f)


def build_jobs(args):
    caps = {}
    for c in args.cap:
        k, v = c.split("=")
        caps[k] = int(v)
    per = {}
    for ds in args.datasets:
        n = dataset_len(args.dataset_dir, ds)
        idx = list(range(n)) if args.indices == ["all"] else [int(i) for i in args.indices if int(i) < n]
        if ds in caps:
            idx = [i for i in idx if i < caps[ds]]
        per[ds] = idx
    if args.order == "dataset":
        return [(ds, i) for ds in args.datasets for i in per[ds]]
    # interleave: round-robin over datasets in index order
    out, k = [], 0
    while any(k < len(v) for v in per.values()):
        for ds in args.datasets:
            if k < len(per[ds]):
                out.append((ds, per[ds][k]))
        k += 1
    return out


def generation_config(repo):
    try:
        ref = hf_revision(repo)
        return json.load(open(os.path.join(HF_HUB, "models--" + repo.replace("/", "--"), "snapshots", ref,
                                           "generation_config.json")))
    except Exception:
        return None


def load_item(ds_dir, ds, idx):
    with open(os.path.join(ds_dir, ds, "test_mcqa.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    it = rows[idx]
    if "question" not in it:
        it["question"] = it.pop("Question")
    if "answer" not in it:
        if "Answer" in it:
            it["answer"] = it.pop("Answer")
        else:
            it["answer"] = {"ground_truths": it["ground_truths"], "misc": it["misc"], "canary": it["canary"], "key": it["key"]}
    return it


def preflight(args):
    h = {"X-API-KEY": args.serper_api_key, "Content-Type": "application/json"}
    r = requests.post("https://google.serper.dev/search", headers=h, json={"q": "Eiffel Tower height", "num": 3},
                      timeout=60, verify=False)
    if r.status_code != 200 or "not enough credits" in r.text.lower() or not r.json().get("organic"):
        raise SystemExit(f"SERPER PREFLIGHT FAILED: HTTP {r.status_code} {r.text[:200]}")
    eng = SerperSearchEngine(serper_api_key=args.serper_api_key)
    out = eng.search_batch(["Eiffel Tower height"])
    if any(s in out for s in BAD_TOOL_STRINGS):
        raise SystemExit(f"SERPER PREFLIGHT FAILED (engine): {out[:200]}")
    b = JinaBrowser(jina_api_key=args.jina_api_key).browse("https://en.wikipedia.org/wiki/Eiffel_Tower")
    if b.startswith("Failed to read") or len(b) < 200:
        raise SystemExit(f"JINA PREFLIGHT FAILED: {b[:200]}")
    try:
        acct = requests.get("https://google.serper.dev/account", headers=h, timeout=30).json()
        print(f"serper account: balance={acct.get('balance')}", flush=True)
    except Exception:
        pass
    print("preflight ok: serper + jina", flush=True)


def main():
    urllib3.disable_warnings()
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", "-m", required=True)
    p.add_argument("--base_url", required=True)
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--no_ledger", action="store_true")
    p.add_argument("--ledger_model_name", default="dayoon/LiveLedger-4B")
    p.add_argument("--ledger_base_url", default=None)
    p.add_argument("--ledger_entries_cap", type=int, default=10)
    p.add_argument("--ledger_max_inflight", type=int, default=1)
    p.add_argument("--ledger_max_result_chars", type=int, default=40000)
    p.add_argument("--ledger_max_thinking_chars", type=int, default=12000)
    p.add_argument("--max_turns", type=int, default=30)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--agent_attempts", type=int, default=5,
                   help="attempts per agent call (exception, malformed tool args, harmony leak, empty content)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sampling", choices=["explicit", "server"], default="explicit",
                   help="explicit: send --temperature (gpt-oss, API models). server: omit temperature so vLLM applies "
                        "the model's generation_config (Qwen-style models).")
    p.add_argument("--reasoning_effort", default="high")
    p.add_argument("--tool_arg_fix", action="store_true",
                   help="accept url/link (browse) and queries/q (search); malformed calls get an error message, not \"\"")
    p.add_argument("--replay_reasoning_field", choices=["reasoning_content", "reasoning"], default="reasoning_content",
                   help="field name used for replayed reasoning; gpt-oss on vLLM needs 'reasoning'")
    p.add_argument("--replay_reasoning", action="store_true",
                   help="replay prior assistant reasoning_content in requests (Qwen-style models)")
    p.add_argument("--effort_control", choices=["reasoning_effort", "openrouter", "qwen_thinking", "none"],
                   default="reasoning_effort")
    p.add_argument("--forced_final", action="store_true",
                   help="at the turn cap, make one forced final-answer call. Default: stop with no answer (termination=turn_cap)")
    p.add_argument("--forced_final_max_tokens", type=int, default=16384)
    p.add_argument("--forced_final_fallback_effort", default="low",
                   help="effort for the last 2 attempts of the forced final call ('' = keep --reasoning_effort)")
    p.add_argument("--server_wait", type=float, default=1800,
                   help="seconds to wait for an unreachable agent/tracker server before the item becomes .error")
    p.add_argument("--openrouter", action="store_true")
    p.add_argument("--openrouter_provider_json", default='{"quantizations": ["fp8"]}',
                   help="provider routing for OpenRouter ('' or '{}' = none)")
    p.add_argument("--stop_file", default=None, help="if this file exists, no new item is started")
    p.add_argument("--order", choices=["dataset", "interleave"], default="interleave",
                   help="interleave: round-robin over datasets, so a partial run covers every dataset")
    p.add_argument("--cap", nargs="*", default=[], help="per-dataset item caps, e.g. bioasq=10")
    p.add_argument("--dry_run", action="store_true", help="print the item order and exit")
    p.add_argument("--serper_api_key", default=os.getenv("SERPER_API_KEY", ""))
    p.add_argument("--jina_api_key", default=os.getenv("JINA_API_KEY", ""))
    p.add_argument("--dataset_dir", default=os.path.join(REPO, "datasets"))
    p.add_argument("--datasets", "-d", nargs="+", default=DATASETS)
    p.add_argument("--indices", nargs="+", default=["all"], help="item indices, or 'all'")
    p.add_argument("--output_dir", "-o", required=True)
    p.add_argument("--num_workers", "-w", type=int, default=10)
    p.add_argument("--skip_preflight", action="store_true")
    p.add_argument("--ledger_feedback", choices=["on", "off"], default="on",
                   help="off: ledger message without the Feedbacks lines (used for GLM-5.2 + LiveLedger)")
    args = p.parse_args()
    global REPLAY_REASONING
    REPLAY_REASONING = args.replay_reasoning
    global REPLAY_FIELD
    REPLAY_FIELD = args.replay_reasoning_field
    global TOOL_ARG_FIX
    TOOL_ARG_FIX = args.tool_arg_fix
    global LEDGER_FEEDBACK
    LEDGER_FEEDBACK = args.ledger_feedback == "on"
    if args.api_key.startswith("env:"):  # keeps secrets out of `ps` output
        args.api_key = os.environ.get(args.api_key[4:], "")
    if not args.no_ledger and not args.ledger_base_url:
        p.error("--ledger_base_url required unless --no_ledger")
    pj = (args.openrouter_provider_json or "").strip()
    args.openrouter_provider = json.loads(pj) if pj and pj != "{}" else {}
    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps({"n_items": len(jobs), "first": jobs[:12]}))
        return
    if not args.skip_preflight:
        preflight(args)

    meta = {"runner_version": RUNNER_VERSION, "sampling": args.sampling,
            "generation_config": generation_config(args.model_name) if args.sampling == "server" else
            {"temperature": args.temperature},
            "n_items": len(jobs),
            "args": {k: v for k, v in vars(args).items() if "api_key" not in k},
            "agent_revision": hf_revision(args.model_name),
            "tracker_revision": None if args.no_ledger else hf_revision(args.ledger_model_name),
            "system_prompt": build_system_prompt(not args.no_ledger), "forced_final_msg": FORCED_FINAL_MSG,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(meta, open(os.path.join(args.output_dir, "run_meta.json"), "w"), indent=2)
    json.dump(meta, open(os.path.join(args.output_dir, f"run_meta_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=2)
    lock = Lock()

    def work(ds, idx):
        out = os.path.join(args.output_dir, ds, f"{idx}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            return ds, idx, "exists"
        if args.stop_file and os.path.exists(args.stop_file):
            return ds, idx, "stopped"
        if FATAL["reason"]:
            return ds, idx, "stopped"
        item = load_item(args.dataset_dir, ds, idx)
        t0 = time.time()
        try:
            res = Agent(args).run(item["question"])
            status = "success"
        except Exception as exc:
            res = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                   "infra_error": isinstance(exc, InfraError)}
            status = "error"
        res.update(runner_version=RUNNER_VERSION, question=item["question"], answer=item["answer"], dataset=ds, index=idx, status=status,
                   wall_time=time.time() - t0, args=meta["args"], agent_revision=meta["agent_revision"],
                   tracker_revision=meta["tracker_revision"])
        path = out if status == "success" else out + ".error"
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        os.rename(path + ".tmp", path)
        with lock:
            print(f"[done] {ds}/{idx} status={status} term={res.get('termination')} turns={res.get('turns')} "
                  f"wall={res['wall_time']:.0f}s pred={str(res.get('prediction'))[:80]!r} "
                  f"ledger={res.get('ledger_status_counts')} bad_tool={res.get('bad_tool_msgs')}", flush=True)
        return ds, idx, status

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(work, ds, i) for ds, i in jobs]
        for f in as_completed(futs):
            f.result()
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["fatal"] = FATAL["reason"]
    meta["search_errors_global"] = SEARCH_ERRORS
    json.dump(meta, open(os.path.join(args.output_dir, "run_meta.json"), "w"), indent=2)
    print("ALL DONE", SEARCH_ERRORS, "FATAL:", FATAL["reason"], flush=True)


if __name__ == "__main__":
    main()
