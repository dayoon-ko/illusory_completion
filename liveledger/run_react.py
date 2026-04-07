#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import urllib3
from openai import OpenAI

from prompt import SYSTEM_PROMPT_MAIN_WO_LEDGER
from tools import TOOLS_SEARCH
from search_engine import SerperSearchEngine, JinaBrowser
from utils import (
    ANSI_RESET,
    ANSI_RESPONSE,
    ANSI_THINK,
    ANSI_TOOL_CALL,
    ANSI_TOOL_MSG,
    ANSI_USER,
    log_event,
    parse_boxed,
)

logger = logging.getLogger(__name__)


def build_openai_client(base_url: str, api_key: str) -> OpenAI:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=0,
        timeout=300.0,
        http_client=httpx.Client(verify=ssl_context),
    )


def _parse_tool_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        return json.loads(raw_args)

    raise ValueError(f"Unsupported tool arguments type: {type(raw_args).__name__}")


def extract_cleaned_queries(tool_call_arguments: Any, min_length: int = 2) -> List[str]:
    raw = _parse_tool_args(tool_call_arguments)
    raw_queries = raw.get("query")
    if raw_queries is None:
        raw_queries = raw.get("queries")

    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]

    if not isinstance(raw_queries, list):
        raise ValueError("Tool arguments must contain 'query' or 'queries' as a string or list.")

    cleaned_queries = [
        q.strip()
        for q in raw_queries
        if isinstance(q, str) and len(q.strip()) >= min_length
    ]
    if not cleaned_queries:
        raise ValueError(f"No valid queries extracted from tool arguments: {raw}")
    return cleaned_queries


def extract_cleaned_urls(tool_call_arguments: Any) -> List[str]:
    raw = _parse_tool_args(tool_call_arguments)
    raw_urls = raw.get("urls")
    if raw_urls is None:
        raw_urls = raw.get("url")

    if isinstance(raw_urls, str):
        raw_urls = [raw_urls]

    if not isinstance(raw_urls, list):
        raise ValueError("Tool arguments must contain 'urls' or 'url' as a string or list.")

    cleaned_urls = [
        u.strip()
        for u in raw_urls
        if isinstance(u, str) and u.strip().startswith(("http://", "https://"))
    ]
    if not cleaned_urls:
        raise ValueError(f"No valid URLs extracted from tool arguments: {raw}")
    return cleaned_urls


def send_completion_request(
    *,
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    reasoning_effort: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int = 3,
    retry_base_delay_sec: float = 1.0,
    retry_max_delay_sec: float = 10.0,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"reasoning_effort": reasoning_effort},
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=TOOLS_SEARCH,
            )
            return response.model_dump()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = min(retry_base_delay_sec * (2 ** attempt), retry_max_delay_sec)
            logger.warning(
                "Completion request failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Completion request failed after {max_retries} retries."
    ) from last_exc


def validate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []

    for tool_call in tool_calls:
        fn = tool_call.get("function") or {}
        fn_name = fn.get("name")
        raw_args = fn.get("arguments")

        if fn_name == "search":
            cleaned_queries = extract_cleaned_queries(raw_args)
            normalized_args = {"query": cleaned_queries}
        elif fn_name == "browse":
            cleaned_urls = extract_cleaned_urls(raw_args)
            normalized_args = {"urls": cleaned_urls}
        else:
            logger.warning("Skipping unsupported tool call: %s", fn)
            continue

        normalized_tool_call = dict(tool_call)
        normalized_tool_call["function"] = {
            **fn,
            "arguments": json.dumps(normalized_args, ensure_ascii=False),
        }
        validated.append(normalized_tool_call)

    return validated


def run_turn(
    *,
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    reasoning_effort: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_query_retries: int,
) -> Dict[str, Any]:
    last_resp_json: Optional[Dict[str, Any]] = None

    for query_retry_count in range(max_query_retries):
        try:
            resp_json = send_completion_request(
                client=client,
                model_name=model_name,
                messages=messages,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            last_resp_json = resp_json
            choice = resp_json["choices"][0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")
            if finish_reason == "tool_calls":
                # Validate JSON parsability only
                tc = (message.get("tool_calls") or [None])[0]
                if tc:
                    json.loads(tc["function"]["arguments"])
                    break
                else:
                    continue
            else:
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                if not content and not reasoning:
                    continue
                else:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "run_turn request failed (attempt %d/%d): %s",
                query_retry_count + 1,
                max_query_retries,
                exc,
            )
            continue

    if last_resp_json is not None:
        return last_resp_json
    return {
        "choices": [{
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
        }]
    }


def run_model(
    *,
    question: str,
    args: argparse.Namespace,
    search_engine: SerperSearchEngine,
    browser: JinaBrowser,
) -> Tuple[str, List[Dict[str, Any]], str, int, float]:
    client = build_openai_client(args.base_url, args.api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": args.system_message},
        {"role": "user", "content": question},
    ]

    log_event("USER", question, ANSI_USER)

    turn = 0
    start_time = time.time()

    while True:
        turn += 1
        resp_json = run_turn(
            client=client,
            model_name=args.model_name,
            messages=messages,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_query_retries=args.max_query_retries,
        )

        choice = resp_json["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        reasoning_content = message.get("reasoning_content") or ""
        response_content = message.get("content") or ""

        log_event("THINK", reasoning_content, ANSI_THINK)
        log_event("RESPONSE", response_content, ANSI_RESPONSE)

        if finish_reason == "tool_calls":
            tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []
            if not tool_calls:
                logger.warning("tool_calls finish reason returned without tool calls; continuing.")
                continue

            tool_messages: List[Dict[str, Any]] = []
            for tool_call in tool_calls:
                fn = tool_call.get("function") or {}
                fn_name = fn.get("name")
                tool_call_arguments = fn.get("arguments") or "{}"

                log_event(
                    "TOOL_CALL",
                    f"{fn_name}({tool_call_arguments})",
                    ANSI_TOOL_CALL,
                )

                fn_args = _parse_tool_args(tool_call_arguments)

                if fn_name == "search":
                    queries = fn_args.get("query", fn_args.get("queries", []))
                    if isinstance(queries, str):
                        queries = [queries]
                    tool_result = search_engine.search_batch(queries)
                elif fn_name == "browse":
                    urls = fn_args.get("urls", fn_args.get("url", []))
                    if isinstance(urls, str):
                        urls = [urls]
                    tool_result = browser.browse_batch(urls)
                else:
                    continue

                tool_message: Dict[str, Any] = {"role": "tool", "content": tool_result}
                tool_call_id = tool_call.get("id")
                if isinstance(tool_call_id, str) and tool_call_id.strip():
                    tool_message["tool_call_id"] = tool_call_id

                tool_messages.append(tool_message)
                log_event("TOOL_MSG", tool_result, ANSI_TOOL_MSG)

            assistant_message = {
                "role": "assistant",
                "content": response_content,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_message)
            messages.extend(tool_messages)

            if turn >= args.max_turns:
                return response_content, messages, "", turn, time.time() - start_time
            continue

        if finish_reason in {"stop", "length"}:
            prediction = parse_boxed(response_content) or parse_boxed(reasoning_content)
            latency_sec = time.time() - start_time
            return response_content, messages, prediction, turn, latency_sec

        return response_content, messages, "", turn, time.time() - start_time


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
                    "key": item["key"],
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
    args: argparse.Namespace,
    search_engine: SerperSearchEngine,
    browser: JinaBrowser,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
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
        content, messages, prediction, turns, latency = run_model(
            question=question,
            args=args,
            search_engine=search_engine,
            browser=browser,
        )

        result = {
            "question": question,
            "answer": answer,
            "content": content,
            "messages": messages,
            "prediction": prediction,
            "turns": turns,
            "latency": latency,
            "elapsed_time": latency,
            "status": "success",
        }

        temp_path = output_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(temp_path, output_path)

        with progress_lock:
            completed_count += 1
            print(f"\n{'=' * 60}")
            print(f"[{completed_count}/{total_count}] Completed item {idx}")
            print(f"Question: {question[:100]}...")
            print(f"Prediction: {prediction}")
            print(f"Turns: {turns}, Time: {latency:.1f}s")
            print(f"Saved to {output_path}")

        return result

    except Exception as e:  # noqa: BLE001
        error_result = {
            "question": question,
            "answer": answer,
            "error": str(e),
            "status": "error",
        }

        error_path = output_path + ".error"
        with open(error_path, "w") as f:
            json.dump(error_result, f, indent=2)

        with progress_lock:
            completed_count += 1
            print(f"\n[{completed_count}/{total_count}] ERROR on item {idx}: {e}")

        return error_result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline agent on a single query or dataset."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--question", default=None, help="Single question mode.")

    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument(
        "--dataset_names",
        "-d",
        nargs="+",
        type=str,
        default=["frames", "browsecomp", "deepsearchqa", "livedrbench", "webwalkerqa"],
        choices=["frames", "browsecomp", "deepsearchqa", "livedrbench", "webwalkerqa"],
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default="outputs_baseline")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for processing",
    )

    parser.add_argument("--model_name", "-m", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        dest="reasoning_effort",
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    parser.add_argument("--max-tokens", type=int, default=32768, dest="max_tokens")
    parser.add_argument(
        "--system-message",
        default=SYSTEM_PROMPT_MAIN_WO_LEDGER,
        dest="system_message",
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000/v1",
        dest="base_url",
        help="OpenAI-compatible chat/completions endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        dest="api_key",
        help="API key for the endpoint.",
    )
    parser.add_argument("--max-turns", type=int, default=30, dest="max_turns")
    parser.add_argument(
        "--max-query-retries",
        type=int,
        default=10,
        dest="max_query_retries",
    )

    parser.add_argument(
        "--jina-api-key",
        default=os.getenv("JINA_API_KEY", ""),
        dest="jina_api_key",
        help="Jina API key for browsing pages.",
    )
    parser.add_argument(
        "--serper-api-key",
        default=os.getenv("SERPER_API_KEY", ""),
        dest="serper_api_key",
        help="Serper API key for web search.",
    )
    parser.add_argument("--top-k", type=int, default=10, dest="top_k")
    parser.add_argument("--search-timeout", type=int, default=300, dest="search_timeout")
    parser.add_argument(
        "--search-max-workers",
        type=int,
        default=1,
        dest="search_max_workers",
    )
    parser.add_argument("--browse-timeout", type=int, default=300, dest="browse_timeout")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global total_count, completed_count

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if not args.serper_api_key:
        raise ValueError("SERPER_API_KEY is required for Serper search.")
    if not args.jina_api_key:
        raise ValueError("JINA_API_KEY is required for Jina browsing.")

    if args.question:
        search_engine = SerperSearchEngine(
            serper_api_key=args.serper_api_key,
            topk=args.top_k,
            request_timeout=args.search_timeout,
            max_workers=args.search_max_workers,
        )
        browser = JinaBrowser(
            jina_api_key=args.jina_api_key,
            request_timeout=args.browse_timeout,
        )

        question = args.question.strip()
        if not question:
            raise ValueError("Question must be a non-empty string.")

        content, messages, prediction, turns, latency_sec = run_model(
            question=question,
            args=args,
            search_engine=search_engine,
            browser=browser,
        )

        logger.info("Completed in %d turns (%.2fs).", turns, latency_sec)
        if prediction.strip() and prediction.strip() != content.strip():
            logger.info("Parsed answer: %s", prediction.strip())
        return 0

    for dataset_name in args.dataset_names:
        args.dataset_name = dataset_name
        dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
        data_path = os.path.join(dataset_dir, "test_mcqa.jsonl")
        dataset = DataLoader(data_path, args.start_idx, args.end_idx).load_data()

        output_dir = os.path.join(args.output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        total_count = len(dataset)
        completed_count = 0

        print(f"{'=' * 60}")
        print(f"Processing {total_count} items with {args.num_workers} worker(s)")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 60}\n")

        def create_search_engine() -> SerperSearchEngine:
            return SerperSearchEngine(
                serper_api_key=args.serper_api_key,
                topk=args.top_k,
                request_timeout=args.search_timeout,
                max_workers=args.search_max_workers,
            )

        def create_browser() -> JinaBrowser:
            return JinaBrowser(
                jina_api_key=args.jina_api_key,
                request_timeout=args.browse_timeout,
            )

        start_time = time.time()

        if args.num_workers == 1:
            search_engine = create_search_engine()
            browser = create_browser()
            results: List[Optional[Dict[str, Any]]] = []
            for idx, item in enumerate(dataset):
                result = process_item(idx, item, args, search_engine, browser, output_dir)
                results.append(result)
        else:
            search_engines = [create_search_engine() for _ in range(args.num_workers)]
            browsers = [create_browser() for _ in range(args.num_workers)]

            results = []
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {}
                for idx, item in enumerate(dataset):
                    worker_idx = idx % args.num_workers
                    future = executor.submit(
                        process_item,
                        idx,
                        item,
                        args,
                        search_engines[worker_idx],
                        browsers[worker_idx],
                        output_dir,
                    )
                    futures[future] = idx

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:  # noqa: BLE001
                        print(f"[ERROR] Future for item {idx} raised exception: {e}")
                        results.append({"idx": idx, "status": "error", "error": str(e)})

        total_time = time.time() - start_time

        successful = sum(1 for r in results if r and r.get("status") == "success")
        errors = sum(1 for r in results if r and r.get("status") == "error")
        skipped = sum(1 for r in results if r is None)

        print(f"\n{'=' * 60}")
        print("COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total items: {total_count}")
        print(f"Successful: {successful}")
        print(f"Errors: {errors}")
        print(f"Skipped (already existed): {skipped}")
        print(f"Total time: {total_time:.1f}s")
        if successful > 0:
            print(f"Average time per successful item: {total_time / successful:.1f}s")

        summary = {
            "total_items": total_count,
            "successful": successful,
            "errors": errors,
            "skipped": skipped,
            "elapsed_time": total_time,
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
