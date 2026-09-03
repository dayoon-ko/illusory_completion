"""Unified tag-based search baseline runner.

Supports baselines that generate text with search tags (e.g., <search>query</search>),
execute web search, and inject results back. Covers Search-R1, SmartSearch, RAG-R1,
ReSeek, HiPRAG, and similar methods.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import requests
import torch
import transformers
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASELINE_CONFIGS = {
    "search-r1": {
        "model": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        "search_tag": ("<search>", "</search>"),
        "result_tag": ("<information>", "</information>"),
        "answer_tag": ("<answer>", "</answer>"),
        "prompt": (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between "
            "<information> and </information>. You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer "
            "inside <answer> and </answer>, without detailed illustrations. "
            "For example, <answer> Beijing </answer>. Question: {question}\n"
        ),
    },
    "smartsearch": {
        "model": "vvv111222/SmartSearch-3B",
        "search_tag": ("<search>", "</search>"),
        "result_tag": ("<result>", "</result>"),
        "answer_tag": ("<answer>", "</answer>"),
        "prompt": (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between "
            "<result> and </result>. You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer "
            "inside <answer> and </answer>, without detailed illustrations. "
            "For example, <answer> Beijing </answer>. Question: {question}\n"
        ),
    },
    "rag-r1": {
        "model": "yaoyueduzhen/RAG-R1-mq-7b",
        "search_tag": ("<search>", "</search>"),
        "result_tag": ("<information>", "</information>"),
        "answer_tag": ("<answer>", "</answer>"),
        "prompt": (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between "
            "<information> and </information>. You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer "
            "inside <answer> and </answer>, without detailed illustrations. "
            "For example, <answer> Beijing </answer>. Question: {question}\n"
        ),
    },
    "reseek": {
        "model": "TencentBAC/ReSeek-qwen2.5-7b-em-grpo",
        "search_tag": ("<search>", "</search>"),
        "result_tag": ("<information>", "</information>"),
        "answer_tag": ("<answer>", "</answer>"),
        "prompt": (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between "
            "<information> and </information>. You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer "
            "inside <answer> and </answer>, without detailed illustrations. "
            "For example, <answer> Beijing </answer>. Question: {question}\n"
        ),
    },
    "hiprag": {
        "model": "qualidea1217/Qwen2.5-7B-Instruct-GRPO-HiPRAG",
        "search_tag": ("<search>", "</search>"),
        "result_tag": ("<context>", "</context>"),
        "answer_tag": ("</answer>",),
        "prompt": (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by "
            "<search> query </search> and it will return the top searched results between "
            "<context> and </context>. You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer "
            "inside <answer> and </answer>. Question: {question}\n"
        ),
    },
}


def jina_web_search(query: str, api_key: str, top_k: int = 10, max_retries: int = 8) -> str:
    search_url = f"https://s.jina.ai/{query.replace(' ', '+')}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Respond-With": "no-content",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(search_url, headers=headers, verify=False, timeout=10)
            data = resp.json()
            results = []
            for e in data.get("data", [])[:top_k]:
                results.append(f"Doc (Title: {e.get('title', '')}) {e.get('description', '')}")
            return "\n".join(results) if results else "No results found."
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(min(0.5 * (2 ** attempt), 10))
    return "Search failed."


def serper_web_search(query: str, api_key: str, top_k: int = 10) -> str:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers=headers, json={"q": query, "num": top_k},
            timeout=30, verify=False,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic", [])[:top_k]
        return "\n".join(
            f"[{i+1}] {r.get('title', '')} ({r.get('link', '')})\n    {r.get('snippet', '')}"
            for i, r in enumerate(organic)
        ) or "No results found."
    except Exception:
        return "Search failed."


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(s, add_special_tokens=False) for s in target_sequences]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(self.target_ids):
            t = torch.as_tensor(target, device=input_ids.device)
            if torch.equal(input_ids[0, -self.target_lengths[i]:], t):
                return True
        return False


def extract_query(text: str, search_close_tag: str) -> str | None:
    tag_name = search_close_tag.strip("<>/")
    pattern = re.compile(rf"<{tag_name}>(.*?){re.escape(search_close_tag)}", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None


def run_baseline(args):
    cfg = BASELINE_CONFIGS[args.baseline]
    model_name = args.model or cfg["model"]
    search_open, search_close = cfg["search_tag"]
    result_open, result_close = cfg["result_tag"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    device = next(model.parameters()).device

    stop_variants = [search_close, f" {search_close}", f"{search_close}\n"]
    stopping_criteria = transformers.StoppingCriteriaList([
        StopOnSequence(stop_variants, tokenizer)
    ])
    eos_ids = set()
    for tok_name in ["eos_token_id", "eot_token_id"]:
        tid = getattr(tokenizer, tok_name, None)
        if tid is not None:
            eos_ids.add(tid)

    search_fn = (
        lambda q: serper_web_search(q, args.serper_api_key)
        if args.search_engine == "serper"
        else lambda q: jina_web_search(q, args.jina_api_key)
    )

    for dataset_name in args.datasets:
        data_path = os.path.join(args.dataset_dir, dataset_name, "test_mcqa.jsonl")
        with open(data_path) as f:
            dataset = [json.loads(line) for line in f]

        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        done = set()
        output_path = os.path.join(output_dir, "results.jsonl")
        if os.path.exists(output_path):
            with open(output_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["question"])
                    except Exception:
                        pass

        print(f"[{dataset_name}] {len(done)} done, {len(dataset) - len(done)} remaining")

        for item in dataset:
            if item["question"] in done:
                continue

            prompt_text = cfg["prompt"].format(question=item["question"])
            if tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True, tokenize=False,
                )

            for step in range(args.max_steps):
                input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
                outputs = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=1024,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                )

                gen_tokens = outputs[0][input_ids.shape[1]:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                if outputs[0][-1].item() in eos_ids or step >= args.max_steps - 1:
                    prompt_text += f"\n\n{gen_text}"
                    break

                query = extract_query(
                    tokenizer.decode(outputs[0], skip_special_tokens=True),
                    search_close,
                )
                if query:
                    results = search_fn(query)
                else:
                    results = ""

                prompt_text += f"\n\n{gen_text}{result_open}{results}{result_close}\n\n"

            item["output"] = prompt_text
            with open(output_path, "a") as f:
                f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", "-b", required=True, choices=list(BASELINE_CONFIGS.keys()))
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--datasets", "-d", nargs="+", default=["frames", "browsecomp", "deepsearchqa", "livedrbench", "webwalkerqa", "bioasq"])
    parser.add_argument("--dataset_dir", default="../datasets")
    parser.add_argument("--output_dir", "-o", default="outputs")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--search_engine", choices=["jina", "serper"], default="jina")
    parser.add_argument("--jina_api_key", default=os.getenv("JINA_API_KEY", ""))
    parser.add_argument("--serper_api_key", default=os.getenv("SERPER_API_KEY", ""))
    args = parser.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
