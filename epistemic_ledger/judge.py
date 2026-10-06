"""Answer correctness (paper setting: gpt-5.6-sol). Adds `is_correct` to every ledger file written by build_ledger.py.

The answer judged is the final answer of the trajectory: `content` for liveledger/run.py outputs; for other agents it
is extracted per format (Search-R1 <answer> tags, WebExplorer last message, ...; see EXTRACTORS, keyed on the agent
folder name). No answer (nothing extracted, or an answer forced after the turn cap) is incorrect without a call. LiveDRBench gold answers are stored encrypted and are decrypted here.
Items that already have `is_correct` are skipped (use --force to re-judge).

    export OPENAI_API_KEY=...
    python epistemic_ledger/judge.py --ledger_dir ledgers -a gpt-oss-120b_liveledger
"""
import argparse
import base64
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from glob import glob

from openai import OpenAI

from judge_prompts import BIOASQ_PROMPT, PROMPT

USAGE, LOCK = {"in": 0, "out": 0, "calls": 0, "errors": 0}, threading.Lock()


def derive_key(password: str, length: int) -> bytes:
    """From https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py"""
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """From https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py"""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def gold_answer(d):
    ans = d.get("answer", d.get("Answer", ""))
    if isinstance(ans, dict) and "ground_truths" in ans:          # LiveDRBench, as stored by liveledger/run.py
        return decrypt(ans["ground_truths"], ans["canary"])
    if not ans and d.get("ground_truths") and d.get("canary"):   # LiveDRBench row copied as-is
        return decrypt(d["ground_truths"], d["canary"])
    return ans


# ---------------------------------------------------------------- final answer, per trajectory format
def _answer_react(d):
    """liveledger/run.py and other ReAct-style outputs: the final `content` (else `prediction`)."""
    pred = d.get("content") or d.get("prediction") or ""
    if not pred and d.get("messages"):
        last = d["messages"][-1]
        if last.get("role") == "assistant":
            pred = (last.get("content") or "").split("</think>")[-1].strip()
    return pred


def _answer_tag(d):
    """Search-R1 / RAG-R1: last <answer>...</answer> of the raw trajectory string."""
    found = re.findall(r"<answer>(.*?)</answer>", str(d.get("output", "")).split("</think>")[-1], re.DOTALL)
    return found[-1] if found else ""


def _answer_asearcher(d):
    out = d["output"]["thinking_blocks"][-1]
    return out.split("prediction of the answer: ")[-1].strip() if "prediction of the answer:" in out else ""


def _answer_last_message(d):
    """WebExplorer / TongyiDR: the last assistant message after </think>."""
    if not d.get("messages") or d["messages"][-1]["role"] != "assistant":
        return ""
    return d["messages"][-1]["content"].split("</think>")[-1].strip()


def _answer_search_o1(d):
    pred = d["history"][-1]
    if "boxed" in pred:
        return pred.split("boxed")[-1].strip()
    if "assistantfinal" in pred:
        return pred.split("assistantfinal")[-1].strip()
    return pred


def _answer_blocks(d):
    """Any other {thinking_blocks, query_blocks, results_blocks} trajectory."""
    out = d.get("output", {})
    if isinstance(out, dict) and out.get("content"):
        return out["content"]
    if isinstance(out, dict) and "prediction" in out:
        return out["prediction"]
    if d.get("prediction"):
        return d["prediction"]
    if isinstance(out, dict) and out.get("thinking_blocks"):
        last = out["thinking_blocks"][-1]
        for marker in ("prediction of the answer:", "final answer:", "answer:"):
            if marker in last.lower():
                return last.lower().split(marker)[-1].strip()
        return last.strip()
    return d.get("content") or ""


EXTRACTORS = {"search-r1": _answer_tag, "rag-r1": _answer_tag, "asearcher": _answer_asearcher,
              "webexplorer": _answer_last_message, "tongyidr": _answer_last_message,
              "tongyidr-liveledger-4B": _answer_last_message,
              "search_o1_gpt-oss-20b": _answer_search_o1, "search_o1_gpt-oss-120b": _answer_search_o1,
              "dr-tulu": lambda d: d.get("final_response", "")}


def final_answer(d, agent):
    """Answer text to judge; "" = no answer. An answer forced after the turn cap counts as no answer."""
    if "forced" in str(d.get("termination") or ""):
        return ""
    base = re.sub(r"_(\d+|rerun\d+)$", "", agent)
    if base in EXTRACTORS:
        fn = EXTRACTORS[base]
    elif "turn_records" in d or "content" in d or "messages" in d and not isinstance(d.get("output"), dict):
        fn = _answer_react
    else:
        fn = _answer_blocks
    try:
        return str(fn(d) or "")
    except (KeyError, IndexError, TypeError):
        return ""


def verdict(client, model, question, gold, pred, dataset):
    tmpl = BIOASQ_PROMPT if dataset == "bioasq" else PROMPT
    prompt = tmpl.format(question=question, answer=gold, predicted_answer=pred.split("<tool_call>")[0].strip())
    last = None
    for _ in range(5):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
            with LOCK:
                USAGE["in"] += r.usage.prompt_tokens
                USAGE["out"] += r.usage.completion_tokens
                USAGE["calls"] += 1
            txt = (r.choices[0].message.content or "").replace("```json", "").replace("```", "")
            m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", txt, re.DOTALL)
            v = json.loads(m.group(0) if m else txt)
            if isinstance(v.get("verdict"), bool):
                return v["verdict"], txt.strip()
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    with LOCK:
        USAGE["errors"] += 1
    raise RuntimeError(f"judge failed 5x: {last}")


def judge_file(path, client, model, force):
    d = json.load(open(path))
    if "is_correct" in d and not force:
        return
    dataset = os.path.basename(os.path.dirname(path))
    agent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    pred = final_answer(d, agent)
    if not pred.strip():
        d.update(is_correct=False, verdict_raw="no answer (not judged)", judge=model)
    else:
        ok, txt = verdict(client, model, d.get("question", ""), gold_answer(d), pred, dataset)
        d.update(is_correct=ok, verdict_raw=txt, judge=model)
    tmp = path + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger_dir", required=True)
    ap.add_argument("--agents", "-a", nargs="+", required=True)
    ap.add_argument("--model_name", default="gpt-5.6-sol")
    ap.add_argument("--base_url", default="https://api.openai.com/v1")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="re-judge items that already have is_correct")
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=600)
    files = [p for ag in a.agents for p in glob(os.path.join(a.ledger_dir, ag, "*", "item_*.json"))]
    print(len(files), "ledger files")

    def run(p):
        try:
            judge_file(p, client, a.model_name, a.force)
        except Exception as e:  # noqa: BLE001
            print("ERROR", p, str(e)[:200], flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, files))
    print("DONE", USAGE)


if __name__ == "__main__":
    main()
