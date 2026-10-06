<h1 align="center">When Is Enough Not Enough?<br> Illusory Completion 🧠 in Search Agents 🔍</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2602.07549)
[![Model](https://img.shields.io/badge/Model-LiveLedger--4B-ffd21e.svg?logo=huggingface)](https://huggingface.co/dayoon/LiveLedger-4B)

</div>

Code for the paper **"When Is Enough Not Enough? Illusory Completion in Search Agents"**.

Search agents often **stop while some constraints of the question are still unverified** (illusory completion).
This repo has three parts:

| Part | What it does |
|---|---|
| **Run an agent** (`liveledger/`) | ReAct agent with `search` + `browse`, with or without the **LiveLedger** tracker (a 4B model that shows the agent which constraints are verified) |
| **Evaluate** (`epistemic_ledger/`) | Builds the **Epistemic Ledger** of any trajectory, judges the answer, and reports **Acc** and **UAR** |
| **Train the tracker** (`training/`) | How `dayoon/LiveLedger-4B` was trained |

```
 question ──▶ agent (+ LiveLedger) ──▶ trajectory ──▶ Epistemic Ledger ──▶ judge ──▶ Acc, UAR, constraint outcomes
              liveledger/run.py                       build_ledger.py        judge.py   metrics.py
```

---

## Repository Overview

```
.
├── liveledger/            # run an agent (with / without the tracker)
│   ├── run.py             #   the runner (one JSON per question)
│   ├── ledger.py          #   the ledger table the agent sees
│   ├── prompts.py         #   tracker prompts (constraint extraction, ledger update)
│   ├── tools.py           #   tool schemas
│   └── search.py          #   Serper search + Jina page reader
├── epistemic_ledger/      # evaluate trajectories
│   ├── build_ledger.py    #   1. Epistemic Ledger (gpt-5-nano)
│   ├── judge.py           #   2. answer correctness (gpt-5.6-sol)
│   ├── metrics.py         #   3. Acc, UAR, Verified / Assumed / Refuted / Unchecked
│   ├── ledger_merge.py    #   keeps evidence once found (used for every number)
│   ├── prompts.py         #   evaluator prompts
│   └── judge_prompts.py   #   judge prompts
├── training/              # train the 4B tracker
├── baselines/             # runner for tag-based agents (Search-R1, RAG-R1, ...) + example outputs
└── datasets/              # the 484 questions (6 benchmarks)
```

---

## Run it in 4 steps

```bash
# 1. install
git clone https://github.com/dayoon-ko/illusory_completion.git && cd illusory_completion
pip install -r requirements.txt            # + `pip install vllm` on the GPU machine that serves the models

# 2. keys
export SERPER_API_KEY=...                  # web search  (serper.dev)
export JINA_API_KEY=...                    # page reader (jina.ai/reader)
export OPENAI_API_KEY=...                  # evaluation only (gpt-5-nano ledger, gpt-5.6-sol judge)

# 3. serve the agent and the tracker (two vLLM servers)
vllm serve openai/gpt-oss-120b --port 8000 --enable-auto-tool-choice --tool-call-parser openai --enable-prefix-caching
vllm serve dayoon/LiveLedger-4B --port 8100 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3

# 4. run both arms on all 484 questions
python liveledger/run.py -m openai/gpt-oss-120b --base_url http://localhost:8000/v1 --tool_arg_fix \
  --ledger_base_url http://localhost:8100/v1 -o outputs/gpt-oss-120b_liveledger
python liveledger/run.py -m openai/gpt-oss-120b --base_url http://localhost:8000/v1 --tool_arg_fix \
  --no_ledger -o outputs/gpt-oss-120b
```

Then evaluate:

```bash
python epistemic_ledger/build_ledger.py -i outputs -a gpt-oss-120b gpt-oss-120b_liveledger -o ledgers
python epistemic_ledger/judge.py --ledger_dir ledgers -a gpt-oss-120b gpt-oss-120b_liveledger
python epistemic_ledger/metrics.py --ledger_dir ledgers
```

`metrics.py` prints one row per agent: `Acc`, `UAR`, then `Verified / Assumed / Refuted / Unchecked`.

> **Try it small first:** add `-d frames --indices 0 1 2` to `run.py` (3 questions).
> Every script **skips finished items**, so re-running the same command resumes it.

---

## Paper settings

Defaults of `run.py` = the paper: **30 tool-calling turns**, no forced answer at the cap, temperature 1.0,
reasoning effort `high`, all 484 questions. Add `--ledger_base_url ...` for **+ LiveLedger**, or `--no_ledger` for the base agent.

| Agent | Extra flags |
|---|---|
| gpt-oss-20b / gpt-oss-120b (vLLM) | `-m openai/gpt-oss-120b --base_url http://localhost:8000/v1 --tool_arg_fix` |
| DeepSeek-V4-Pro (OpenRouter) | `-m deepseek/deepseek-v4-pro --base_url https://openrouter.ai/api/v1 --api_key env:OPENROUTER_API_KEY --openrouter --effort_control openrouter --openrouter_provider_json '{"order":["novita","siliconflow"],"allow_fallbacks":false,"quantizations":["fp8"]}'` |
| GLM-5.2 (OpenRouter) | `-m z-ai/glm-5.2 --base_url https://openrouter.ai/api/v1 --api_key env:OPENROUTER_API_KEY --openrouter --effort_control openrouter` (+ LiveLedger arm: also `--ledger_feedback off`) |
| Agents-A1 (vLLM) | `-m InternScience/Agents-A1 --base_url http://localhost:8000/v1 --sampling server --effort_control qwen_thinking --replay_reasoning` (serve with `--max-model-len 131072 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`) |

Evaluation: `gpt-5-nano` builds the ledger (reasoning effort `medium`, first 30 turns), `gpt-5.6-sol` judges the answer.
These are the defaults of `build_ledger.py` and `judge.py`.

Trained agents (Search-R1, ASearcher, RAG-R1, DR-Tulu, WebExplorer, TongyiDR) and Search-o1 run with their own code;
`baselines/run_tag_search.py` runs the tag-based ones. `build_ledger.py` reads their saved trajectories as
`<input_dir>/<agent>/<dataset>.jsonl` (format and examples: `baselines/results/`).

---

## I want to …

| … | Do this |
|---|---|
| run one benchmark only | `-d browsecomp` (`browsecomp`, `deepsearchqa`, `frames`, `livedrbench`, `webwalkerqa`, `bioasq`) |
| run a few questions | `--indices 0 1 2` |
| run more questions in parallel | `-w 20` (`run.py`), `-w 128` (`build_ledger.py`), `--workers 128` (`judge.py`) |
| force a final answer at the turn cap | `--forced_final` (the paper does **not**; `metrics.py` counts such answers as no answer) |
| use a different tracker server or model | `--ledger_base_url http://host:port/v1 --ledger_model_name <name>` |
| evaluate with a local model instead of OpenAI | `--model_name <name> --base_url http://localhost:8000/v1 --api_key EMPTY` (both `build_ledger.py` and `judge.py`) |
| see results per benchmark | `python epistemic_ledger/metrics.py --ledger_dir ledgers --per_dataset` |
| save the table | `--out_csv results.csv` |
| evaluate another agent's trajectories | save them as `<dir>/<agent>/<dataset>.jsonl` with `output = {thinking_blocks, query_blocks, results_blocks}` and run the 3 evaluation commands with `-i <dir> -a <agent>` |
| train the tracker | see [`training/`](training/) |

---

## Words used

| Word | Meaning |
|---|---|
| **constraint** | one condition the answer must meet (e.g. "erected in the 19th century") |
| **Epistemic Ledger** | per (candidate, constraint): what the retrieved evidence shows (`obj` = true / false / null) and what the agent states (`per`) |
| **committed candidate** | the answer candidate the agent ends with |
| **Verified** | evidence establishes the constraint |
| **Assumed** | no evidence, but the agent states it holds |
| **Refuted** | evidence contradicts the constraint, but the answer is kept |
| **Unchecked** | no evidence, and the agent never addresses it |
| **UAR** | unsubstantiated answer rate: % of questions whose final answer is not fully verified (no answer counts too) |
| **LiveLedger** | the 4B tracker: after every `search` / `browse` it updates the ledger and shows it to the agent as a table |

---

## Outputs

| File | Written by | Contents |
|---|---|---|
| `outputs/<agent>/<dataset>/<i>.json` | `run.py` | full transcript, per-turn records, every ledger update, final answer (`content`), `termination` |
| `outputs/<agent>/run_meta.json` | `run.py` | all settings + the exact system prompt |
| `ledgers/<agent>/<dataset>/item_<i>.json` | `build_ledger.py`, `judge.py` | checklist, per-turn ledgers, `is_correct` |

`termination` = `answered`, `turn_cap` (no answer within 30 turns) or `agent_error`.

---

## Citation

```bibtex
@misc{ko2026enoughillusorycompletionsearch,
      title={When Is Enough Not Enough? Illusory Completion in Search Agents},
      author={Dayoon Ko and Jihyuk Kim and Sohyeon Kim and Haeju Park and Dahyun Lee and Gunhee Kim and Moontae Lee and Kyungjae Lee},
      year={2026},
      eprint={2602.07549},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.07549},
}
```

## License

Apache 2.0. The questions in `datasets/` come from BrowseComp, DeepSearchQA, FRAMES, LiveDRBench, WebWalkerQA and BioASQ; their own licenses apply.

## Contact

Dayoon Ko · dayoon.ko@vision.snu.ac.kr
