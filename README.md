<h1 align="center">When is Enough Not Enough?<br> Illusory Completion 🧠 in Search Agents 🔍</h1>

<div align="center"> 
    
[![Static Badge](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2602.07549)

</div>

This repository contains implementations of the paper **"Is Enough Not Enough? Illusory Completion in Search Agents."**

We present a novel framework for analyzing failure modes in agentic search systems through **epistemic ledger tracking**, which systematically evaluates whether agents properly verify constraints before claiming task completion.

## Repository Overview

```
.
├── liveledger/           # LiveLedger: three-phase epistemic agent
│   ├── run.py            # Main agent (extract → search → update ledger)
│   ├── run_baseline.py   # ReAct baseline without ledger
│   ├── prompt.py         # System prompts
│   ├── tools.py          # Tool definitions (extract/search/update)
│   ├── utils.py          # EpistemicLedger, AgentStateMachine
│   ├── search_engine.py  # Serper (web search) + Jina (page reader)
│   └── train/            # SFT training pipeline
├── epistemic_ledger/     # Post-hoc evaluation framework
│   ├── build_ledger.py   # Build (candidate × constraint) ledger from any trajectory
│   ├── evaluate.py       # LLM-as-judge answer correctness
│   ├── accuracy.py       # Correct/Incorrect × Verified/Underverified classification
│   ├── failure_modes.py  # Failure mode taxonomy
│   └── prompts.py        # Constraint extraction + ledger update prompts
├── baselines/            # Baseline runners + sample results
│   ├── run_tag_search.py # Unified runner for tag-based baselines
│   └── results/          # Sample result files
├── datasets/             # Benchmark datasets
└── run_evaluation.py     # End-to-end evaluation pipeline
```

## Key Contributions

1. **Epistemic Ledger Framework**: Structured constraint verification tracking
2. **Failure Mode Taxonomy**: Bare Assertion, Overlooked Refutation, Stagnation, Premature Exit
3. **Live Ledger Agent**: Three-phase agent with real-time verification
4. **Evaluation Pipeline**: Automated analysis of existing agent trajectories

## Quick Start

### Installation

```bash
git clone https://github.com/dayoon-ko/illusory_completion.git
cd illusory_completion
pip install -r requirements.txt

# Set API keys
export SERPER_API_KEY="your-serper-key"
export JINA_API_KEY="your-jina-key"
```

### 1. Start vLLM Server

```bash
# Qwen3.5-27B (8 GPUs)
vllm serve Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 8 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching --enforce_eager
```

### 2. Run LiveLedger Agent

```bash
cd liveledger
python run.py \
  --model_name Qwen/Qwen3.5-27B \
  -o outputs \
  -w 4 \
  -d browsecomp frames deepsearchqa
```

### 3. Run Baseline (without ledger)

```bash
cd liveledger
python run_baseline.py --model_name Qwen/Qwen3.5-27B -o outputs_baseline -d browsecomp
```

### 4. Run Tag-Based Baselines

```bash
cd baselines
python run_tag_search.py -b search-r1 -d browsecomp frames --search_engine serper
```

Supported: `search-r1`, `smartsearch`, `rag-r1`, `reseek`, `hiprag`

### 5. Evaluate

```bash
# LiveLedger outputs (ledger already built inline)
python run_evaluation.py --output_dir liveledger/outputs --has_ledger

# Baseline outputs (build ledger post-hoc, then evaluate)
python run_evaluation.py --output_dir baselines/outputs/search-r1 \
  --baseline_name search-r1 --base_url http://localhost:8000/v1
```

### 6. SFT Training

```bash
cd liveledger/train
python curate_sft_data.py --input_dir ../outputs --output_dir sft_data
python train_sft.py --model_name Qwen/Qwen3.5-27B --dataset_path sft_data
python eval_checkpoint.py --checkpoint_dir output/checkpoint-xxx -d browsecomp
```

## Supported Datasets

```
browsecomp      # Browse & compose multi-hop questions
deepsearchqa    # Deep research questions requiring synthesis
frames          # Multi-constraint factual questions
livedrbench     # Real-time information retrieval
webwalkerqa     # Web navigation questions
bioasq          # Biomedical question answering
```

Dataset files: `datasets/{dataset_name}/test_mcqa.jsonl`

## Epistemic Ledger Concept

An epistemic ledger tracks verification status for each (candidate, constraint) pair:

```python
ledger = {
    "candidate": {
        "constraints": {
            "C1": {
                "obj": true,              # Objective: proven with evidence
                "obj_evidence": "quote",  # Supporting evidence
            }
        }
    }
}
```

**Verification Status:**
- `obj=true`: Proven with evidence
- `obj=false`: Disproven with evidence  
- `obj=null`: No evidence found

**Failure Modes:**
- `obj=null, per=true`: **Bare Assertion** (claim without evidence)
- `obj=false, per=true`: **Overlooked Refutation** (ignoring contradictory evidence)
- No progress 3+ turns: **Stagnation**
- Exit with unverified: **Premature Exit**

## Requirements

### System
- Python 3.8+
- CUDA-compatible GPU (80GB+ VRAM for large models)

### API Keys
- **Serper**: Web search ([serper.dev](https://serper.dev))
- **Jina Reader**: Web content ([jina.ai/reader](https://jina.ai/reader))

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

Apache 2.0

## Contact

- Paper: [https://arxiv.org/abs/2602.07549](https://arxiv.org/abs/2602.07549)
- Email: dayoon.ko@vision.snu.ac.kr
