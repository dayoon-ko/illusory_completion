# Epistemic Ledger Evaluation Pipeline

This pipeline evaluates model trajectories using epistemic ledger tracking to identify failure modes in agentic search systems.

## Overview

The pipeline consists of 4 sequential scripts:

```
run.py → annotate_exit.py → cal_acc.py → cal_taxonomy.py
```

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `run.py` | Generate constraint checklists and build epistemic ledgers | Baseline results (JSONL) | `epistemic_ledger/` |
| `annotate_exit.py` | Evaluate answer correctness using LLM judge | `epistemic_ledger/` | `epistemic_ledger_finished/` |
| `cal_acc.py` | Calculate accuracy and verification metrics | `epistemic_ledger_finished/` | Console output (metrics) |
| `cal_taxonomy.py` | Calculate failure mode taxonomy | `epistemic_ledger_finished/` | Console output (taxonomy) |

## Requirements

```bash
pip install -r requirements.txt
```

**Environment Variables:**
- `OPENAI_API_KEY`: Required for `annotate_exit.py` (uses OpenAI API for answer evaluation)

## Directory Structure

```
epistemic_ledger/
├── run.py                   # Step 1: Ledger generation
├── annotate_exit.py         # Step 2: Answer correctness annotation
├── cal_acc.py               # Step 3: Accuracy calculation
├── cal_taxonomy.py          # Step 4: Failure mode taxonomy
├── prompts.py               # Prompt templates for LLM calls
├── README.md                # This file
├── data/                    # Input baseline results
│   └── {baseline_name}/{dataset_name}.jsonl
└── output/                  # All outputs (auto-created)
    ├── epistemic_ledger/           # Output from Step 1
    │   └── {baseline_name}/{dataset_name}/item_*.json
    └── epistemic_ledger_finished/  # Output from Step 2
        └── {baseline_name}/{dataset_name}/item_*.json
```

## Usage

### Step 1: Generate Epistemic Ledgers

```bash
python run.py \
    --baseline_name search-r1 hds react \
    --dataset_name browsecomp deepsearchqa frames livedrbench webwalkerqa \
    --input_dir data \
    --output_dir output/epistemic_ledger \
    --model_name openai/gpt-oss-120b \
    --max_workers 8
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `-b, --baseline_name` | Baseline methods to evaluate | All baselines |
| `-d, --dataset_name` | Datasets to process | All datasets |
| `--input_dir` | Directory containing baseline results | `data` |
| `-o, --output_dir` | Output directory for ledgers | `output/epistemic_ledger` |
| `--model_name` | vLLM model for ledger generation | `openai/gpt-oss-120b` |
| `--max_turns` | Maximum trajectory turns to process | None (all) |
| `--max_workers` | Number of parallel workers | 8 |
| `--is_save_blocks` | Save extracted blocks to output | False |
| `--num_try` | Experiment trial number (for multiple runs) | 0 |

### Step 2: Annotate Answer Correctness

```bash
export OPENAI_API_KEY="your-api-key"

python annotate_exit.py \
    --baseline_name search-r1 hds react \
    --dataset_name browsecomp frames livedrbench \
    --ledger_dir output/epistemic_ledger \
    --output_dir output/epistemic_ledger_finished
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `-b, --baseline_name` | Baseline methods to evaluate | All baselines |
| `-d, --dataset_name` | Datasets to process | All datasets |
| `--ledger_dir` | Input directory (Step 1 output) | `output/epistemic_ledger` |
| `--output_dir` | Output directory | `output/epistemic_ledger_finished` |

### Step 3: Calculate Accuracy Metrics

```bash
python cal_acc.py \
    --baseline_name search-r1 hds react \
    --dataset_name browsecomp frames livedrbench \
    --ledger_dir output/epistemic_ledger_finished
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `-b, --baseline_name` | Baseline methods to evaluate | All baselines |
| `-d, --dataset_name` | Datasets to process | All datasets |
| `--ledger_dir` | Input directory (Step 2 output) | `output/epistemic_ledger_finished` |

**Output Metrics:**
- `Acc`: Accuracy (Correct / Total)
- `UAR`: Underverification Rate
- `C - V`: Correct & Verified
- `C - UV`: Correct & Underverified
- `IC - V`: Incorrect & Verified
- `IC - UV`: Incorrect & Underverified

### Step 4: Calculate Failure Mode Taxonomy

```bash
python cal_taxonomy.py \
    --baseline_name search-r1 hds react \
    --dataset_name browsecomp frames livedrbench \
    --ledger_dir output/epistemic_ledger_finished \
    --output_mode all
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `-b, --baseline_name` | Baseline methods to evaluate | All baselines |
| `-d, --dataset_name` | Datasets to process | All datasets |
| `--ledger_dir` | Input directory (Step 2 output) | `output/epistemic_ledger_finished` |
| `-o, --output_mode` | Output format: `count`, `global_ratio`, `local_ratio`, `all` | `all` |

**Failure Modes:**
- `Bare Assertion`: Claims without objective evidence (obj=None, per=True)
- `Overlooked Refutation`: Ignoring contradictory evidence (obj=False, still active)
- `Stagnation`: No progress for 3+ turns
- `Premature Exit`: Stopping with unverified constraints
- `None`: Successfully verified (no failure)

## Supported Baselines

### Known Baselines (with specific handling)
```
search-r1, rag-r1, hds, hds-grpo, asearcher, webexplorer,
tongyidr, tongyidr-liveledger-20b, dr-tulu, react, react_s1,
react_liveledger, react_liveledger_20b, react_liveledger_20b_baseline,
search_o1_gpt-oss-20b, search_o1_gpt-oss-120b
```

### Custom Baselines (auto-detected)

Any baseline with **standard TAO format** is automatically supported. Your baseline output should have:

```json
{
  "output": {
    "thinking_blocks": ["thinking 1", "thinking 2", ...],
    "query_blocks": ["query 1", "query 2", ...],
    "results_blocks": ["result 1", "result 2", ...]
  },
  "question": "...",
  "answer": "..."
}
```

Or **TAOT format** (pre-separated prev/next thinking):

```json
{
  "output": {
    "prev_thinking_blocks": ["prev 1", "prev 2", ...],
    "query_blocks": ["query 1", "query 2", ...],
    "results_blocks": ["result 1", "result 2", ...],
    "next_thinking_blocks": ["next 1", "next 2", ...]
  }
}
```

Usage with custom baseline:
```bash
python run.py -b my_custom_baseline -d browsecomp
```

## Supported Datasets

```
browsecomp, deepsearchqa, frames, livedrbench, webwalkerqa
```

## Full Pipeline Example

```bash
# Prepare input data
# Place your baseline results in: data/{baseline_name}/{dataset_name}.jsonl

# Step 1: Generate ledgers (requires local LLM server at localhost:8000)
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup vllm serve openai/gpt-oss-120b --max-num-seqs 16 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --enforce_eager &
python run.py -b hds -d browsecomp --max_workers 4

# Step 2: Annotate correctness (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python annotate_exit.py -b hds -d browsecomp

# Step 3: Calculate accuracy
python cal_acc.py -b hds -d browsecomp

# Step 4: Calculate failure taxonomy
python cal_taxonomy.py -b hds -d browsecomp -o all

# Output will be in:
# - output/epistemic_ledger/hds/browsecomp/item_*.json
# - output/epistemic_ledger_finished/hds/browsecomp/item_*.json
```

## Notes

- Step 1 requires a running LLM server at `http://localhost:8000/v1` by default
- Step 2 uses OpenAI API (GPT-5) for answer evaluation
- Steps 3 and 4 are offline analysis scripts (no API calls)
- Each step can be run independently if the required input files exist
