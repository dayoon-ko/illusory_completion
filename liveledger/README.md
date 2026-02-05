# Epistemic Agent - Three-Phase Approach

A systematic question-answering agent that uses constraint extraction, evidence search, and epistemic ledger tracking to provide verifiable answers.

## Overview

The agent operates in three sequential phases:

```
EXTRACT → SEARCH → UPDATE (loop) → ANSWER
```

| Phase | Purpose | Tools |
|-------|---------|-------|
| **EXTRACT** | Parse question into atomic constraints | `extract_constraints` |
| **SEARCH** | Gather evidence for constraints | `search`, `browse` |
| **UPDATE** | Update epistemic ledger with evidence | `update_ledger` |

The agent maintains an **epistemic ledger** that tracks verification status (true/false/null) and supporting evidence for each (candidate, constraint) pair.

## Requirements

```bash
pip install vllm openai httpx requests urllib3
```

**Environment Variables:**
- `SERPER_API_KEY`: Required for web search (Serper API)
- `JINA_API_KEY`: Required for web page reading (Jina Reader API)

## Directory Structure

```
.
├── run.py                # Main agent implementation
├── tools.py              # Tool definitions (EXTRACT/SEARCH/UPDATE)
├── utils.py              # State machine, epistemic ledger, utilities
├── search_engine.py      # Search and browse implementations
├── prompt.py             # System prompts (customize these)
├── README.md             # This file
└── outputs/              # Output directory (auto-created)
    └── {dataset_name}/
        └── {idx}.json    # Individual results
```

## Usage

### Basic Usage

```bash
export SERPER_API_KEY="your-serper-key"
export JINA_API_KEY="your-jina-key"

python run.py \
    --dataset_names frames \
    --output_dir outputs \
    --max_turns 100
```

### Full Configuration

```bash
python run.py \
    --model_name openai/gpt-oss-120b \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY \
    --serper_api_key "your-key" \
    --jina_api_key "your-key" \
    --reasoning_effort high \
    --max_turns 50 \
    --dataset_dir ../datasets \
    --dataset_names frames deepsearchqa \
    --start_idx 0 \
    --end_idx 10 \
    --output_dir outputs \
    --num_workers 4
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | vLLM model identifier | `openai/gpt-oss-120b` |
| `--base_url` | API endpoint URL | `http://localhost:8000/v1` |
| `--api_key` | API authentication key | `EMPTY` |
| `--serper_api_key` | Serper API key (or use env var) | From `SERPER_API_KEY` |
| `--jina_api_key` | Jina API key (or use env var) | From `JINA_API_KEY` |
| `--reasoning_effort` | Reasoning level: `low`, `medium`, `high` | `high` |
| `--max_turns` | Maximum search/update turns | `100` |
| `--dataset_dir` | Root directory containing datasets | `../datasets` |
| `-d, --dataset_names` | Datasets to process (space-separated) | `frames` |
| `-s, --start_idx` | Start index for dataset slice | `0` |
| `-e, --end_idx` | End index for dataset slice | `None` (all) |
| `-o, --output_dir` | Output directory for results | `outputs` |
| `--num_workers` | Number of parallel workers | `1` |

## Supported Datasets

```
browsecomp, deepsearchqa, frames, livedrbench, webwalkerqa
```

Dataset files must be located at: `{dataset_dir}/{dataset_name}/test_mcqa.jsonl`

## Output Format

Each item produces a JSON file: `{output_dir}/{dataset_name}/{idx}.json`

```json
{
  "question": "Which football player got 15+ assists in La Liga 2010-11 and played for Arsenal?",
  "answer": {
    "ground_truths": ["Mesut Özil"],
    "misc": {...}
  },
  "content": "Based on my search and verification...",
  "messages": [...],
  "prediction": "Mesut Özil",
  "turns": 5,
  "latency": 23.4,
  "ledger": {
    "Mesut Özil": {
      "constraints": {
        "C1": {
          "obj": true,
          "obj_evidence": "Özil provided 17 assists in La Liga 2010-11 season"
        },
        "C2": {
          "obj": true,
          "obj_evidence": "Özil joined Arsenal in 2013 and played until 2021"
        }
      }
    }
  },
  "status": "success"
}
```

## Three-Phase Workflow

### Phase 1: Extract Constraints

The agent parses the question into atomic constraints:

```
Question: "Which football player got 15+ assists in La Liga 2010-11 and played for Arsenal?"

Tool Call: extract_constraints(
  constraints=[
    "Got 15 or more assists in La Liga during 2010-2011 season",
    "Played for Arsenal at some point in their career"
  ]
)

Ledger initialized with C1, C2
```

### Phase 2: Search for Evidence

The agent searches for evidence and discovers candidates:

```
Turn 1:
  Tool Call: search(query=["La Liga 2010-11 season 15 assists"])
  Results: Found Mesut Özil with 17 assists, Lionel Messi with 18 assists...

Turn 2:
  Tool Call: browse(urls=["https://en.wikipedia.org/wiki/Mesut_Özil"])
  Results: Career details including Arsenal (2013-2021)...
```

### Phase 3: Update Ledger

After each search, the ledger is updated:

```
Update:
  entries=[
    {
      "candidate": "Mesut Özil",
      "constraint": "C1",
      "obj": true,
      "obj_evidence": "Özil provided 17 assists in La Liga 2010-11 season"
    },
    {
      "candidate": "Mesut Özil",
      "constraint": "C2",
      "obj": true,
      "obj_evidence": "Özil joined Arsenal in 2013 and played until 2021"
    }
  ]

Ledger Status: All constraints verified → COMPLETE
```

### State Transitions

```
INIT: Must call extract_constraints
  ↓
SEARCH: Can call search/browse
  ↓
COMPLETE: All constraints verified → Can provide final answer
```

## Parallel Processing

Process multiple items concurrently:

```bash
python run.py --num_workers 8 --dataset_names browsecomp deepsearchqa frames livedrbench webwalkerqa
```

Features:
- Thread-safe execution
- Automatic resume (skips existing output files)
- Progress tracking with lock
- Error files saved as `{idx}.json.error`

## Customization

### System Prompts

Edit `prompt.py` to customize agent behavior:

```python
SYSTEM_PROMPT_EXTRACT_CONSTRAINTS = """
Your custom constraint extraction instructions...
"""

SYSTEM_PROMPT_MAIN_W_LEDGER = """
Your custom main agent instructions...
"""

SYSTEM_PROMPT_UPDATE_LEDGER = """
Your custom ledger update instructions...
"""
```

### Tool Definitions

Modify `tools.py` to change tool schemas or add new tools.

### Search Implementation

Replace search engine by implementing the interface:

```python
class CustomSearchEngine:
    def search_batch(self, queries: List[str]) -> str:
        # Return formatted search results
        pass
```

Then update `run.py`:
```python
search_engine = CustomSearchEngine(api_key=args.custom_api_key)
```

## Full Pipeline Example

```bash
# Step 1: Start local LLM server (vLLM)
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve openai/gpt-oss-120b \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --max-num-seqs 16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enforce_eager \
    --port 8000

# Step 2: Set API keys
export SERPER_API_KEY="your-serper-key"
export JINA_API_KEY="your-jina-key"

# Step 3: Prepare dataset
# Place your dataset at: ../datasets/frames/test_mcqa.jsonl

# Step 4: Run agent
python run.py \
    --dataset_names frames \
    --dataset_dir ../datasets \
    --output_dir outputs \
    --num_workers 4 \
    --max_turns 50

# Step 5: Check results
# Output: outputs/frames/0.json, outputs/frames/1.json, ...
```

## Debugging

### Console Output

The agent prints colored logs:

| Color | Component |
|-------|-----------|
| 🔵 Blue | Ledger state and feedback |
| 🟢 Green | Model responses |
| 🟡 Yellow | Tool calls |
| 🟣 Purple | Tool results |
| 🔴 Red | Model thinking/reasoning |
| 🌐 Cyan | User input |
| 🎨 Magenta | Phase transitions |

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
```

### Common Issues

**Ledger not updating:**
- Check UPDATE phase logs for reasoning
- Verify search results contain relevant evidence
- Ensure constraints are clear and specific

**Stagnation (no progress):**
- Agent automatically prompts for new strategies after 5 turns
- Consider more specific search queries
- Try alternative candidate answers

**API errors:**
- Verify API keys are correct
- Check LLM server is running at specified `base_url`
- Review rate limits for Serper/Jina APIs

## Notes

- Phase 1 (EXTRACT) requires one LLM call per question
- Phase 2 (SEARCH) and Phase 3 (UPDATE) loop until all constraints verified or max_turns reached
- Search tools support batch queries for efficiency
- Ledger tracks all candidates, not just the final answer
- Resume capability: existing output files are automatically skipped
- Error files (`.error` suffix) are saved for failed items

## Architecture

```
EpistemicAgentThreePhase
├── _call_extract_constraints_phase()    # Phase 1: Extract constraints
├── _call_update_phase()                 # Phase 3: Update ledger
├── _get_search_response()               # Phase 2: Get model response
├── _execute_tool_calls()                # Execute search/browse
└── run()                                # Main loop

Supporting Classes:
├── AgentStateMachine                    # Enforce phase transitions
├── EpistemicLedger                      # Track evidence and completion
├── SerperSearchEngine                   # Google search via Serper
└── JinaBrowser                          # Web page content via Jina
```