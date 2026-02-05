# Is Enough Not Enough?<br>Illusory Completion 🧠 in Search Agents 🔍

This repository contains implementations of the paper **"Is Enough Not Enough? Illusory Completion in Search Agents."**

We present a novel framework for analyzing failure modes in agentic search systems through **epistemic ledger tracking**, which systematically evaluates whether agents properly verify constraints before claiming task completion.

## Repository Overview

```
.
├── dataset/               # Benchmark datasets for agentic search evaluation
├── liveledger/            # Live epistemic ledger agent implementation
└── epistemic_ledger/      # Post-hoc epistemic ledger evaluation pipeline
```

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **dataset/** | Benchmark datasets with constraint-based questions | - | JSONL files with Q&A pairs |
| **liveledger/** | Agent with real-time epistemic ledger tracking | Questions | Agent trajectories with verified answers |
| **epistemic_ledger/** | Retrospective analysis of agent trajectories | Agent outputs | Failure mode taxonomy & metrics |

## Key Contributions

1. **Epistemic Ledger Framework**: A structured approach to tracking constraint verification in search agents
2. **Failure Mode Taxonomy**: Systematic categorization of agent failures (Bare Assertion, Overlooked Refutation, Stagnation, Premature Exit)
3. **Live Ledger Agent**: A three-phase agent that maintains real-time verification status

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/illusory-completion.git
cd illusory-completion

# Install dependencies
pip install -r requirements.txt

# Set API keys
export SERPER_API_KEY="your-serper-key"
export JINA_API_KEY="your-jina-key"
export OPENAI_API_KEY="your-openai-key"
```

### Option 1: Run Live Ledger Agent

```bash
# Start vLLM server
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve openai/gpt-oss-120b \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --max-num-seqs 16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# Run agent on dataset
cd liveledger
python run.py \
    --dataset_names frames browsecomp \
    --output_dir outputs \
    --num_workers 4 \
    --max_turns 50

# Results: liveledger/outputs/{dataset_name}/{idx}.json
```

### Option 2: Evaluate Existing Baseline

```bash
# Prepare baseline outputs in epistemic_ledger/data/
# Format: epistemic_ledger/data/{baseline_name}/{dataset_name}.jsonl

cd epistemic_ledger

# Step 1: Generate epistemic ledgers
python run.py -b your_baseline -d frames --max_workers 4

# Step 2: Annotate answer correctness
python annotate_exit.py -b your_baseline -d frames

# Step 3: Calculate metrics
python cal_acc.py -b your_baseline -d frames

# Step 4: Analyze failure modes
python cal_taxonomy.py -b your_baseline -d frames -o all
```

## Repository Structure

### 1. Datasets (`dataset/`)

Benchmark datasets for evaluating agentic search systems with constraint-based questions.

**Supported Datasets:**
```
browsecomp      # Browse & compose multi-hop questions
deepsearchqa    # Deep research questions requiring synthesis
frames          # Multi-constraint factual questions
livedrbench     # Real-time information retrieval
webwalkerqa     # Web navigation questions
```

**Dataset Format:**
```json
{
  "question": "Which football player got 15+ assists in La Liga 2010-11 and played for Arsenal?",
  "answer": "ground truth answer(s)"
}
```

**Location:**
```
dataset/
├── browsecomp/
│   └── test_mcqa.jsonl
├── frames/
│   └── test_mcqa.jsonl
├── deepsearchqa/
│   └── test_mcqa.jsonl
├── livedrbench/
│   └── test_mcqa.jsonl
└── webwalkerqa/
    └── test_mcqa.jsonl
```

### 2. Live Ledger Agent (`liveledger/`)

A three-phase agent that maintains real-time epistemic ledger tracking during question answering.

**Architecture:**
```
EXTRACT Phase → SEARCH Phase → UPDATE Phase (loop) → ANSWER
     ↓               ↓                ↓
 Constraints    Evidence      Ledger Updates
```

**Key Features:**
- Real-time constraint verification
- Structured evidence tracking
- State machine enforcement
- Automatic stagnation detection

**Files:**
```
liveledger/
├── run.py                # Main agent entry point
├── tools.py              # Tool definitions (extract/search/update)
├── utils.py              # State machine & epistemic ledger
├── search_engine.py      # Search/browse implementations
├── prompt.py             # Customizable system prompts
└── README.md             # Detailed documentation
```

**Usage:**
```bash
cd liveledger
python run.py --dataset_names frames --num_workers 4
```

See [`liveledger/README.md`](liveledger/README.md) for detailed documentation.

### 3. Epistemic Ledger Evaluation (`epistemic_ledger/`)

Post-hoc analysis pipeline for evaluating existing agent trajectories.

**Pipeline:**
```
Baseline → Ledger Generation → Answer Annotation → Metrics → Taxonomy
Results     (run.py)            (annotate_exit.py)  (cal_acc.py) (cal_taxonomy.py)
```

**Key Metrics:**
- **Accuracy (Acc)**: Correct answers / Total questions
- **Underverification Rate (UAR)**: Correct but unverified answers
- **Verification Status**: Correct & Verified, Incorrect & Verified, etc.

**Failure Modes:**
- **Bare Assertion**: Claims without evidence (obj=None, per=True)
- **Overlooked Refutation**: Ignoring contradictory evidence (obj=False)
- **Stagnation**: No progress for 3+ consecutive turns
- **Premature Exit**: Stopping with unverified constraints

**Files:**
```
epistemic_ledger/
├── run.py                # Step 1: Generate ledgers
├── annotate_exit.py      # Step 2: Annotate correctness
├── cal_acc.py            # Step 3: Calculate metrics
├── cal_taxonomy.py       # Step 4: Failure taxonomy
├── prompts.py            # Prompt templates
├── data/                 # Input baseline results
│   └── {baseline}/{dataset}.jsonl
└── output/               # Generated outputs
    ├── epistemic_ledger/
    └── epistemic_ledger_finished/
```

**Usage:**
```bash
cd epistemic_ledger
python run.py -b hds -d frames --max_workers 4
python annotate_exit.py -b hds -d frames
python cal_acc.py -b hds -d frames
python cal_taxonomy.py -b hds -d frames -o all
```

See [`epistemic_ledger/README.md`](epistemic_ledger/README.md) for detailed documentation.

## Complete Workflow Example

### Scenario: Evaluate a New Search Agent

```bash
# 1. Prepare your baseline agent output
# Format: epistemic_ledger/data/my_agent/frames.jsonl
# Each line: {"output": {...}, "question": "...", "answer": "..."}

# 2. Generate epistemic ledgers
cd epistemic_ledger
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve openai/gpt-oss-120b \
    --max-num-seqs 16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --port 8000 &

python run.py -b my_agent -d frames browsecomp --max_workers 8

# 3. Annotate answer correctness
export OPENAI_API_KEY="sk-..."
python annotate_exit.py -b my_agent -d frames browsecomp

# 4. Calculate accuracy and underverification
python cal_acc.py -b my_agent -d frames browsecomp

# 5. Analyze failure modes
python cal_taxonomy.py -b my_agent -d frames browsecomp -o all

# 6. Compare with live ledger agent
cd ../liveledger
python run.py \
    --dataset_names frames browsecomp \
    --output_dir outputs/my_comparison \
    --num_workers 4

# Results:
# - epistemic_ledger/output/epistemic_ledger_finished/my_agent/
# - liveledger/outputs/my_comparison/
```

## Requirements

### System Requirements

- Python 3.8+
- CUDA-compatible GPU (for vLLM)
- 40+ GPU memory recommended (for 120B models)

### Python Dependencies

**For epistemic_ledger pipeline:**
```bash
cd epistemic_ledger
pip install -r requirements.txt
```

**For liveledger agent:**
```bash
cd liveledger
pip install openai httpx requests urllib3
```

### API Keys

| Service | Purpose | Environment Variable |
|---------|---------|---------------------|
| Serper | Web search | `SERPER_API_KEY` |
| Jina Reader | Web page content | `JINA_API_KEY` |
| OpenAI | Answer evaluation (epistemic_ledger only) | `OPENAI_API_KEY` |

Get API keys:
- Serper: https://serper.dev
- Jina Reader: https://jina.ai/reader
- OpenAI: https://platform.openai.com

## Key Concepts

### Epistemic Ledger

A structured tracking system for constraint verification:

```python
ledger = {
    "candidate_answer": {
        "constraints": {
            "C1": {
                "obj": true,              # Objective verification status
                "obj_evidence": "quote",  # Supporting evidence
                "per": true               # Perceived status (if different)
            },
            "C2": {
                "obj": false,
                "obj_evidence": "contradictory quote",
                "per": true               # Overlooked refutation!
            }
        }
    }
}
```

### Constraint Types

- **Atomic**: Single, independently verifiable condition
- **Verifiable**: Can be checked with evidence
- **Independent**: Not dependent on other constraints

Example:
```
Question: "Which player got 15+ assists in La Liga 2010-11 and played for Arsenal?"

Constraints:
- C1: "Got 15 or more assists in La Liga during 2010-2011 season"
- C2: "Played for Arsenal at some point in their career"
```

### Verification Status

| Status | obj | obj_evidence | Interpretation |
|--------|-----|--------------|----------------|
| Verified | true | "quote" | Proven with evidence |
| Refuted | false | "quote" | Disproven with evidence |
| Unverified | null | null | No evidence found |

### Failure Modes

1. **Bare Assertion**
   - Agent claims constraint is verified (per=true)
   - No objective evidence (obj=null)
   - Example: "The player played for Arsenal" without citation

2. **Overlooked Refutation**
   - Evidence contradicts constraint (obj=false)
   - Agent still treats as verified (per=true)
   - Example: Ignoring "Player never played in La Liga"

3. **Stagnation**
   - No verification progress for 3+ consecutive turns
   - Repeated similar searches
   - Example: Multiple searches for same information

4. **Premature Exit**
   - Agent stops before all constraints verified
   - Partial solution accepted as complete
   - Example: Verifying C1 but not checking C2

## Supported Baselines

The `epistemic_ledger` pipeline supports both known and custom baselines:

### Known Baselines
```
search-r1, rag-r1, hds, hds-grpo, asearcher, webexplorer,
tongyidr, dr-tulu, react, react_liveledger, search_o1
```

### Custom Baselines

Any baseline with **TAO format** (Thinking-Action-Observation):

```json
{
  "output": {
    "thinking_blocks": ["thought 1", "thought 2"],
    "query_blocks": ["query 1", "query 2"],
    "results_blocks": ["result 1", "result 2"]
  },
  "question": "...",
  "answer": "..."
}
```

Or **TAOT format** (separated prev/next thinking):

```json
{
  "output": {
    "prev_thinking_blocks": ["prev thought 1"],
    "query_blocks": ["query 1"],
    "results_blocks": ["result 1"],
    "next_thinking_blocks": ["next thought 1"]
  }
}
```
<!-- 
## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@article{ko2024,
  title={Is Enough Not Enough? Illusory Completion in Search Agents},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
``` -->
<!-- 
## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

See individual component READMEs for specific contribution guidelines.

## License

[Your license here - e.g., MIT, Apache 2.0]

## Contact

- Paper: [arXiv link]
- Issues: [GitHub Issues](https://github.com/your-org/illusory-completion/issues)
- Email: your.email@institution.edu -->

## Acknowledgments
- This work was supported by LG AI Research.
- Serper API for web search capabilities
- Jina AI for web content extraction
- vLLM for efficient LLM serving
- OpenAI for evaluation models
