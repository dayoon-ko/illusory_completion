# Is Enough Not Enough? Illusory Completion 🧠 in Search Agents 🔍

This repository contains implementations of the paper **"Is Enough Not Enough? Illusory Completion in Search Agents."**

We present a novel framework for analyzing failure modes in agentic search systems through **epistemic ledger tracking**, which systematically evaluates whether agents properly verify constraints before claiming task completion.

## Repository Overview

```
.
├── dataset/               # Benchmark datasets for agentic search evaluation
├── liveledger/           # Live epistemic ledger agent implementation
└── epistemic_ledger/     # Post-hoc epistemic ledger evaluation pipeline
```

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| **dataset/** | Constraint-based benchmark datasets | [datasets/README.md](dataset/README.md) |
| **liveledger/** | Real-time epistemic ledger agent | [liveledger/README.md](liveledger/README.md) |
| **epistemic_ledger/** | Post-hoc trajectory analysis | [epistemic_ledger/README.md](epistemic_ledger/README.md) |

## Key Contributions

1. **Epistemic Ledger Framework**: Structured constraint verification tracking
2. **Failure Mode Taxonomy**: Bare Assertion, Overlooked Refutation, Stagnation, Premature Exit
3. **Live Ledger Agent**: Three-phase agent with real-time verification
4. **Evaluation Pipeline**: Automated analysis of existing agent trajectories

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/illusory-completion.git
cd illusory-completion
pip install -r requirements.txt

# Set API keys
export SERPER_API_KEY="your-serper-key"
export JINA_API_KEY="your-jina-key"
export OPENAI_API_KEY="your-openai-key"
```

### Run Live Ledger Agent

```bash
# Start vLLM server
vllm serve openai/gpt-oss-120b --port 8000

# Run agent
cd liveledger
python run.py --dataset_names frames --num_workers 4
```

See [liveledger/README.md](liveledger/README.md) for detailed usage.

### Evaluate Existing Baseline

```bash
cd epistemic_ledger
python run.py -b your_baseline -d frames --max_workers 4
python annotate_exit.py -b your_baseline -d frames
python cal_acc.py -b your_baseline -d frames
python cal_taxonomy.py -b your_baseline -d frames -o all
```

See [epistemic_ledger/README.md](epistemic_ledger/README.md) for detailed usage.


## Supported Datasets (Multi-Constraint Problems)

```
browsecomp      # Browse & compose multi-hop questions
deepsearchqa    # Deep research questions requiring synthesis
frames          # Multi-constraint factual questions
livedrbench     # Real-time information retrieval
webwalkerqa     # Web navigation questions
```

Dataset files: `dataset/{dataset_name}/test_mcqa.jsonl`

## Requirements

### System
- Python 3.8+
- CUDA-compatible GPU (80GB+ VRAM for 120B models)

### API Keys
- **Serper**: Web search ([serper.dev](https://serper.dev))
- **Jina Reader**: Web content ([jina.ai/reader](https://jina.ai/reader))
- **OpenAI**: Answer evaluation ([platform.openai.com](https://platform.openai.com))

### Python Packages
```bash
pip install vllm openai httpx requests urllib3
```

## Epistemic Ledger Concept

An epistemic ledger tracks verification status for each (candidate, constraint) pair:

```python
ledger = {
    "candidate": {
        "constraints": {
            "C1": {
                "obj": true,              # Objective: proven with evidence
                "obj_evidence": "quote",  # Supporting evidence
                "per": true               # Perceived: agent's belief
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

## Citation (TBD)

```bibtex
@article{illusory2024,
  title={Is Enough Not Enough? Illusory Completion in Search Agents},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

Apache 2.0

## Contact

- Paper: [arXiv link] (TBD)
- Issues: [GitHub Issues](https://github.com/your-org/illusory-completion/issues)
- Email: your.email@institution.edu