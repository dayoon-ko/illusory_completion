# Training the LiveLedger tracker

How [`dayoon/LiveLedger-4B`](https://huggingface.co/dayoon/LiveLedger-4B) was made:
gpt-oss-120b runs LiveLedger on training questions → its constraint extractions and ledger updates become SFT data → Qwen3.5-4B is fine-tuned on them.

None of the 484 evaluation questions is used here.

## 4 steps

```bash
# 1. rollouts with gpt-oss-120b (it searches AND writes the ledger)
vllm serve openai/gpt-oss-120b --port 8000 --tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser openai
export SERPER_API_KEY=... JINA_API_KEY=...
python training/1_collect_rollouts.py -o rollouts

# 2. (optional) keep only correct + verified rollouts; the released checkpoint skipped this step
python training/2_verify.py -i rollouts -o rollouts_verified

# 3. SFT data (1,393 extraction + 2,885 update examples for the released checkpoint)
python training/3_make_sft_data.py -i rollouts -o sft_data

# 4. full SFT of Qwen3.5-4B (16 GPUs; on one 8-GPU node add --grad_accum 16), then make it servable by vLLM
torchrun --nnodes 2 --nproc_per_node 8 ... training/4_train.py --data_dir sft_data --deepspeed training/ds_zero3_offload.json
python training/merge_for_vllm.py --checkpoint_dir sft_output/<run>/checkpoint-52 --output_dir LiveLedger-4B
```

## Files

| File | What it does |
|---|---|
| `1_collect_rollouts.py` | gpt-oss-120b LiveLedger rollouts (30 turns max) |
| `2_verify.py` | optional: Epistemic Ledger + correctness → `is_correct`, `verified` |
| `3_make_sft_data.py` | rollouts → ms-swift JSONL; single-candidate updates capped at 20 % |
| `4_train.py` | ms-swift full SFT (lr 3e-5 cosine, warmup 0.05, wd 0.1, max 8,192 tokens, 10 epochs, 90/10 split, seed 42) |
| `merge_for_vllm.py` | renames the vision-tower keys so vLLM can load the checkpoint |
| `rollout_utils.py` | ledger + tool state machine used by step 1 |
| `ds_zero3_offload.json` | DeepSpeed ZeRO-3 config used for step 4 |
| `data/` | training questions (`question`, `answer`, one JSON per line) |

The released weights are **epoch 2 (step 52 of 260)**: validation loss is lowest there and rises afterwards.

## Data

| Folder | Rows | Source |
|---|---:|---|
| `data/hds-qa/` | 1,003 | multi-constraint questions from [HDS-QA](https://huggingface.co/datasets/dayoon/HDS-QA-Questions) |
| `data/multiconir_books/` | 831 | Books domain of [MultiConIR](https://github.com/EIT-NLP/MultiConIR) (Apache-2.0), with a short answer extracted from the positive document |

The released checkpoint used the 1,002 HDS-QA and 392 MultiConIR-books rollouts that had finished when its SFT data was built (no correctness filter; step 3 above reproduces that data exactly from those rollouts).
