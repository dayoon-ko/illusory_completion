"""
SFT training using Swift Python API with Seq2SeqTrainer.

Usage:
    # Single node, all GPUs
    torchrun --nproc_per_node=8 4_train_sft.py

    # With deepspeed
    torchrun --nproc_per_node=8 4_train_sft.py --deepspeed ds_zero3_offload.json

    # Custom settings
    torchrun --nproc_per_node=8 4_train_sft.py --model Qwen/Qwen3.5-4B --lr 1e-5 --epochs 10
"""

import argparse
import json
import os
import random
import sys

from swift.model import get_model_processor
from swift.template import get_template
from swift.dataset import EncodePreprocessor
from swift.utils import get_logger, seed_everything, get_model_parameter_info
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import datasets


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================
# Data preprocessing
# =====================

def _render_tool_calls_qwen(tool_calls):
    """Render tool_calls list into Qwen3.5 <tool_call> text format."""
    parts = []
    for tc in tool_calls:
        func = tc["function"] if "function" in tc else tc
        name = func["name"]
        args = func.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        lines = [f"<tool_call>\n<function={name}>"]
        for k, v in args.items():
            v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            lines.append(f"<parameter={k}>\n{v_str}\n</parameter>")
        lines.append("</function>\n</tool_call>")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def load_and_preprocess(data_files, max_length, tokenizer, seed=42):
    """Load JSONL files, inline tools/tool_calls, filter by length, shuffle, return HF Dataset."""
    all_records = []
    skipped = 0

    for fname in data_files:
        with open(fname) as f:
            for line in f:
                d = json.loads(line)

                # Move tools into system message
                if "tools" in d:
                    tools_str = json.dumps(d.pop("tools"), ensure_ascii=False)
                    if d["messages"] and d["messages"][0]["role"] == "system":
                        d["messages"][0]["content"] += f"\n\nAvailable tools:\n{tools_str}"
                    else:
                        d["messages"].insert(0, {"role": "system", "content": f"Available tools:\n{tools_str}"})

                # Inline tool_calls into assistant content
                for m in d["messages"]:
                    if m.get("tool_calls"):
                        tc_text = _render_tool_calls_qwen(m["tool_calls"])
                        content = m.get("content") or ""
                        m["content"] = (content + "\n\n" + tc_text) if content.strip() else tc_text
                        del m["tool_calls"]

                # Remove metadata
                d.pop("metadata", None)

                # Filter by token length
                text = " ".join(m.get("content") or "" for m in d["messages"])
                n_tokens = len(tokenizer.encode(text))
                if n_tokens <= max_length:
                    all_records.append(d)
                else:
                    skipped += 1

    if skipped:
        print(f"  Filtered out {skipped} samples exceeding {max_length} tokens", file=sys.stderr)

    random.seed(seed)
    random.shuffle(all_records)
    print(f"  Total samples after filtering: {len(all_records)}", file=sys.stderr)

    # Keep messages format as-is (Swift 4.x expects 'messages' key)
    processed = []
    for d in all_records:
        processed.append({"messages": d["messages"]})

    ds = datasets.Dataset.from_list(processed)
    split = ds.train_test_split(test_size=0.1, seed=seed)
    return split["train"], split["test"]


def main():
    parser = argparse.ArgumentParser(description="SFT training with Swift Python API")
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--task", "-t", type=str, default="both",
                        choices=["extract", "update", "both"])
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output_dir", "-o", type=str, default="sft_output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    logger = get_logger()
    seed_everything(args.seed)

    # =====================
    # Collect data files
    # =====================
    data_files = []
    if args.task in ("update", "both"):
        p = os.path.join(args.data_dir, "sft_ledger_update.jsonl")
        if os.path.exists(p):
            data_files.append(os.path.abspath(p))
    if args.task in ("extract", "both"):
        p = os.path.join(args.data_dir, "sft_constraint_split.jsonl")
        if os.path.exists(p):
            data_files.append(os.path.abspath(p))

    if not data_files:
        print(f"ERROR: No training data found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    for f in data_files:
        n = sum(1 for _ in open(f))
        print(f"  {f}: {n} examples", file=sys.stderr)

    # =====================
    # Model & Template
    # =====================
    logger.info(f"Loading model: {args.model}")
    import torch
    model, processor = get_model_processor(args.model, torch_dtype=torch.bfloat16)
    tokenizer = processor
    template = get_template(
        tokenizer,
        default_system=None,
        max_length=args.max_length,
        template_type=model.model_meta.template,
        truncation_strategy='raise',
    )
    template.set_mode('train')

    # Pre-set dummy_model to prevent lazy init inside DeepSpeed ZeRO-3 context
    template.dummy_model = model

    model.enable_input_require_grads()

    model_parameter_info = get_model_parameter_info(model)
    logger.info(f"model_parameter_info: {model_parameter_info}")

    # =====================
    # Dataset
    # =====================
    logger.info("Loading and preprocessing data...")
    train_dataset, val_dataset = load_and_preprocess(
        data_files, args.max_length, tokenizer, seed=args.seed
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} samples")

    # Tokenize with Swift template
    encoder = EncodePreprocessor(template=template)
    train_dataset = encoder(train_dataset, num_proc=1)
    val_dataset = encoder(val_dataset, num_proc=1)
    n_samples = len(train_dataset) if hasattr(train_dataset, '__len__') else sum(1 for _ in train_dataset)
    logger.info(f"Tokenized - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # =====================
    # Output dir
    # =====================
    model_short = args.model.split("/")[-1]
    run_name = f"{model_short}_lr{args.lr}_ep{args.epochs}_ml{args.max_length}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # =====================
    # Training
    # =====================
    import math
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = math.ceil(n_samples / (args.batch_size * args.grad_accum * num_gpus))
    max_steps = steps_per_epoch * args.epochs

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        report_to=["tensorboard"],
        logging_first_step=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        metric_for_best_model="loss",
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        dataloader_num_workers=0,
        data_seed=args.seed,
        bf16=True,
        deepspeed=args.deepspeed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )

    logger.info(f"Starting training: {output_dir}")
    trainer.train()

    last_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f"Last checkpoint: {last_checkpoint}")

    # Save tokenizer to last checkpoint
    if last_checkpoint:
        tokenizer.save_pretrained(last_checkpoint)
        logger.info(f"Tokenizer saved to {last_checkpoint}")


if __name__ == "__main__":
    main()
