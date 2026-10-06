"""
Make an SFT checkpoint loadable by vLLM.

The SFT checkpoint has visual keys under 'model.language_model.visual.*'
but vLLM expects 'model.visual.*' (matching the original HF model layout).

This script:
1. Takes text (language_model) weights from the SFT checkpoint
2. Takes visual weights from the original model
3. Saves a merged model directory that vLLM can load directly
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to SFT checkpoint")
    parser.add_argument("--base_model_dir", default=None,
                        help="Local copy of the base model (default: download the config files of --base_model)")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the merged model")
    args = parser.parse_args()

    # Base model: only its config/tokenizer files are used, for any the checkpoint lacks
    if args.base_model_dir is None:
        from huggingface_hub import snapshot_download
        args.base_model_dir = snapshot_download(args.base_model, allow_patterns=["*.json", "*.jinja"])

    print(f"Checkpoint:  {args.checkpoint_dir}")
    print(f"Base model:  {args.base_model_dir}")
    print(f"Output:      {args.output_dir}")

    # Load checkpoint weights
    print("\nLoading checkpoint weights...")
    ckpt_weights = {}
    for fname in sorted(os.listdir(args.checkpoint_dir)):
        if fname.endswith(".safetensors"):
            w = load_file(os.path.join(args.checkpoint_dir, fname))
            ckpt_weights.update(w)
    print(f"  Loaded {len(ckpt_weights)} keys from checkpoint")

    # Rename keys:
    #   'model.language_model.visual.*' -> 'model.visual.*'  (fix visual prefix)
    #   'model.language_model.*'        -> keep as is         (text weights)
    merged = {}
    renamed_count = 0
    for k, v in ckpt_weights.items():
        if "language_model.visual." in k:
            new_k = k.replace("model.language_model.visual.", "model.visual.")
            merged[new_k] = v
            renamed_count += 1
        else:
            merged[k] = v
    print(f"  Renamed {renamed_count} visual keys")
    print(f"  Merged total: {len(merged)} keys")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Split into 2 shards (text / visual+remaining)
    text_keys = {k: v for k, v in merged.items() if "visual" not in k}
    visual_keys = {k: v for k, v in merged.items() if "visual" in k}

    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"

    print(f"\nSaving shard 1 ({len(text_keys)} text keys)...")
    save_file(text_keys, os.path.join(args.output_dir, shard1_name))
    print(f"Saving shard 2 ({len(visual_keys)} visual keys)...")
    save_file(visual_keys, os.path.join(args.output_dir, shard2_name))

    # Build index
    weight_map = {}
    for k in text_keys:
        weight_map[k] = shard1_name
    for k in visual_keys:
        weight_map[k] = shard2_name

    total_size = sum(v.numel() * v.element_size() for v in merged.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Copy config files from checkpoint (they have the correct architecture)
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "preprocessor_config.json",
                  "processor_config.json", "generation_config.json"]:
        src = os.path.join(args.checkpoint_dir, fname)
        if not os.path.exists(src):
            src = os.path.join(args.base_model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))

    print(f"\nDone! Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
