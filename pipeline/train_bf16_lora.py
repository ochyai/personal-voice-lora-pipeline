#!/usr/bin/env python3
"""LoRA training for causal LMs on a single GPU — bf16 native or 4-bit QLoRA.

Designed for the NVIDIA GB10 (DGX Spark, 128GB unified) but works on any
GPU with enough VRAM for an 8B-class model in bf16 + LoRA + gradient
checkpointing (≈ 25GB for the configuration here). Native bf16 by default —
no quantization, no bitsandbytes.

If your GPU is smaller than ~24GB (e.g. a 16GB Colab T4, an RTX 3060/4060),
pass `--qlora`. The base model is then loaded in 4-bit (nf4) and the LoRA
adapter trains on top of it. Same adapter shape, same output, ~12-16GB VRAM,
~30% slower. Requires `pip install bitsandbytes`.

Continuous-friendly: when invoked with `--resume_from <adapter_dir>`, loads
the LoRA weights from that dir before starting a fresh training run on this
corpus. This is how the orchestrator chains versions: each step inherits
the previous step's adapter and trains for one more (epoch-on-corpus) pass.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--qlora", action="store_true",
                    help="Load the base model in 4-bit (nf4) for small GPUs (<24GB). "
                         "Requires bitsandbytes. ~30%% slower, same adapter output.")
    ap.add_argument("--resume_from", default=None,
                    help="Path to adapter to resume from")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # built-in scaled dot product attention
    )
    if args.qlora:
        print(f"Loading model (4-bit QLoRA): {args.model}")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print(f"Loading model (bf16): {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from adapter: {args.resume_from}")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        print("Creating fresh LoRA adapter")
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"Loading data: {args.data}")
    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    print(f"Records: {len(records):,}")

    ds = Dataset.from_list([{"text": r["text"]} for r in records])

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.seq_len, padding=False)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"],
                num_proc=4, desc="Tokenizing")
    total_tokens = sum(len(x) for x in ds["input_ids"])
    print(f"Total tokens: {total_tokens:,}")

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=collator)
    print("Starting training...")
    trainer.train()

    out = Path(args.output)
    trainer.model.save_pretrained(out / "final_adapter")
    tok.save_pretrained(out / "final_adapter")
    print(f"Saved adapter: {out / 'final_adapter'}")


if __name__ == "__main__":
    main()
