#!/usr/bin/env python3
"""Compare LoRA checkpoints by generating sample text from a fixed prompt set.

Runs after training completion. Loads the base model once, then swaps LoRA
adapters from multiple checkpoints and generates outputs for a curated prompt
set covering three categories: known territory, generalization, and style.

The prompt set is loaded from `eval_prompts.yaml` (next to this script) so it
can be adapted to your subject. See `eval_prompts.example.yaml` for the shape.

Output: markdown report at <run_dir>/voice_compare_report_<timestamp>.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

DEFAULT_PROMPTS_PATH = Path(__file__).parent / "eval_prompts.yaml"
EXAMPLE_PROMPTS_PATH = Path(__file__).parent / "eval_prompts.example.yaml"

GEN_KWARGS = dict(
    max_new_tokens=350,
    do_sample=True,
    temperature=0.85,
    top_p=0.95,
    repetition_penalty=1.08,
)


def load_prompts(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        print(f"WARNING: {path} not found, falling back to "
              f"{EXAMPLE_PROMPTS_PATH.name}", file=sys.stderr)
        path = EXAMPLE_PROMPTS_PATH
    data = yaml.safe_load(path.read_text())
    return [(p["tag"], p["prompt"]) for p in data["prompts"]]


def find_checkpoints(run_dir: Path, last_n: int = 4) -> list[Path]:
    ckpts = sorted(
        [p for p in run_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    picked = ckpts[-last_n:] if len(ckpts) > last_n else ckpts
    final = run_dir / "final_adapter"
    if final.is_dir() and final not in picked:
        picked.append(final)
    return picked


def read_step(ckpt: Path) -> str:
    state = ckpt / "trainer_state.json"
    if state.exists():
        try:
            d = json.loads(state.read_text())
            step = d.get("global_step")
            epoch = d.get("epoch")
            h = d.get("log_history") or []
            last_loss = h[-1].get("loss") if h else None
            return f"step={step} epoch={epoch:.3f} loss={last_loss}"
        except Exception as e:
            return f"(trainer_state parse failed: {e})"
    return "(no trainer_state)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base model dir")
    ap.add_argument("--run_dir", required=True, help="path to run dir containing checkpoint-* and final_adapter")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH),
                    help="YAML file with eval prompts (see eval_prompts.example.yaml)")
    ap.add_argument("--last_n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260422)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = find_checkpoints(run_dir, args.last_n)
    if not ckpts:
        print(f"No checkpoints found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    prompts = load_prompts(Path(args.prompts))

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else run_dir / f"voice_compare_report_{ts}.md"

    print(f"[compare] base={args.base}")
    print(f"[compare] checkpoints: {[c.name for c in ckpts]}")
    print(f"[compare] prompts: {len(prompts)} from {args.prompts}")
    print(f"[compare] output: {out_path}")

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("[compare] loading base model (bf16)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.eval()

    model = None
    adapter_names: list[tuple[str, Path]] = []
    for i, ck in enumerate(ckpts):
        name = f"ck_{i}_{ck.name}"
        adapter_names.append((name, ck))
        print(f"[compare] loading adapter {name} <- {ck}")
        if model is None:
            model = PeftModel.from_pretrained(base, str(ck), adapter_name=name)
        else:
            model.load_adapter(str(ck), adapter_name=name)

    results: dict[str, dict[str, str]] = {}
    for name, ck in adapter_names:
        print(f"[compare] generating with {name}")
        model.set_adapter(name)
        model.eval()
        per_prompt = {}
        for tag, prompt in prompts:
            set_seed(args.seed)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    **GEN_KWARGS,
                )
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            per_prompt[tag] = text.strip()
        results[name] = per_prompt

    # Base-only baseline
    print("[compare] generating base-only baseline")
    with model.disable_adapter():
        base_only = {}
        for tag, prompt in prompts:
            set_seed(args.seed)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    **GEN_KWARGS,
                )
            base_only[tag] = tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

    # Write report
    lines = [
        "# Voice LoRA Checkpoint Comparison",
        "",
        f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Base model: `{args.base}`",
        f"- Run dir: `{run_dir}`",
        f"- Seed: {args.seed}",
        "- Generation: temp=0.85, top_p=0.95, max_new=350",
        "",
        "## Checkpoints",
        "",
        "| # | Name | Metrics |",
        "|---|---|---|",
        "| baseline | (no adapter) | — |",
    ]
    for i, (name, ck) in enumerate(adapter_names):
        lines.append(f"| {i+1} | `{ck.name}` | {read_step(ck)} |")

    for tag, prompt in prompts:
        lines += ["", f"## {tag}", "", "**Prompt:**", "", f"> {prompt.strip()}", ""]
        lines += ["### baseline (no LoRA)", "", base_only[tag], ""]
        for name, ck in adapter_names:
            lines += [f"### {ck.name}", "", results[name][tag], ""]

    out_path.write_text("\n".join(lines))
    print(f"[compare] wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
