# personal-voice-lora-pipeline

> A pipeline for training a LoRA adapter that **writes like a specific person**
> from their accumulated text. Built and tuned on an NVIDIA GB10 (DGX Spark)
> for a ~100M-token Japanese-essayist corpus.

This repo packages the practical scaffolding: orchestrator, corpus builder,
training script, evaluation harness, and systemd unit. The strategy it
encodes — **corpus growth, then penetration** — is what made the difference
between "model knows the topics" and "model writes in this voice."

---

## What this is for

You have ~50-200M tokens of text written by one person (books, articles,
emails, chat logs, talk transcripts). You want a model that, given a prompt
the person has never seen, produces something that reads like them — same
sentence rhythm, same idiosyncratic turns, same way of opening an essay. Not
a chatbot; a continuation engine in their voice.

This is **continued pre-training as a LoRA adapter** on top of a strong base
model (Llama-3.1-Swallow-8B for Japanese, swap for your language). LoRA, not
full finetune, because the base's general knowledge stays intact while the
adapter encodes the voice. r=64, alpha=128 across all linear modules gives a
~670MB adapter — small enough to ship, large enough to carry the style.

It is **not** instruction tuning, **not** RLHF, **not** dialog SFT. Those are
later steps if you want a chatbot. The objective here is one thing: pure
continuation in voice.

---

## Why "corpus growth, then penetration"

The trap on a first attempt is to train a single epoch on a fixed corpus and
call it done. The model will reliably reproduce the subject's *vocabulary*
(named concepts, signature terms) but write them inside a textbook frame
("This concept refers to…"). It hasn't internalized the voice; it's
*translating into* the voice from a generic frame.

What works better is two phases:

1. **Growth phase** (typically 2-3 versions). Start with a curated seed
   corpus. Each next version adds another bundle of source material — emails,
   chat logs, talk transcripts, manuscripts. Each version is one epoch.
   Adapter inherits from the previous version. The model picks up vocabulary
   and topic priors fast (loss drops from ~1.2 to ~0.95 in the first epoch).

2. **Penetration phase** (typically 10-12 versions). Freeze the corpus at
   the largest size. Iterate epochs at decaying learning rate (5e-5 → 3e-6
   over ~12 epochs). This is where style locks in. Around epoch 4-6 you'll
   see the model stop opening with "Let me explain…" and start opening with
   the subject's actual signature moves — sentence fragments, sensory
   openings, self-questioning. By epoch 10+ the off-domain prompts also
   carry the voice.

The total schedule is encoded in [`pipeline/versions.example.yaml`](pipeline/versions.example.yaml).
On a GB10 with 8B base, ~100M tokens, seq_len 4096, ~25s/step, ~3,300
steps/epoch, ~23h/epoch → **~2 weeks of continuous training** for 15
versions.

---

## What's in the box

```
personal-voice-lora-pipeline/
├── pipeline/
│   ├── orchestrate.py            # The state machine. Reads versions.yaml,
│   │                               runs versions sequentially, resumes after
│   │                               failure. systemd-managed.
│   ├── build_corpus.py           # Template. Source-loader functions for
│   │                               common shapes (mbox-JSONL, Slack export,
│   │                               docx, text-glob). Edit BUILD_PLAN.
│   ├── train_bf16_lora.py        # bf16 LoRA training. No quantization
│   │                               (GB10 has the VRAM). 4-bit QLoRA is a
│   │                               trivial swap if you're VRAM-bound.
│   ├── compare_checkpoints.py    # Eval harness. Generates the same prompts
│   │                               from every checkpoint, writes a markdown
│   │                               diff report.
│   ├── eval_prompts.example.yaml # The 6-prompt diagnostic set (known /
│   │                               generalize / style).
│   └── versions.example.yaml     # 15-version growth + penetration schedule.
├── systemd/
│   └── voice-lora.service        # User-mode unit. linger + Restart=on-failure.
├── examples/
│   └── voice_compare_report_example.md   # Real ~100M-token Japanese-corpus
│                                            output, ~58KB. Read top-to-bottom
│                                            to see voice transfer in action.
├── docs/
│   ├── setup.md                  # Step-by-step install on a fresh GPU host
│   ├── lessons.md                # What worked, what failed, gotchas
│   └── corpus-strategy.md        # Why corpus growth → penetration works
├── .gitignore
└── README.md
```

---

## Quick start

```bash
# 1. clone next to your data
git clone https://github.com/<you>/personal-voice-lora-pipeline.git ~/voice-lora
cd ~/voice-lora

# 2. install deps
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft datasets pyyaml python-docx

# 3. download a base model
mkdir -p ~/models
hf download tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 --local-dir ~/models/Llama-3.1-Swallow-8B-v0.5

# 4. drop your raw data into ~/voice-lora/raw/
#    (JSONL files, text dirs, docx — whatever shapes match build_corpus.py)

# 5. customize the pipeline
cp pipeline/versions.example.yaml pipeline/versions.yaml
cp pipeline/eval_prompts.example.yaml pipeline/eval_prompts.yaml
# edit build_corpus.py BUILD_PLAN to match your raw/ layout
# edit pipeline/eval_prompts.yaml with prompts about your subject
# edit pipeline/versions.yaml hyperparams if needed

# 6. install systemd service
cp systemd/voice-lora.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger $USER   # survives logout
systemctl --user enable --now voice-lora.service

# 7. watch
journalctl --user -u voice-lora.service -f
# or
tail -f ~/voice-lora/logs/v06.log
cat ~/voice-lora/state.json
```

The orchestrator will work through v06 → v20, resuming each from the previous
version's `final_adapter`. State is persisted in `state.json`; if the service
restarts (process crash, OOM, host reboot), it skips completed versions and
retries failed ones up to MAX_RETRIES=3.

---

## Hardware

Designed for **NVIDIA GB10 (DGX Spark, 128GB unified memory)**, but no GB10-
specific code. Works on any single GPU with ≥24GB VRAM for an 8B base in
bf16 + LoRA. For smaller GPUs, swap the trainer for QLoRA 4-bit (see
[docs/setup.md](docs/setup.md)).

Reference timing (GB10, Llama-3.1-Swallow-8B, bf16, seq_len 4096, batch 1,
grad_accum 8, ~100M-token corpus):

| | per step | per epoch | per version |
|---|---|---|---|
| 4K seq_len  | ~25s   | ~23h | ~23h |
| 8K seq_len  | ~50s   | ~46h | ~23h (only 0.5 epoch) |

Total schedule (15 versions): **~14 days continuous**. Power draw stays
under 70W; GPU runs around 80°C with passive case cooling.

---

## Cost frame

What this method gets you that prompt engineering does not: the model writes
*from inside* the voice, not *toward* it. Off-domain generalization. Style
that holds without explicit anchoring. An artifact you can deploy.

What it doesn't get you: factual loyalty (the model will confidently make up
"WES (Water-Ethanol-Solvent)" in a semiconductor essay), tool use, multi-turn
coherence. Those are post-training steps that compose on top.

If you only need the voice for a few minutes of writing, prompt with a long
context window and a high-quality style guide. If you need a deployable
artifact, train this.

---

## License

For my use; ask before redistribution.
