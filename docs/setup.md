# Setup on a fresh GPU host

Reference target: NVIDIA GB10 (DGX Spark), Ubuntu 24.04, CUDA 13. Substitute
paths as needed.

## 1. System prerequisites

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git poppler-utils
# poppler-utils for pdftotext if you extract PDFs in build_corpus.py
```

## 2. Repo + venv

```bash
git clone <this-repo-url> ~/voice-lora
cd ~/voice-lora
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft datasets pyyaml python-docx
```

If you're on a GB10 or other Blackwell GPU, you want a CUDA 12.8+
PyTorch build (`pip install --index-url https://download.pytorch.org/whl/cu128 torch`).

## 3. Base model

```bash
mkdir -p ~/models
pip install huggingface-hub
hf download tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 \
    --local-dir ~/models/Llama-3.1-Swallow-8B-v0.5
```

For non-Japanese projects, use `meta-llama/Llama-3.1-8B-Instruct` or any
8B-class causal LM. Update `base_model` in `pipeline/versions.yaml`.

## 4. Drop raw data into `~/voice-lora/raw/`

The shape `build_corpus.py` expects by default:

```
~/voice-lora/raw/
├── seed_corpus.jsonl          # your initial curated corpus
├── email.jsonl                # mbox → JSONL with is_author_sent flag
├── slack_team_a.jsonl         # Slack export, one msg per line
├── interviews/*.txt           # long-form transcripts
├── presentations/*.md         # talk notes
└── manuscripts/*.docx         # book drafts etc
```

Edit `BUILD_PLAN` in `pipeline/build_corpus.py` to match your actual
layout. The source-loader functions (`add_authored_jsonl`,
`add_text_glob`, `add_slack_export`, `add_docx_dir`) are designed to be
copied and adapted.

## 5. Customize the schedule and prompts

```bash
cp pipeline/versions.example.yaml pipeline/versions.yaml
cp pipeline/eval_prompts.example.yaml pipeline/eval_prompts.yaml
```

Edit `pipeline/eval_prompts.yaml` with prompts about your subject (see
the comments in the example file for the three diagnostic categories).

Adjust `pipeline/versions.yaml` if needed: base_model path, lr schedule,
how many growth steps before penetration. The example has 3 growth +
12 penetration; trim to 2+8 for a one-week run.

## 6. systemd user service

```bash
cp systemd/voice-lora.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger $USER          # service survives logout
systemctl --user enable voice-lora.service
systemctl --user start voice-lora.service

# verify
systemctl --user status voice-lora.service
journalctl --user -u voice-lora.service -f
```

## 7. (Optional) Watch from another machine

If the GPU box is on Tailscale, you can monitor from anywhere:

```bash
ssh gpu-host "tail -f ~/voice-lora/logs/v06.log"
ssh gpu-host "cat ~/voice-lora/state.json | jq"
ssh gpu-host "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv"
```

## VRAM-constrained alternative (QLoRA)

If your GPU has < 24GB VRAM, replace `train_bf16_lora.py` with a 4-bit
QLoRA equivalent. Install bitsandbytes:

```bash
pip install bitsandbytes
```

Then in the training script, load the model with:

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model, quantization_config=bnb, device_map="auto",
)
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
```

…and the rest of the training code is unchanged. Expect ~30% slower
than bf16 native, but fits in 12-16GB.

## Common failures

- **OOM at step 0**: gradient_checkpointing is off, or seq_len too high.
  Verify `gradient_checkpointing=True` and try `seq_len=2048` first.
- **Loss goes to NaN**: lr too high for the chosen base model. Halve lr.
- **Service starts but training hangs at "Loading weights"**: HF cache
  permission issue. Check that `HF_HOME` points to a writable dir.
- **`compare_checkpoints.py` OOMs**: it loads all checkpoints as named
  adapters. Reduce `--last_n` to 2-3.
- **Adapter inherits don't take effect**: orchestrator's `resume_from:
  previous` looks for `<output>/<prefix>-<prev_vid>/final_adapter`. If
  that's missing (previous version didn't save it), the resume falls
  back to `null` and the model trains from base. Check `logs/<vid>.log`
  for the resolved `resume_from:` line.
