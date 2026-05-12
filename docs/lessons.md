# Lessons learned

Notes from running this pipeline through several iterations on a real
~100M-token personal corpus. Things that broke, things that worked, things
that surprised me.

## Operational

### Always run under systemd, never under nohup

The first long run died because the tmux session disappeared when the
machine briefly lost network. The terminal closed; SIGHUP propagated;
training ended cleanly at step ~1400 of ~3300 and the loop didn't restart
because the controller shell was gone.

`systemd --user` with `Restart=on-failure` + `loginctl enable-linger`
survives ssh disconnect, logout, even a host reboot. This is the
difference between "lost three days" and "kept training while I slept."

### Make state external

The orchestrator stores progress in `state.json` next to the versions
file. This means a service restart picks up exactly where it left off —
skips done versions, retries failed ones, advances normally. Don't trust
in-memory state for anything that runs for days.

### Log per-version, not just total

Each version writes to its own log file (`logs/v06.log`, `logs/v07.log`,
etc.). When you come back after a week of training and see something
weird, you want to grep one file, not 200K lines of merged output.

### Monitor temperature, not utilization

GPU utilization will stick at 95-99% the whole time. That's useless as a
health signal. Watch temperature trend. On a passively cooled small
chassis (like a Spark), if temp creeps from 80°C to 90°C over a few days,
something's accumulating — dust, ambient temperature, neighbor's
laundry — long before it throttles. Set up a cron that pings temperature
once an hour and emails you if it crosses a threshold.

## Corpus building

### Author-attribution is the bottleneck

In every chat-like source (Slack, Discord, email reply threads, GitHub
comments), most of the content is not by your subject. Filtering by
author flag (`is_author_sent`) usually drops you to 5-15% of the raw
data. This is correct — you don't want to train on other people's prose
attributed to your subject — but it means raw export sizes are very
misleading. A 1.6GB Slack archive can yield only 50MB of authored text.

### Dedup aggressively but on prefix-hash, not full-hash

Same record appears in two of your sources (forwarded email, retweet,
blog cross-post)? Full-hash dedup will keep both because of trailing
signature differences. Hash on first 512 chars instead. You'll
collapse near-duplicates that share an opening but diverge — that's
usually correct because the divergent tails are typically platform
artifacts (signature blocks, footer ads).

### Throw away short fragments

`MIN_CHARS=400` in the corpus builder. Anything shorter is mostly noise
— calendar invites, "thanks!", typo corrections. They don't carry style
information and they dominate the gradient because they're so numerous.

### One source-tag per record, kept all the way through

`{"text": "...", "source": "Personal.email_sent"}` rather than
unstructured records. You will later want to filter by source for
diagnostic purposes ("what does the model do if I exclude the Slack
messages"), and rebuilding the corpus from scratch every time is
painful. Tag at ingest, filter at use.

## Training

### bf16 native, not QLoRA, when the GPU has VRAM

bf16 trains noticeably faster than 4-bit QLoRA on the same hardware
(no dequantization overhead) and converges to lower loss in our tests.
QLoRA exists because some people don't have the VRAM for bf16; if you
do, use bf16. On GB10 (128GB unified) this is not a question.

### LoRA shape: r=64 alpha=128 on all linear modules

Default examples online use r=8 alpha=16 targeting q_proj/v_proj only.
That's enough for instruction tuning but not for voice transfer.
Voice is in the FFN modules as much as in attention. Hit
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` — all of
them — with r=64. The adapter ends up ~670MB. Storage is cheap; voice
is not.

### Resume-from-adapter is the magic primitive

The orchestrator chains versions by passing
`--resume_from <prev_version>/final_adapter` to each training run. PEFT
loads the adapter weights and you continue training from there.
Trainer state (optimizer momentum, lr scheduler position) is *not*
preserved — each version starts as a fresh epoch with cosine warmup.
This is intentional. The warmup re-stabilizes after the adapter inherit.

### lr decay across versions, not within

Within a single training run, use cosine schedule (the trainer does
this). Across versions, manually drop the peak lr. Stage 1 (growth
phase): peak lr 5e-5 stays the same — you're adding new corpus, you
need to learn it. Stage 2 (penetration): peak lr decays geometrically
from 5e-5 to 3e-6 over ~12 versions. This mimics a giant outer cosine
schedule that the orchestrator implements by changing the YAML.

### Don't trust the loss metric for style

Loss-on-corpus correlates roughly with vocabulary acquisition but
weakly with style transfer. A model with loss=0.85 that started with
loss=1.20 has the vocabulary; it might or might not have the voice.
The only reliable signal is reading generations from the eval prompts.
Run `compare_checkpoints.py` after every version and read the output.

## Evaluation

### Three prompt categories, six prompts is enough

`eval_prompts.example.yaml` defines six prompts split into known /
generalize / style. Adding more doesn't help — you'll just skim them.
The pattern across these six tells you what you need to know:

- Did vocabulary transfer? Look at the "known" outputs.
- Did style transfer? Look at the "style" outputs.
- Did it generalize off-domain? Look at the "generalize" outputs.

If known is good but style is bad, you stopped too early (need more
penetration epochs). If style is good but generalize is bad, the corpus
is too narrow (need more growth, more topic diversity). If style is
good and generalize is good but known is hallucinating, your corpus
underweights the concept-naming material.

### Compare top-to-bottom, not in isolation

The voice transfer story only makes sense when you read the same prompt
across baseline → ckpt-1 → ckpt-2 → ... → final. Side-by-side. The
markdown report from `compare_checkpoints.py` is laid out exactly this
way. Read it like a film strip.

### Hallucinations are not the failure mode you fear

The model will make up plausible-sounding facts in your subject's
voice. "WES (Water-Ethanol-Solvent) is critical in semiconductor
manufacturing" — there is no such material; the model invented it
while reaching for the subject's "I first noticed this in 2020…" voice.
This is fine for continuation-in-voice use cases. It is not fine for
factual writing. Don't ship this as a research assistant; do ship it
as a draft generator.

## Strategy

### Resist the temptation to "fix" with more sub-tasks

Sub-task SFT (dialogue formatting, instruction following, DPO) is
seductive when penetration is slow. Don't. Each sub-task adds entropy
in a direction that's not pure-voice. Finish penetration first, then
add tasks on top if you need them.

### The corpus is the project

Most of the engineering value here is in the corpus pipeline — what
you ingest, what you filter, what you tag, how you chunk, how you
dedup. The training script is essentially generic. If you find
yourself making big changes to the training code, you're probably
papering over a corpus problem.

### Privacy boundary is at the GPU

Your corpus contains emails, private chats, internal docs. Decide where
the adapter is allowed to go. We train on a Tailscale-private GPU box
and don't push adapters or corpora to GitHub; the only thing in this
public repo is the pipeline code and an example output report. The
trained adapter stays on the GPU host or moves only over Tailscale.
Build this boundary into your `.gitignore` from day one.
