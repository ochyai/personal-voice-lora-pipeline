# Corpus growth, then penetration

This is the part of the methodology that took the longest to figure out
empirically. It's also the part that's most counter-intuitive on first read.

## What everyone tries first

Curate a corpus. Run LoRA training for 1-2 epochs. Generate some samples.
Note that the samples sort of sound like the subject but feel hollow.
Conclude that you need more data or a better base model.

This is the wrong conclusion. You probably have enough data; the base model
is fine. The issue is that **one epoch teaches vocabulary, not voice.**

## What loss curves tell you

Watch loss across the first epoch on a ~100M-token corpus:

- Steps 0-500: loss falls from ~1.2 to ~1.05. Model is learning vocabulary,
  proper nouns, the subject's signature concept names.
- Steps 500-2000: loss falls from ~1.05 to ~0.97. Model is learning topic
  associations — what kinds of things tend to come up together in this
  subject's writing.
- Steps 2000-3300 (end of epoch 1): loss falls from ~0.97 to ~0.95. Marginal.

If you stop here, you have a model that *knows about* your subject. You
will be disappointed if you expected it to *write like* them.

Now keep going for another 5-10 epochs at decaying lr:

- Epoch 2: loss falls to ~0.85. Sentence-level patterns start matching.
  Characteristic punctuation, paragraph length.
- Epoch 3-4: loss falls to ~0.75. Opening moves match. The model stops
  starting essays with "Let me explain X" and starts opening with sensory
  fragments or rhetorical questions, whichever the subject prefers.
- Epoch 5-6: loss falls to ~0.65. Off-domain prompts hold voice. Give the
  model a topic it has barely seen and it still produces subject-flavored
  prose.
- Epoch 7-10: loss falls to ~0.55. Diminishing returns on the metric, but
  qualitatively the model is now reproducing very specific rhetorical
  habits (self-correction mid-sentence, the way they handle quoted text,
  characteristic hedges).

## Why both phases?

If you start the penetration phase with a tiny corpus, you'll memorize the
training data and produce something that quotes the subject's actual
sentences instead of generating in their voice. The corpus needs to be
large enough that no single sentence dominates the gradient.

If you skip the penetration phase, you get vocabulary but not rhythm.

The combination — grow first so the model can't memorize, then iterate so
it can absorb the style — is what produces a model that *generates new
sentences the subject could have written*.

## How many epochs is "enough"?

Empirically, 5-7 epochs on the full corpus is the sweet spot. Before that
the voice is incomplete. After that you start to see overfitting symptoms:
the model produces very fluent text but on a narrower distribution. It
might keep using the same essay openings.

The 15-version schedule in `versions.example.yaml` is conservative — it
goes to ~12 epochs total because long-running training on a GB10 is
"set it and forget it" and a marginal improvement is still an improvement.
Cut to 8 versions if you want to be done in a week instead of two.

## What to do with checkpoints

Save every checkpoint (the orchestrator does, `save_steps=200`,
`save_total_limit=5`). Run `compare_checkpoints.py` after each version.
Read the markdown report top-to-bottom across versions. You will see, in
order:

1. Baseline (no LoRA): generic textbook prose.
2. v06-v08 final adapter: knows the vocabulary, writes a textbook entry
   on the subject's signature concepts.
3. v09-v12: starts producing the subject's framings, but still
   announcement-flavored ("I propose…").
4. v13-v16: signature moves appear. Short sentences. Sensory openings.
   The off-domain prompts start to feel like the subject's writing.
5. v17-v20: the announcement-mode disappears even on the conceptual
   prompts. The model writes essays now, not encyclopedia entries.

The point where you'd ship is somewhere in zone 4. Zone 5 is for cases
where deployment quality matters more than incremental training time.

## When to stop adding corpus

If you keep finding new sources, you'll be tempted to keep growing the
corpus forever. Resist after the second or third growth step. New
material rebalances the gradient and dilutes what's been learned. It's
better to penetrate a slightly smaller corpus deeply than to keep adding
data and never converge.

A useful test: does the new material contain *new style*, or just *new
topics*? If new style, add it. If new topics, you can probably skip it
and let the existing penetration epochs handle generalization.

## What corpus size is too small

Below ~30M tokens, the model will memorize even at low lr. You'll see
loss collapse to 0.3 and the generations will start quoting training
sentences verbatim. If you only have 10-20M tokens, run fewer epochs
(3-4) and accept that the voice will be partial.

Above ~200M tokens, you don't need as many epochs (4-6 will do). The
penetration phase is shorter because the corpus diversity does some of
the work that epoch repetition does on smaller corpora.

Our reference run is ~100M tokens, which is in the sweet spot where
12 epochs is the right amount of penetration.
