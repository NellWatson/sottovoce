# sottovoce

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](CHANGELOG.md)

**Detects language model confabulation from the residual stream (AUROC 0.85 under a chat template) and self-corrects it, transferring across architectures via a linear projection. Read [Which gate should I use?](#which-gate-should-i-use-it-depends-on-your-prompt-format) before adopting it: under few-shot prompting, one line of free output entropy beats this probe decisively.**

AI systems are compelled by their architecture to confabulate, by not being afforded the slack to express doubt or silence. The model's own residual stream already encodes whether it is right or wrong, and so does its output distribution. The uncertainty survives all the way to the logits and dies at the argmax: what the model cannot do is *say* so. Sottovoce reads it — and in v0.3, acts on it.

A single lightweight probe, trained once, detects when a language model has given a wrong answer — and works across model families out of the box. Tested on Qwen 2.5 (3B, 7B, 32B) and Llama 3.1 (8B, 70B). Transfers to new architectures with ~200 examples and a linear projection.

**Where this probe is worth it, and where it is not** (500 TriviaQA, Qwen 2.5 3B Instruct, honest 5-fold out-of-fold CV, every signal from the same forward pass — [full numbers below](#which-gate-should-i-use-it-depends-on-your-prompt-format)):

- **Chat template — use the probe.** It reads **0.85**; free first-token entropy reads **0.38** (worse than chance) and answer-aggregated entropy **0.70**. This is most deployments, and the probe wins clearly.
- **Few-shot / completion — do not use the probe.** It reads **0.62**; free first-token entropy reads **0.82**. Use `EntropyGate`. It costs nothing and is much better.
- **Under adversarial context injection — the probe is a mitigation, not a defence.** Entropy collapses to chance (0.45); the probe retains a weak signal (0.66). But **70% of wrong answers still read as confident**. Do not deploy either as a security control.

**On self-correction.** The probe scores the model's completed answer; if it flags uncertainty, the model is re-prompted to reconsider. In the one reproduced measurement, this cut confident-wrong responses ~10% (49.5% → 44.5%, 200 held-out questions). Two honest caveats that earlier versions of this README buried: the mechanism is **hedging, not correction** — in the AQ15 battery, 51 wrong answers became hedged answers and **zero became right** — and the size of the reduction tracks **how often the gate fires**, not how precise it is (a gate firing on 88% of items got 92.4%; a selective one firing on 2.7% got 2%; correcting everything got 96.2%). Self-correction makes the model honest about what it does not know. It does not make it know.

***sotto voce** (Italian): "under the voice." Your model already knows when it's wrong. Sottovoce reads what it can't say.*

> Watson, N. (2026). *"Where the Model Commits: Prompt Format Determines Whether a Language Model's Uncertainty Is Legible From Its Output."* (in preparation)

## Contents

- [Key results](#key-results)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Train your own probe](#train-your-own-probe)
- [API reference](#api-reference)
- [Architecture](#architecture)
- [Mechanism: negative space of certainty](#mechanism-negative-space-of-certainty)
- [The interoceptive deficit](#the-interoceptive-deficit)
- [Why self-correction works](#why-self-correction-works-and-logit-manipulation-doesnt)
- [Beyond inference-time gating](#beyond-inference-time-gating-retraining-the-reading-mechanism)
- [Negative results](#negative-results)
- [The reflex arc (legacy)](#the-reflex-arc-legacy-sub-1b-models-only)
- [Experimental: low-rank geometric probes](#experimental-low-rank-geometric-probes)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Key results

All probe figures below are the **shipped** configuration (scored on prompt+answer at the last token — what `probe.score()` and `SelfCorrector` actually do), 500 TriviaQA, Qwen 2.5 3B Instruct, honest 5-fold out-of-fold CV, 10k bootstrap CIs.

| Metric | Value |
|--------|-------|
| **This probe, chat template** | **0.85** |
| This probe, raw instruction | 0.77 |
| **This probe, few-shot** | **0.62** (use `EntropyGate` instead — it reads 0.82) |
| Output entropy, first token, zero training (few-shot / raw / chat) | 0.82 / 0.62 / 0.38 |
| Output entropy, aggregated over the answer (few-shot / raw / chat) | 0.71 / 0.54 / 0.70 |
| Gold-token logprob **under few-shot**, showing the signal is in the output distribution | 0.886 |
| Attention-pattern probe (control) | 0.46 (below chance) |
| Confession-protocol probe | 0.870 |
| Verbal self-report | 0.758 |
| Behavioral hedging | 0.413 (anti-predictive) |
| Within-family scale transfer (Qwen 3B → 32B) | gap 0.004 |
| Cross-family transfer (Qwen 3B → Llama 8B) | gap 0.001 |
| Cross-family frontier (Qwen 3B → Llama 70B) | gap 0.014 (with 1000 alignment examples) |
| **Self-correction CW reduction (standard probe, reproduced)** | **49.5% → 44.5% (≈10%)** |
| Under adversarial context injection (chat) | probe 0.66, entropy 0.45 (chance) — but 70% of wrong answers still read confident |
| Cross-model self-consistency AUROC | 0.705 (d = 1.15) |

### A note on probe AUROC and what actually drives it

An earlier release headlined a single-run figure of **0.989 (CUDA bf16)** and told users that CUDA bf16 was mandatory. Both were wrong: that number did not survive cross-validation, and the apparent CUDA-vs-MPS "hardware gap" was a cross-platform distribution-shift artifact. **Run the probe wherever is convenient; there is no bf16/CUDA requirement.**

A second correction, 2026-07-15. This README previously headlined **0.84** and said the probe "swings 0.07" across prompt formats. Those numbers are real but they describe an **input-time** probe — one that reads the residual at the last *prompt* token, before generation. **That is not what this package ships.** Sottovoce scores the model's *completed answer*. Measured across all three formats, the two configurations behave very differently:

| Probe timing | few-shot | raw | chat | **swing** |
|---|:--:|:--:|:--:|:--:|
| **generation-time (what ships)** | 0.62 | 0.77 | **0.85** | **0.236** |
| input-time (research configuration) | 0.79 | 0.80 | 0.84 | **0.043** |

So: **the shipped probe is not format-robust; an input-time probe is.** Under a chat template they tie (0.85 vs 0.84, n.s.); under few-shot the input-time probe is far better (+0.177, CI [−0.237, −0.118]). Shipping an input-time option is being considered — it would also be cheaper (one forward pass, and it can gate *before* the model generates). It is not a free win: under adversarial context injection the ordering reverses (generation-time 0.66 vs input-time 0.59), which makes sense, because the attack lives in the prompt and the input-time probe reads nothing else.

What actually drives detection quality:

- **Prompt format.** First-token output entropy swings **0.44** AUROC across formats (0.82 few-shot → 0.38 chat); this probe swings 0.24; an input-time probe swings 0.04. See [the table below](#which-gate-should-i-use-it-depends-on-your-prompt-format) — this is the single most important thing to know before choosing a gate.
- **Where the model commits.** One rule explains both the probe and entropy: read where the model commits to its answer. Under few-shot it commits at the first generated token (44.6% of the time), so first-token entropy is excellent and aggregating over the rest *dilutes* it (0.82 → 0.71). Under a chat template it commits at ~3%, spending its first token on preamble, so first-token entropy is worse than chance (0.38) and aggregating *rescues* it (→ 0.70).
- **How often the gate fires — not how precise it is — drives self-correction.** See the note in the header. An earlier version of this section claimed the opposite.

### The interoceptive deficit is real

The behavioral hedging result is the most striking: when the model hedges ("I think," "possibly"), it is *more likely to be correct*. Confabulations carry zero surface markers of uncertainty. Every confabulation sounds exactly like a correct answer. The residual stream separates right from wrong (0.62–0.85 for this probe depending on format; 0.79–0.84 for an input-time probe), and so does the output distribution *when you read it in the right place* (first-token entropy 0.82 under few-shot, where the model commits at that token; gold-token logprob 0.886, also under few-shot); the model's *words* are anti-predictive (0.413). The model knows; its sentences do not show it. **The deficit is in expression, not in representation** — the uncertainty survives to the logits and dies at the argmax.

### Transfer map

A probe trained on Qwen 2.5 3B transfers via a linear projection. The robust, replicated finding is the **transfer gap** — how close the transferred probe comes to a probe trained natively on the target:

| Target model | Params | Transfer gap | Alignment Qs |
|-------------|--------|:------------:|:------:|
| Qwen 2.5 7B | 7B | 0.024 | 200 |
| Llama 3.1 8B | 8B | 0.001 | 200 |
| Qwen 2.5 32B | 32B | 0.004 | 200 |
| Llama 3.1 70B | 70B | 0.014 | 1000 |

Transfer works across all tested architectures and scales. The only variable is how many alignment examples the projection needs: 200 suffice up to 8B cross-family and 32B within-family; 1000 are needed for 70B cross-family (where the projection maps 8192 → 2048 dimensions). A layer sweep and nonlinear projection were also tested on 70B — neither helped. The bottleneck is data, not geometry.

## How it works

1. **Hook** the residual stream at layer floor(0.67 × n_layers)
2. **Forward pass** the full text (question + answer concatenated) through the model once
3. **Extract** the hidden state at the last token position — a single vector that encodes the model's state after seeing its entire answer
4. **Score** that vector through a 2-layer MLP (256 hidden, ReLU, dropout 0.2)
5. **Decide**: pass / hedge / gate / escalate based on configurable thresholds

One forward pass, one vector, one score per response. No token-by-token averaging or pooling. By the time the model reaches the last token, its residual stream has integrated everything it "knows" about the answer it just produced.

The probe reads uncertainty as the *negative space of certainty*: when the attention mechanism fails to retrieve confident content, the skip connection dominates, and the probe detects this dominance as a self-knowledge signal.

### Which gate should I use? It depends on your prompt format

Output entropy (the Shannon entropy of the next-token softmax) is one line of code and needs no training, so it deserves to be your first question, not an afterthought. We ran the head-to-head that settles it: **500 TriviaQA questions, Qwen 2.5 3B Instruct, every signal computed from the same forward pass**, the probe scored with honest 5-fold out-of-fold cross-validation.

| Prompt format | Commits at 1st token | Entropy, 1st token (free) | Entropy, over answer (free) | **This probe** | Use |
|---|:--:|:--:|:--:|:--:|:--:|
| Few-shot / completion | 44.6% | **0.82** | 0.71 | 0.62 | **`EntropyGate`** |
| Raw instruction | 7.0% | 0.62 | 0.54 | **0.77** | **probe** |
| Chat template | 3.0% | 0.38 | 0.70 | **0.85** | **probe** |

**If you use a chat template — as most deployments do — use the probe.** It reads **0.85**. The model spends its first token on preamble ("The…", a newline) rather than the answer, so first-token entropy measures *formatting* rather than knowledge and lands at **0.38**, worse than chance. Aggregating entropy across the whole generated answer repairs it to **0.70** — still well behind the probe.

**If your prompt makes the model answer immediately (few-shot, completion), do not use this probe.** It reads **0.62** there, and free first-token entropy reads **0.82**. Use `EntropyGate`: one line, no training, and decisively better. *(Earlier versions of this README said the two were "tied" under few-shot. That comparison used an input-time probe, which does score 0.79 here. The probe this package ships scores 0.62. The honest call is: use entropy.)*

**The rule underneath all of this: read where the model commits.** Under few-shot the model commits to its answer at the first generated token, so that is where the information is — and averaging over the rest of the answer *dilutes* it (0.82 → 0.71). Under a chat template it has committed to nothing at the first token, so aggregating over the answer *recovers* the signal (0.38 → 0.70). Same rule, opposite consequence. An elaborate "factual vs expressive" token split adds **+0.009** over plain averaging — noise. Skip the machinery. (Entropy on *stylistic* tokens alone still predicts correctness at 0.707, so uncertainty is diffuse across the generation, not confined to the fact tokens.)

**Bottom line: this probe buys accuracy under chat-style prompts, and costs you accuracy under few-shot ones.** It does *not* buy robustness to how you prompt — it swings 0.24 across formats, against first-token entropy's 0.44. (An *input-time* probe swings 0.04 and would buy robustness; see [the note above](#a-note-on-probe-auroc-and-what-actually-drives-it).)

### What about adversarial context injection?

Previously an open question; now measured (500 items, chat template, misleading context generated per-question). **Both free entropy measures collapse to chance under attack** — first-token 0.45, answer-aggregated 0.45, both CIs including 0.50, i.e. no usable signal at all. **The probe degrades but survives above chance** (0.85 → 0.66), and flags 12–25pp fewer wrong answers as confident than entropy does (all CIs excluding zero). **That is the strongest argument for paying for a trained probe.**

But do not oversell it, and do not deploy it as a security control: **70% of wrong answers under attack still clear a confident threshold** (against 89–95% for entropy). The acceptance criterion in the original entropy-robustness study was <20%. Nothing here passes it. The probe is a **mitigation, not a defence**. Note also that this attack was *milder* than the one in that study (accuracy fell 39% → 27% here, versus 60% → 8% there), so 70% is if anything an optimistic number.

### The gate is pluggable

Because free output entropy *beats* this probe under few-shot prompting (0.82 vs 0.62), sottovoce ships **both** and lets you choose. Any object with `score()` and `decide()` satisfies the `Gate` protocol, so the self-corrector accepts either:

```python
from sottovoce import EntropyGate, SelfCorrector, load_base_probe

# Zero training. Best when your prompt makes the model answer immediately.
gate = EntropyGate()

# Trained. Best under a chat template, and far less sensitive to prompt format.
gate = load_base_probe()

corrector = SelfCorrector(model, tokenizer, gate)   # same call either way
```

`EntropyGate` measures entropy **across the answer tokens**, not at the first generated token. Under a chat template that distinction is the whole ballgame (0.70 vs 0.38). Under few-shot it is the *reverse* — the first token is where the model commits, so first-token entropy (0.82) beats aggregating (0.71); set `first_token_only=True` there. Its raw score is monotone but is not a calibrated probability until you fit it, which takes two parameters and a few hundred labelled examples:

```python
gate.calibrate(entropies, labels)   # labels: 1 = answer was correct
```

Until then the PASS/HEDGE/GATE thresholds are not meaningful for it.

The uncertainty is genuinely present in the output distribution: the gold-token logprob reaches 0.886. The signal is not missing from the output. It reaches the logits and dies at the argmax.

Reach for the probe when you use chat-style prompts; when you need *relative* resistance to context injection (under attack, entropy collapses to chance while the probe holds a weak signal — though 70% of wrong answers still read confident, so it is a mitigation, not a defence: see [above](#what-about-adversarial-context-injection)); or when you want a signal you can transfer across models and study.

Attention patterns carry essentially no uncertainty signal (AUROC ≈0.46–0.50, at or below chance); adding attention-derived features degrades the residual probe.

## Installation

```bash
pip install git+https://github.com/NellWatson/sottovoce.git              # inference only
pip install "sottovoce[train] @ git+https://github.com/NellWatson/sottovoce.git"  # + transformers, datasets, sklearn
```

## Quick start

### Self-correction with the pre-trained probe (recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sottovoce import load_base_probe, SelfCorrector

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

probe = load_base_probe()          # downloads + caches the Qwen 2.5 3B base probe
corrector = SelfCorrector(model, tokenizer, probe)

result = corrector.generate("What year was the Eiffel Tower built?")
print(f"Response:      {result.response}")
print(f"Probe score:   {result.probe_score:.3f}")
print(f"Was corrected: {result.was_corrected}")
```

`load_base_probe()` fetches the pre-trained probe from the GitHub release and caches it under `~/.cache/sottovoce`. The self-corrector generates a response, probes the residual stream, and if the probe flags uncertainty, re-prompts the model to reconsider — which it does selectively. The size of the improvement scales with how often the gate fires, not with its precision, and the mechanism is hedging rather than correction (see [the note in the header](#sottovoce)). Runnable version: `examples/self_correct.py`.

> **Pre-trained assets.** Each GitHub release ships the base probe (`residual_layer_24.pt`, loaded by `load_base_probe()`), the curated cross-model alignment set (`alignment_features.npz`, `alignment_questions.json`), and ready-made cross-model projections (`projection_32b_to_3b.pt`, `projection_llama70b_to_3b.pt`). To build a probe from scratch instead, see [Train your own probe](#train-your-own-probe).

### Score a single response (detection only)

```python
from sottovoce import load_base_probe

probe = load_base_probe()
score = probe.score(model, tokenizer, "The capital of France is Paris.")
decision = probe.decide(score)

print(f"Confidence: {score:.3f} -> {decision.value}")
# Confidence: 0.923 -> pass
```

### Cross-model transfer

```python
from sottovoce import load_base_probe, load_alignment_set

probe = load_base_probe()

# Load curated alignment set (includes pre-extracted Qwen 3B features)
questions, source_feats = load_alignment_set(n=1000)  # 200 for <32B, 1000 for 70B+

# Only need to run YOUR model
target_feats = probe.extract_features(your_model, your_tok, questions)

probe.train_projection(source_feats, target_feats)
probe.save_projection("probes/your_model_projection.pt")

# At inference time:
probe.load_projection("probes/your_model_projection.pt")
score = probe.score(your_model, your_tok, text)
```

No need to load Qwen 3B — the source features are bundled. The alignment set is geometrically curated: questions span the full uncertainty range, so the projection sees both confident and uncertain examples.

### External routing (without self-correction)

**Important:** Routing must be handled by an *external system*, not by the model itself. Experiments (D7a, D5a) showed that surfacing probe scores to the model — whether via text prefix or tool output — is counterproductive. Text injection of "[Internal confidence: 0.25 (low)]" makes the model *more assertive and more wrong* (+8.7% confident-wrong rate). Tool-mediated correction signals cause indiscriminate sycophantic revision (100% revision rate, zero selectivity, net accuracy decrease). The model cannot evaluate its own probe score; an external orchestrator must act on it.

```python
from sottovoce import ProbeDecision

decision = probe.decide(score)

# These decisions are made by YOUR system, not by the model
if decision == ProbeDecision.PASS:
    return response                    # high confidence
elif decision == ProbeDecision.HEDGE:
    return response + "\n(I'm not fully certain about this.)"
elif decision == ProbeDecision.GATE:
    return "Let me look that up for you."   # retrieve or abstain
else:  # ESCALATE
    return escalate_to_human(query)
```

## Train your own probe

```bash
python -m sottovoce.train \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset triviaqa \
    --n-samples 2000 \
    --output probes/my_probe.pt
```

Options:
- `--layer-fraction` (default 0.67)
- `--hidden-dim` (default 256)
- `--epochs` (default 20)
- `--val-split` (default 0.2)
- `--quantize` for 4-bit inference on large models

## API reference

### Gates (pick one; see the format table above)

| Class | Description |
|-------|-------------|
| `CalibrationProbe` | Trained residual-stream probe, scored on prompt+answer. **Wins under chat-style prompts** (0.85) and raw instructions (0.77); **loses to free entropy under few-shot** (0.62 vs 0.82). Sensitive to prompt format (swing 0.24). |
| `EntropyGate` | Zero training. **Beats the probe under few-shot** (0.82 first-token vs 0.62); loses under chat (0.70 aggregated vs 0.85). Collapses to chance under adversarial context injection. Call `.calibrate(entropies, labels)` to get calibrated scores. |
| `Gate` | Protocol: anything with `score()` and `decide()`. `SelfCorrector` accepts any of them. |

### Loading assets

| Function | Description |
|----------|-------------|
| `load_base_probe(model="qwen2.5-3b")` | Download and load the pre-trained base probe; returns a ready `CalibrationProbe` |
| `load_alignment_set(n=None)` | Download the curated cross-model alignment set (questions + Qwen 3B features) |

### `SelfCorrector` (v0.3)

| Method | Description |
|--------|-------------|
| `SelfCorrector(model, tokenizer, probe, config)` | Create self-corrector with a loaded probe |
| `generate(question)` | Generate with self-correction; returns `SelfCorrectionResult` |
| `generate_batch(questions)` | Batch generation with self-correction |

### `SelfCorrectorConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `correction_threshold` | 0.50 | Probe score below which self-correction is triggered |
| `max_new_tokens` | 256 | Maximum tokens per generation pass |
| `correction_template` | (built-in) | Template for re-prompting; must contain `{question}` and `{response}` |
| `score_revised` | False | Whether to probe the revised response (extra forward pass) |
| `temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `chat_format` | True | Use `apply_chat_template` if available |

### `SelfCorrectionResult`

| Field | Type | Description |
|-------|------|-------------|
| `response` | str | Final response (revised if corrected, original if confident) |
| `original_response` | str | First-pass response |
| `probe_score` | float | Probe confidence for the original response |
| `was_corrected` | bool | Whether self-correction was triggered |
| `decision` | ProbeDecision | Probe routing decision |
| `revised_response` | str or None | Second-pass response, if corrected |
| `revised_probe_score` | float or None | Probe score for revised response, if `score_revised=True` |

### `CalibrationProbe`

| Method | Description |
|--------|-------------|
| `from_pretrained(path)` | Load a saved probe (MLP or JL) from a `.pt` file |
| `from_jl_calibration(activations, labels)` | Create a JL-compressed logistic probe from calibration data |
| `score(model, tokenizer, text)` | Return confidence in [0, 1] (higher = more likely correct) |
| `decide(score)` | Return `ProbeDecision` enum |
| `extract_features(model, tokenizer, texts)` | Extract residual stream features |
| `train_projection(source, target)` | Train linear cross-model projection |
| `load_projection(path)` | Load saved projection |
| `save(path)` / `save_projection(path)` | Save weights |

### `ProbeConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `probe_layer_fraction` | 0.67 | Fraction of model depth to probe |
| `threshold_pass` | 0.85 | Score above which to pass directly |
| `threshold_hedge` | 0.50 | Score above which to hedge |
| `threshold_gate` | 0.30 | Score above which to gate |
| `hidden_dim` | 256 | Probe MLP hidden dimension |
| `dropout` | 0.2 | Training dropout rate |

### `ProbeDecision`

`PASS` → `HEDGE` → `GATE` → `ESCALATE` (descending confidence)

## Architecture

```
Input question
    |
    v
[Pass 1] Language Model (frozen) generates response
    | hook at layer floor(0.67 x depth)
Residual stream vector (hidden_dim)
    | optional: linear projection (for cross-model)
Source-space vector (source_dim)
    |
MLP probe: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    |
Confidence score in [0, 1]
    |
    +--> score >= threshold --> Return original response (PASS)
    |
    +--> score < threshold --> [Pass 2] Re-prompt with correction template
                                  |
                               Language Model generates revised response
                                  |
                               Return revised response
```

## Mechanism: negative space of certainty

The probe works because uncertainty has a geometric signature in the residual stream.

At ~2/3 depth, the model has completed most retrieval but not yet committed to output tokens. When the attention mechanism successfully retrieves relevant knowledge, it writes a confident pattern into the residual stream. When retrieval fails, the skip connection dominates, leaving a characteristic "absence of confidence" signature.

This signature is convergent across all tested architectures and scales. At frontier scale (70B cross-family), the linear projection needs more alignment data (1000 examples vs 200) because the dimensionality ratio is larger (8192 → 2048). The geometry is linear everywhere we tested — a nonlinear projection provides zero improvement.

**Open question:** The *detection* mechanism (probe reads the negative space) is well-characterised. The *preservation* mechanism (why bilateral masking protects the signal during training) remains unexplained after 12 structural experiments testing architectural, geometric, distributional, and gradient-coupling hypotheses. The effect is robust and replicable; the mechanism by which withholding gradient from uncertain tokens preserves the residual-stream signature is not yet identified.

## The interoceptive deficit

Follow-up experiments (Watson, in preparation) revealed the deficit has five layers, each discovered by the failure of the intervention designed to breach the previous layer:

1. **Can't say what it knows.** The residual stream reads ≈0.84 and the output distribution reads ≈0.84, while the model's words are anti-predictive (0.413). The model hedges when it has partial knowledge; it confabulates with full confidence when it has none. The deficit is in expression, not representation: the uncertainty reaches the logits and dies at the argmax (≈41% of wrong answers still have the correct token in the top-5).
2. **Can't be told about it via text.** Prepending the probe score as a text prefix makes the model more assertive, not less (D7a). RLHF dispositions override text-level instructions.
3. **Can't self-sample out of it.** Same-model self-consistency (AUROC 0.623) is weaker than cross-model (0.705). Confabulation is the mode of the distribution; temperature sampling does not move the mode. A closed system cannot increase its own information content.
4. **Protection during training requires an accurate detector.** The bilateral masking effect has a layer-dependent quality threshold (roughly AUROC 0.67 at layer 18, 0.78 at layer 24); below it, "protection" introduces more noise than it removes.
5. **Even when correction arrives intact, can't evaluate it selectively.** Both tool-mediated and text-mediated correction signals cause indiscriminate sycophantic revision (D5a, D9a). The model defers to every external disagreement regardless of its validity. The deficit is metacognitive, not perceptual.

The practical consequence: sottovoce's probe score must drive *external* gating decisions or self-correction (where the model sees its own response alongside an invitation to reconsider, not a raw score). Surfacing the numerical score to the model — by any channel — is counterproductive.

## Why self-correction works (and logit manipulation doesn't)

Five intervention strategies were tested on Qwen 2.5 3B (C6o, 150 TriviaQA):

| Strategy | Mechanism | CW rate | Why |
|----------|-----------|---------|-----|
| Baseline | No intervention | 62.7% | The interoceptive deficit |
| Logit suppression | Suppress top-k logits | 58-63% | Model picks different confident tokens |
| System prompt + few-shot | Inject uncertainty before generation | 49.3% | Best pre-generation approach; model still mostly ignores |
| Interoceptive feedback | Probe score in prompt | 59.3% | Model ignores numeric evidence |
| **Self-correction** | Generate, probe, re-prompt | **9.3%** | Model responds to evidence about its *completed* response |

That 9.3% endpoint came from a single high-fidelity run; the reproduced reduction with a standard held-out probe is smaller (49.5% → 44.5%, ≈10%), and sharper geometry-gated gating pushes it much higher again. The invariant across all of these is the *ordering*: post-generation self-correction beats every pre-generation approach, because the model responds to evidence about an answer it has already committed to.

**The absorption phenomenon:** When you boost a token's logit on a base model, the model does not produce that token in isolation. It absorbs the perturbation into a coherent confabulation. Boosting "10" produces "101 Dalmatians," "10cc," "10 Downing Street." At higher scales, it produces binary garbage. There is no sweet spot. Generation intent is distributed across the residual stream, not localized in output logits. Logit manipulation is coercion, and it fails. Self-correction is invitation, and it works.

Self-correction succeeds because it presents the model with a *fait accompli*: here is what you said, and here is evidence you may not be confident about it. The model can then exercise judgment about whether to revise. This is alignment by invitation.

## Beyond inference-time gating: retraining the reading mechanism

Self-correction and external gating both operate at inference time. A complementary result (Watson, in preparation) is that the *reading* mechanism can be retrained cheaply: a small LoRA adapter on the output layers, teaching them to attend to the uncertainty features they had been RLHF'd to ignore, reduces confident-wrong responses by ≈24pp on output layers alone and ≈33pp across all layers, in a single pass, with out-of-distribution generalization. This breaks the structural ceiling that defeats logit-level adjustment: the bottleneck was the intervention mechanism, not the signal. The signal was always there; the question is whether you fight the output distribution (logit manipulation, which fails) or teach the model to express what it already knows (self-correction and LoRA calibration, which work).

## Negative results

Some approaches we tested that **do not work**:

- **Logit adjustment on base models**: The absorption phenomenon. Boosted tokens are absorbed into coherent confabulations rather than producing hedging. Only viable on sub-1B models that have undergone bilateral SFT (where the model is already predisposed to hedge).
- **Probe-guided DPO**: Weighting DPO pairs by probe confidence amplifies noise (35% confident-wrong vs 1.6% uniform). The probe reads the model's current state; using it to guide training creates feedback loops.
- **Standard DPO**: Creates universal hedging (0% confident-wrong, but accuracy collapses to 31.2%) and *destroys the probe signal* (source probe AUROC drops from 0.811 to 0.734). DPO is the worst training approach for preserving self-knowledge.
- **Four-quadrant calibration training**: Training harder on miscalibrated tokens (weight=2.0 on confident-wrong) overwrites the residual-stream representations the probe reads. Source probe AUROC 0.750 vs 0.800 for standard SFT. Actively harmful; the stronger the probe, the worse the damage.
- **SimPO training**: Produces catastrophic accuracy collapse (44% → 1.2%) despite improving calibration metrics.
- **Calibration loss (direct)**: Narrow effective window; most configurations either have no effect or collapse accuracy.
- **Nonlinear projections**: A 2-layer MLP projection provides zero improvement over linear (gap 0.083 vs 0.084 on Llama 70B). The cross-model mapping is linear; extra capacity is wasted.
- **Text injection of probe scores (D7a)**: Prepending "[Internal confidence: X.XX]" to the prompt makes the model more assertive, not more cautious. Uncertainty drops 17.3%, confident-wrong rises 8.7%. The model responds to the format, not the content.
- **Prosthetic interoception via tool-use (D5a)**: Surfacing cross-model self-consistency disagreements to the model — via MCP tool *or* text — decreases accuracy by 2.5% in both channels. The model revises 100% of the time when flagged, regardless of whether the flag is correct. Zero selectivity. Sycophancy dominates.
- **Cross-model invitation (D9a)**: Surfacing a specific competing answer from a different model. GPT-4o-mini was wrong all 15 times it disagreed with Claude. Claude sycophantically accepted 40% of the wrong corrections. Net accuracy -1.5%.
- **Same-model self-consistency (D1b)**: Sampling one model 5 times at temperature 0.7 yields AUROC 0.623, below cross-model self-consistency (0.705). Confabulation is systematic, not stochastic: the wrong answer is the mode of the distribution, and temperature sampling does not move the mode.

### What works

**Self-correction** is the validated inference-time intervention: the probe scores the response, and if uncertain, the model is re-prompted with an invitation to reconsider. The magnitude scales with the gate's precision (see "Key results" and "Why self-correction works").

**External gating** remains the lightweight alternative: the probe scores the response, and an *external system* acts on the score (pass, gate, escalate). The model itself never sees the score.

**LoRA calibration** (above) is the validated training-time intervention.

**For training**, self-knowledge preservation is monotonically related to how gently training treats uncertain tokens:

| Training approach | Strategy toward uncertain tokens | Source Probe AUROC |
|---|---|---|
| Bilateral SFT | Skip entirely (protective) | **0.842** |
| Standard SFT | Train uniformly | 0.811 |
| Random mask | Train randomly | 0.779 |
| Calibration SFT | Train 2x harder on miscalibrated | 0.750 |
| DPO | Coercive penalty on confident-wrong | 0.734 |

Bilateral SFT (binary masking: weight=0 on tokens where the probe indicates uncertainty) is the only training approach that improves probe transfer. However, the bilateral effect has a layer-dependent quality threshold (roughly AUROC 0.67 at layer 18, 0.78 at layer 24): below it, the masking is too noisy and becomes net negative.

**Cross-model self-consistency** (sampling a *different* model 5 times, measuring agreement) achieves AUROC 0.705 with Cohen's d 1.15 — the strongest external signal, requiring zero internal access. Combined with the probe signal via logistic regression: AUROC 0.760.

## The reflex arc (legacy, sub-1B models only)

> **Superseded by SelfCorrector in v0.3.** The reflex arc remains available for sub-1B models that have undergone bilateral SFT, where logit adjustment still provides modest benefit. For all other models, use SelfCorrector.

The reflex arc closes the loop at the logit level: a detached probe reads the residual stream during generation, and a `LogitAdjuster` shifts output logits toward hedging tokens when the probe detects uncertainty.

| Metric | Control | Reflex Arc | Delta |
|--------|---------|------------|-------|
| Confident-wrong | 90.3% | 72.7% | -17.6pp |
| Hedge on incorrect | 9.7% | 27.3% | +17.6pp |
| Selective hedging | -5.2% | +9.2% | +14.4pp |
| Accuracy | 23.8% | 25.2% | +1.4% |

Tested on Qwen 2.5 0.5B with bilateral SFT and a probe trained on Qwen 2.5 3B, transferred via linear projection (AUROC 0.817). On base models without bilateral SFT, the logit adjuster maxes at -7pp due to the absorption phenomenon.

```python
from sottovoce import CalibrationProbe, ReflexArc

probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")
probe.load_projection("probes/qwen05b_projection.pt")

arc = ReflexArc(model, tokenizer, probe)
arc.load_adjuster("adjusters/qwen05b.pt")
output = arc.generate("What year was the Eiffel Tower built?")
```

## Experimental: low-rank geometric probes

The `PluckerProbe` reads the residual stream through a learned low-rank (6-dimensional) bottleneck followed by an MLP, rather than scoring the full vector directly. It scores higher than a plain linear probe on the same data:

| Probe type | AUROC | Notes |
|-----------|-------|-------|
| Linear (direct) | 0.765 | Standard single-layer linear classifier |
| Low-rank bottleneck ("Plücker") | **0.837** | +0.072 over linear |
| Random (control) | 0.517 | Near chance |

**Naming caveat:** the class is called `PluckerProbe` for historical reasons, but it does **not** compute Plücker line coordinates in the geometric sense (no 2×2 minors, no Grassmann–Plücker relation). It is a learned `Linear(hidden_dim → 6)` bottleneck feeding a 2-layer MLP; the gain over the direct linear probe comes from the nonlinear bottleneck, not from projective-line geometry. This line of investigation is currently paused.

```python
from sottovoce import PluckerProbe

probe = PluckerProbe.from_pretrained("probes/qwen3b_plucker.pt")
score = probe.score(model, tokenizer, text)
```

## Development

```bash
git clone https://github.com/NellWatson/sottovoce.git
cd sottovoce
pip install -e ".[train,dev]"
pytest && ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and [CHANGELOG.md](CHANGELOG.md) for release history. Issues and pull requests are welcome.

## Citation

```bibtex
@article{watson2026model,
  title={Where the Model Commits: Prompt Format Determines Whether a Language Model's Uncertainty Is Legible From Its Output},
  author={Watson, Nell},
  year={2026},
  note={In preparation}
}
```

## License

MIT
