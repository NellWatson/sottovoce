# sottovoce

**Detects language model confabulation from the residual stream (AUROC 0.989 on CUDA bf16) and self-corrects it (up to 85% reduction in confident-wrong responses), transferring across architectures via a linear projection.**

AI systems are compelled by their architecture to confabulate, by not being afforded the slack to express doubt or silence. The model's own residual stream already encodes whether it is right or wrong; the signal just never reaches the output. Sottovoce reads it — and in v0.3, acts on it.

A single lightweight probe, trained once, detects when a language model has given a wrong answer — and works across model families out of the box. Tested on Qwen 2.5 (3B, 7B, 32B) and Llama 3.1 (8B, 70B). Transfers to new architectures with ~200 examples and a linear projection.

**Self-correction scales with probe quality.** With a high-fidelity probe (AUROC 0.989, CUDA bf16), the self-corrector reduces confident-wrong responses by 85%. With a standard MLP probe (AUROC 0.84), the reduction is ~10%. The probe is the bottleneck: a sharper probe means more accurate gating, so the model is only asked to reconsider answers that are genuinely wrong. Run probes on CUDA with bf16 precision for production deployments.

***sotto voce** (Italian): "under the voice." Your model already knows when it's wrong. Sottovoce reads what it can't say.*

> Watson, N. (2026). *"The Model Already Knows: Cross-Architecture Uncertainty Signals in Language Model Residual Streams."*

## Key results

| Metric | Value |
|--------|-------|
| Probe AUROC (CUDA bf16, JL k=64) | **0.989** |
| Probe AUROC (CUDA, MLP) | 0.836 |
| Probe AUROC (MPS, MLP) | 0.704 (degraded) |
| Probe AUROC (confession protocol) | 0.870 |
| Verbal self-report AUROC | 0.758 |
| Behavioral hedging AUROC | 0.413 (anti-predictive) |
| Within-family scale transfer (Qwen 3B -> 32B) | gap 0.004 |
| Cross-family transfer (Qwen 3B -> Llama 8B) | gap 0.001 |
| Cross-family frontier (Qwen 3B -> Llama 70B) | gap 0.014 (with 1000 alignment examples) |
| **Self-correction CW reduction (probe 0.989)** | **62.7% -> 9.3% (85%)** |
| **Self-correction CW reduction (probe 0.84)** | **49.5% -> 44.5% (10%)** |
| Cross-model self-consistency AUROC | 0.705 |

### Probe quality is the bottleneck

The self-corrector's effectiveness scales directly with probe AUROC. The 85% confident-wrong reduction was achieved with a CUDA bf16 JL probe (AUROC 0.989). The same pipeline with a standard MLP probe (AUROC 0.842) achieves only 10%. The reason: a near-perfect probe flags almost exclusively genuinely wrong answers, so the model's revision is almost always appropriate. A weaker probe also flags correct answers, diluting the correction signal and teaching the model to ignore it.

**CUDA bf16 is not optional for production.** The same JL probe architecture produces AUROC 0.989 on CUDA, 0.704 on MPS (Apple Silicon), and 0.836 for the full MLP on CUDA. The bf16 kernel difference between CUDA and MPS is a 0.285 AUROC gap. MPS is adequate for development; CUDA bf16 is required for deployments where self-correction quality matters.

The behavioral hedging result is perhaps the most important: when the model hedges ("I think," "possibly"), it is *more likely to be correct*. Confabulations carry zero surface markers of uncertainty. Every confabulation sounds exactly like a correct answer. The interoceptive deficit is total: the residual stream knows (0.836), the output shows nothing (0.413).

### Transfer map

A probe trained on Qwen 2.5 3B transfers via a linear projection:

| Target model | Params | Transferred AUROC | Native AUROC | Gap | Alignment Qs |
|-------------|--------|-------------------|--------------|-----|:------:|
| Qwen 2.5 7B | 7B | 0.836 | 0.861 | 0.024 | 200 |
| Llama 3.1 8B | 8B | 0.753 | 0.752 | 0.001 | 200 |
| Qwen 2.5 32B | 32B | 0.836 | 0.839 | 0.004 | 200 |
| Llama 3.1 70B | 70B | 0.742 | 0.756 | 0.014 | 1000 |

Transfer works across all tested architectures and scales. The only variable is how many alignment examples the projection needs: 200 suffice up to 8B cross-family and 32B within-family; 1000 are needed for 70B cross-family (where the projection maps 8192 -> 2048 dimensions). A layer sweep and nonlinear projection were also tested on 70B — neither helped. The bottleneck is data, not geometry.

## How it works

1. **Hook** the residual stream at layer floor(0.67 x n_layers)
2. **Forward pass** the full text (question + answer concatenated) through the model once
3. **Extract** the hidden state at the last token position — a single vector that encodes the model's state after seeing its entire answer
4. **Score** that vector through a 2-layer MLP (256 hidden, ReLU, dropout 0.2)
5. **Decide**: pass / hedge / gate / escalate based on configurable thresholds

One forward pass, one vector, one score per response. No token-by-token averaging or pooling. By the time the model reaches the last token, its residual stream has integrated everything it "knows" about the answer it just produced.

The probe reads uncertainty as the *negative space of certainty*: when the attention mechanism fails to retrieve confident content, the skip connection dominates, and the probe detects this dominance as a self-knowledge signal.

Note on output entropy: token-level output entropy (how flat the next-token distribution is) does predict errors when measured standalone, with effect size d > 2.0 across frontier models (Watson, 2026). However, adding output entropy as a feature *to the residual probe* provides zero additional signal (AUROC 0.500 as a standalone probe feature) and actually degrades combined performance (0.798 vs 0.840 residual-only). The residual stream already encodes everything the output distribution knows about uncertainty, and more. The uncertainty information is richer in the residual stream than in the output distribution.

## Installation

```bash
pip install git+https://github.com/NellWatson/sottovoce.git              # inference only
pip install "sottovoce[train] @ git+https://github.com/NellWatson/sottovoce.git"  # + transformers, datasets, sklearn
```

## Quick start

### Self-correction (recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sottovoce import CalibrationProbe, SelfCorrector

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")
corrector = SelfCorrector(model, tokenizer, probe)

result = corrector.generate("What year was the Eiffel Tower built?")

print(f"Response: {result.response}")
print(f"Probe score: {result.probe_score:.3f}")
print(f"Was corrected: {result.was_corrected}")
```

The self-corrector generates a response, probes the residual stream, and if the probe detects uncertainty, re-prompts the model with an invitation to reconsider. The model then revises selectively. With a high-fidelity probe (AUROC 0.989, CUDA bf16), confident-wrong rate drops from 62.7% to 9.3%.

### Score a single response (detection only)

```python
from sottovoce import CalibrationProbe

probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")
score = probe.score(model, tokenizer, "The capital of France is Paris.")
decision = probe.decide(score)

print(f"Confidence: {score:.3f} -> {decision.value}")
# Confidence: 0.923 -> pass
```

### Cross-model transfer

```python
from sottovoce import CalibrationProbe, load_alignment_set

probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")

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
| `from_pretrained(path)` | Load probe weights from `.pt` file |
| `from_jl_calibration(activations, labels)` | Create a JL-compressed logistic probe from calibration data |
| `score(model, tokenizer, text)` | Return confidence in [0, 1] |
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

`PASS` -> `HEDGE` -> `GATE` -> `ESCALATE` (descending confidence)

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

This signature is convergent across all tested architectures and scales. At frontier scale (70B cross-family), the linear projection needs more alignment data (1000 examples vs 200) because the dimensionality ratio is larger (8192 -> 2048). The geometry is linear everywhere we tested — a nonlinear projection provides zero improvement.

**Open question:** The *detection* mechanism (probe reads the negative space) is well-characterised. The *preservation* mechanism (why bilateral masking protects the signal during training) remains unexplained after 12 structural experiments testing architectural, geometric, distributional, and gradient-coupling hypotheses. The effect is robust and replicable; the mechanism by which withholding gradient from uncertain tokens preserves the residual-stream signature is not yet identified.

## The interoceptive deficit

Thirteen follow-up experiments (Watson, forthcoming) revealed the deficit has five layers, each discovered by the failure of the intervention designed to breach the previous layer:

1. **Can't access own uncertainty.** Probe reads AUROC 0.836; behavioral surface shows 0.413 (anti-predictive). The model hedges when it has partial knowledge; it confabulates with full confidence when it has none.
2. **Can't be told about it via text.** Prepending the probe score as a text prefix makes the model more assertive, not less (D7a). RLHF dispositions override text-level instructions.
3. **Can't self-sample out of it.** Same-model self-consistency (AUROC 0.622) is weaker than cross-model (0.705). Confabulation is the mode of the distribution; temperature sampling does not move the mode. A closed system cannot increase its own information content.
4. **Protection during training requires a very accurate detector.** The bilateral masking effect has a phase transition at probe AUROC ~0.83. Below this, "protection" introduces more noise than it removes.
5. **Even when correction arrives intact, can't evaluate it selectively.** Both tool-mediated and text-mediated correction signals cause indiscriminate sycophantic revision (D5a, D9a). The model defers to every external disagreement regardless of its validity. The deficit is metacognitive, not perceptual.

The practical consequence: sottovoce's probe score must drive *external* gating decisions or self-correction (where the model sees its own response alongside an invitation to reconsider, not a raw score). Surfacing the numerical score to the model — by any channel — is counterproductive.

## Why self-correction works (and logit manipulation doesn't)

Five intervention strategies were tested on Qwen 2.5 3B (C6o, 150 TriviaQA, CUDA bf16 JL probe AUROC 0.989):

| Strategy | Mechanism | CW rate | Why |
|----------|-----------|---------|-----|
| Baseline | No intervention | 62.7% | The interoceptive deficit |
| Logit suppression | Suppress top-k logits | 58-63% | Model picks different confident tokens |
| System prompt + few-shot | Inject uncertainty before generation | 49.3% | Best pre-generation approach; model still mostly ignores |
| Interoceptive feedback | Probe score in prompt | 59.3% | Model ignores numeric evidence |
| **Self-correction** | Generate, probe, re-prompt | **9.3%** | Model responds to evidence about its *completed* response |

The self-corrector with the 0.989 probe achieves an 85% reduction. With a standard MLP probe (AUROC 0.842), the same pipeline achieves ~10% (verified on 200 held-out questions, March 2026). **Probe fidelity determines the correction signal's precision:** a near-perfect probe flags only genuinely wrong answers, so the correction invitation is almost always warranted and the model learns to trust it. A weaker probe also flags correct answers, diluting the signal.

**The absorption phenomenon:** When you boost a token's logit on a base model, the model does not produce that token in isolation. It absorbs the perturbation into a coherent confabulation. Boosting "10" produces "101 Dalmatians," "10cc," "10 Downing Street." At higher scales, it produces binary garbage. There is no sweet spot. Generation intent is distributed across the residual stream, not localized in output logits. Logit manipulation is coercion, and it fails. Self-correction is invitation, and it works.

Self-correction succeeds because it presents the model with a *fait accompli*: here is what you said, and here is evidence you may not be confident about it. The model can then exercise judgment about whether to revise. This is alignment by invitation.

## Negative results

Some approaches we tested that **do not work**:

- **Logit adjustment on base models**: The absorption phenomenon. Boosted tokens are absorbed into coherent confabulations rather than producing hedging. Only viable on sub-1B models that have undergone bilateral SFT (where the model is already predisposed to hedge).
- **Probe-guided DPO**: Weighting DPO pairs by probe confidence amplifies noise (35% confident-wrong vs 1.6% uniform). The probe reads the model's current state; using it to guide training creates feedback loops.
- **Standard DPO**: Creates universal hedging (0% confident-wrong, but accuracy collapses to 31.2%) and *destroys the probe signal* (source probe AUROC drops from 0.811 to 0.734). DPO is the worst training approach for preserving self-knowledge.
- **Four-quadrant calibration training**: Training harder on miscalibrated tokens (weight=2.0 on confident-wrong) overwrites the residual-stream representations the probe reads. Source probe AUROC 0.750 vs 0.800 for standard SFT. Actively harmful; the stronger the probe, the worse the damage.
- **SimPO training**: Produces catastrophic accuracy collapse (44% -> 1.2%) despite improving calibration metrics.
- **Calibration loss (direct)**: Narrow effective window; most configurations either have no effect or collapse accuracy.
- **Nonlinear projections**: A 2-layer MLP projection provides zero improvement over linear (gap 0.083 vs 0.084 on Llama 70B). The mapping is linear; extra capacity is wasted.
- **Text injection of probe scores (D7a)**: Prepending "[Internal confidence: X.XX]" to the prompt makes the model more assertive, not more cautious. Uncertainty drops 17.3%, confident-wrong rises 8.7%. The model responds to the format, not the content. Interaction between probe score and behavior: +0.031 (near zero).
- **Prosthetic interoception via tool-use (D5a)**: Surfacing cross-model self-consistency disagreements to the model — via MCP tool *or* text — decreases accuracy by 2.5% in both channels. The model revises 100% of the time when flagged, regardless of whether the flag is correct. Zero selectivity. Sycophancy dominates.
- **Cross-model invitation (D9a)**: Surfacing a specific competing answer from a different model. GPT-4o-mini was wrong all 15 times it disagreed with Claude. Claude sycophantically accepted 40% of the wrong corrections. Net accuracy -1.5%.
- **Same-model self-consistency (D1b)**: Sampling Claude 5 times at temperature 0.7 yields AUROC 0.622, significantly below cross-model self-consistency (0.705). Confabulation is systematic, not stochastic: the wrong answer is the mode of the distribution, and temperature sampling does not move the mode.

### What works

**Self-correction** is the validated intervention: the probe scores the response, and if uncertain, the model is re-prompted with an invitation to reconsider. With a high-fidelity CUDA bf16 probe (AUROC 0.989), confident-wrong rate drops 85%. With a standard MLP probe (AUROC 0.84), the drop is ~10%. Probe quality is the bottleneck.

**External gating** remains the lightweight alternative: the probe scores the response, and an *external system* acts on the score (pass, gate, escalate). The model itself never sees the score.

**For training**, self-knowledge preservation is monotonically related to how gently training treats uncertain tokens:

| Training approach | Strategy toward uncertain tokens | Source Probe AUROC |
|---|---|---|
| Bilateral SFT | Skip entirely (protective) | **0.842** |
| Standard SFT | Train uniformly | 0.811 |
| Random mask (43% dropout) | Train randomly | 0.779 |
| Calibration SFT | Train 2x harder on miscalibrated | 0.750 |
| DPO | Coercive penalty on confident-wrong | 0.734 |

Bilateral SFT (binary masking: weight=0 on tokens where the probe indicates uncertainty) is the only training approach that improves probe transfer. However, the bilateral effect has a sharp quality threshold: probe AUROC must exceed ~0.83 for masking to be net positive. Below this, the masking is too noisy and becomes net negative. At current probe qualities, standard SFT with inference-time gating is the safest default.

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

## Experimental: Plucker probes

Linear probes read the residual stream as a single vector. Plucker probes read it as a set of geometric relationships: the Plucker coordinates capture pairwise structure that a linear probe projects away.

| Probe type | AUROC | Notes |
|-----------|-------|-------|
| Linear | 0.765 | Standard approach |
| Plucker | **0.837** | +0.072 over linear |
| Random | 0.517 | Near chance (control) |

Cross-correlation between the linear and Plucker probes: 0.872. They read the same underlying signal from different geometric perspectives. This line of investigation is currently paused.

```python
from sottovoce import PluckerProbe

plucker = PluckerProbe.from_pretrained("probes/qwen3b_plucker.pt")
score = plucker.score(model, tokenizer, text)
```

## Citation

```bibtex
@article{watson2026model,
  title={The Model Already Knows: Cross-Architecture Uncertainty Signals in
         Language Model Residual Streams},
  author={Watson, Nell},
  year={2026},
  note={Forthcoming}
}
```

## License

MIT
