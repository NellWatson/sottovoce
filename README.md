# sottovoce

**Detects language model confabulation from the residual stream with AUROC 0.836, transferring across architectures via a linear projection.**

AI systems are compelled by their architecture to confabulate, by not being afforded the slack to express doubt or silence. The model's own residual stream already encodes whether it is right or wrong; the signal just never reaches the output. Sottovoce reads it.

A single lightweight probe, trained once, detects when a language model has given a wrong answer — and works across model families out of the box. Tested on Qwen 2.5 (3B, 7B, 32B) and Llama 3.1 (8B, 70B). Transfers to new architectures with ~200 examples and a linear projection.

***sotto voce** (Italian): "under the voice." Your model already knows when it's wrong. Sottovoce reads what it can't say.*

> Watson, N. (forthcoming). *"The Model Already Knows: Cross-Architecture Uncertainty Signals in Language Model Residual Streams."*

## Key results

| Metric | Value |
|--------|-------|
| Probe AUROC (same-model) | 0.836 |
| Probe AUROC (confession protocol) | 0.870 |
| Verbal self-report AUROC | 0.758 |
| Behavioral hedging AUROC | 0.413 (anti-predictive) |
| Within-family scale transfer (Qwen 3B -> 32B) | gap 0.004 |
| Cross-family transfer (Qwen 3B -> Llama 8B) | gap 0.001 |
| Cross-family frontier (Qwen 3B -> Llama 70B) | gap 0.014 (with 1000 alignment examples) |
| Inference-time gating: confident-wrong rate | 1.0% (at 70.8% gate rate) |
| Cross-model self-consistency AUROC | 0.705 |

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

### Score a single response

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sottovoce import CalibrationProbe

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

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

### Routing decisions

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

### `CalibrationProbe`

| Method | Description |
|--------|-------------|
| `from_pretrained(path)` | Load probe weights from `.pt` file |
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
Input text
    |
Language Model (frozen)
    | hook at layer floor(0.67 x depth)
Residual stream vector (hidden_dim)
    | optional: linear projection (for cross-model)
Source-space vector (source_dim)
    |
MLP probe: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    |
Confidence score in [0, 1]
    |
Threshold -> ProbeDecision
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

The practical consequence: sottovoce's probe score must drive *external* gating decisions. Surfacing the score to the model — by any channel — is counterproductive until models develop metacognitive selectivity (the capacity to evaluate correction signals rather than defer to them unconditionally).

## Negative results

Some approaches we tested that **do not work**:

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

**Inference-time gating** remains the validated approach: the probe scores the response, and an *external system* acts on the score (pass, gate, escalate). The model itself never sees the score.

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
