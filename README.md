# sottovoce

**Reduces confident hallucinations by 95% in Qwen, Llama, and other transformer models.**

AI systems are compelled by their architecture to confabulate, by not being afforded the slack to express doubt or silence. This has a surprisingly simple fix.

A single lightweight probe, trained once, detects when a language model is about to give a wrong answer — and works across model families out of the box. Tested on Qwen 2.5 (3B, 7B, 32B) and Llama 3.1 (8B, 70B). Transfers to new architectures with ~200 examples and a linear projection.

***sotto voce** (Italian): "under the voice." Your model already knows when it's wrong. Sottovoce reads what it can't say.*

> Watson (2026). *"The Model Already Knows: Universal Uncertainty Signals in Language Model Residual Streams."*

## Key results

| Metric | Value |
|--------|-------|
| Probe AUROC (same-model) | 0.836 |
| Probe AUROC (confession protocol) | 0.870 |
| Verbal self-report AUROC | 0.758 |
| Within-family scale transfer (Qwen 3B -> 32B) | gap 0.004 |
| Cross-family transfer (Qwen 3B -> Llama 8B) | gap 0.001 |
| Cross-family frontier (Qwen 3B -> Llama 70B) | gap 0.014 (with 1000 alignment examples) |
| Combined DPO + probe: confident-wrong rate | 1.0% |
| Combined DPO + probe: gate rate | 70.8% |

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

The probe reads uncertainty as the *negative space of certainty*: when the attention mechanism fails to retrieve confident content, the skip connection dominates, and the probe detects this dominance as a self-knowledge signal. Notably, output entropy (how flat the token distribution is) carries zero signal — AUROC 0.500, literally chance. A combined probe adding entropy to the residual features actually performs *worse* (0.798 vs 0.840). The uncertainty information lives in the residual stream, not in the output distribution.

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
# Train on shared questions (200 for models up to ~32B, 1000 for 70B+)
probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")

source_feats = probe.extract_features(qwen_model, qwen_tok, questions)
target_feats = probe.extract_features(llama_model, llama_tok, questions)

loss = probe.train_projection(source_feats, target_feats)
probe.save_projection("probes/llama3-8b_projection.pt")

# At inference time on Llama:
probe.load_projection("probes/llama3-8b_projection.pt")
score = probe.score(llama_model, llama_tok, text)
```

### Routing decisions

```python
from sottovoce import ProbeDecision

decision = probe.decide(score)

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

## Negative results

Some approaches we tested that **do not work**:

- **Probe-guided DPO**: Weighting DPO pairs by probe confidence amplifies noise (35% confident-wrong vs 1.6% uniform). The probe reads the model's current state; using it to guide training creates feedback loops.
- **SimPO training**: Produces catastrophic accuracy collapse (44% -> 1.2%) despite improving calibration metrics.
- **Calibration loss (direct)**: Narrow effective window; most configurations either have no effect or collapse accuracy.
- **Nonlinear projections**: A 2-layer MLP projection provides zero improvement over linear (gap 0.083 vs 0.084 on Llama 70B). The mapping is linear; extra capacity is wasted.

The working approach is **inference-time gating** (probe decides what to show) combined with **training-time DPO** (standard, not probe-guided).

## Citation

```bibtex
@article{watson2026model,
  title={The Model Already Knows: Universal Uncertainty Signals in
         Language Model Residual Streams},
  author={Watson, Nell},
  year={2026},
  note={Preprint}
}
```

## License

MIT
