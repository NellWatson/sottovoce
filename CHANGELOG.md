# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Corrected: "compute the entropy first" was wrong as blanket advice.** A previous
  entry in this release told users output entropy is "cheaper, needs no training, and
  is at least as accurate," so they should prefer it. We then ran the head-to-head that
  should have preceded that claim (500 TriviaQA items, Qwen 2.5 3B Instruct, all signals
  from the same forward pass, probe scored with honest 5-fold out-of-fold CV). The result
  is conditional, and the condition matters:

  | Prompt format | Model commits at 1st token | Output entropy | Probe |
  |---|:--:|:--:|:--:|
  | Few-shot | 44.6% | **0.821** | 0.793 |
  | Raw instruction | 7.0% | 0.604 | 0.826 |
  | Chat template | 3.0% | **0.365** (worse than chance) | **0.854** |

  Output entropy ties the probe (+0.027, CI [−0.015, +0.070]) **only** when the prompt makes
  the model commit to its answer at the first generated token. Under a chat template the
  model emits preamble instead, so first-token entropy measures formatting rather than
  knowledge and inverts to worse than chance. **Entropy swings 0.46 AUROC across formats;
  the probe swings 0.06.** Since most deployments are chat-style, the earlier advice would
  have steered users to an anti-predictive signal. The README now states the condition.
  What a trained probe buys is robustness to prompt format, not accuracy.

- **The output distribution does carry the signal** (gold-token logprob 0.886), so the
  earlier claim that the uncertainty "never reaches the output" was false and has been
  removed. It reaches the logits and dies at the **argmax**. The interoceptive deficit is
  reframed accordingly: it is a deficit of **expression, not of representation**.
- **Probe AUROC restated from measurement.** An earlier note in this release claimed
  input-time probes "top out around 0.64–0.67." That was inherited from a single prior run
  and is **wrong**: measured with honest 5-fold out-of-fold CV on 500 items, the probe scores
  **0.793–0.863** depending on prompt format. The probe is stronger and far more
  format-stable than that note implied.
- **Documentation now matches the validated experimental record.** Earlier
  releases headlined a single-run probe AUROC of 0.989 (CUDA bf16) and an "85%
  reduction," and stated that CUDA bf16 was mandatory. Follow-up experiments
  found that the 0.989 figure was inflated by cross-platform distribution shift
  (a fresh cross-validated probe scores ≈0.67 input-time / ≈0.77 generation-time),
  and that the apparent CUDA-vs-MPS gap was an artifact, not a hardware effect.
  The README and docstrings now lead with the reproduced numbers: a held-out
  probe AUROC of ≈0.84, and a self-correction reduction that **scales with probe
  precision** (≈10% with a standard probe, higher when the gate is sharpened).
  What actually drives quality is *where* the probe reads (generation-time vs.
  input-time), not the compute platform.
- The `PluckerProbe` docstrings now describe it accurately as a learned low-rank
  (6-dimensional) nonlinear bottleneck probe. It does **not** compute Plücker
  line coordinates; the name is retained only for continuity with the
  experiment logs.

### Added
- **`EntropyGate` — a zero-training, pluggable alternative to the probe.** The measurements
  showed output entropy *ties* the trained probe when the prompt makes the model answer
  immediately (0.821 vs 0.793), so the honest thing is to ship both and let users choose
  rather than to sell one. Any object with `score()` and `decide()` now satisfies the new
  `Gate` protocol, and `SelfCorrector` accepts either. `EntropyGate` measures entropy across
  the **answer tokens**, not the first generated token: that distinction is the difference
  between 0.761 and 0.444 under a chat template.
- `load_base_probe()` — one-line download-and-load for the pre-trained base
  probe shipped with each release. No manual file handling required.
- New `sottovoce.hub` module centralising release-asset download/caching.
- `examples/self_correct.py` — minimal zero-config self-correction example.
- `CONTRIBUTING.md`, `CHANGELOG.md`, and `CITATION.cff`.

### Fixed
- **JL probe routing polarity.** `CalibrationProbe.from_jl_calibration` fit on
  `1 - labels`, so a JL probe returned P(wrong) into a `decide()` that treats a
  high score as PASS — inverted routing. It now fits on `labels`, consistent
  with the MLP probe and the documented "higher = more likely correct" contract.
- **JL probe dtype crash.** The random projection matrix was silently promoted
  to float64 (`float32_array / np.sqrt(k)`), causing a dtype-mismatch error when
  scoring the float32 activations `score()` extracts. The projection is now kept
  in float32.

## [0.3.0] — 2026-03-30

### Added
- `SelfCorrector`: two-pass self-correction (generate → probe → re-prompt to
  reconsider). The validated inference-time intervention; supersedes the reflex
  arc for all models above ~1B parameters.
- Finding: self-correction effectiveness scales with probe precision.

## [0.2.0]

### Added
- `ReflexArc` + `LogitAdjuster`: logit-level uncertainty-aware generation for
  sub-1B models with bilateral SFT. (Deprecated in 0.3.0 in favour of
  `SelfCorrector` for larger models.)
- `PluckerProbe`: experimental low-rank bottleneck probe.

## [0.1.0]

### Added
- `CalibrationProbe`: residual-stream confabulation detector (2-layer MLP at
  ~2/3 model depth).
- Cross-model transfer via a linear projection (`train_projection` /
  `load_projection`) and a curated alignment set (`load_alignment_set`).
- `ProbeDecision` routing (PASS / HEDGE / GATE / ESCALATE) for external gating.
- Training CLI: `python -m sottovoce.train`.

[Unreleased]: https://github.com/NellWatson/sottovoce/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/NellWatson/sottovoce/releases/tag/v0.3.0
[0.2.0]: https://github.com/NellWatson/sottovoce/releases/tag/v0.2.0
[0.1.0]: https://github.com/NellWatson/sottovoce/releases/tag/v0.1.0
