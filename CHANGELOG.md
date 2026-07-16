# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-07-16

### Changed
- **The default (generation-time) probe was retrained and re-shipped with a documented, verified
  provenance.** The previous `residual_layer_24.pt` predated the measurement programme; its
  training-item set was undocumented, and its one end-to-end deployment test read only **0.74-0.77
  on chat** against the 0.85 table number, with the CI excluding 0.85. The replacement is trained
  raw-fed on all three prompt formats pooled (items 0-499 of the seed-42 TriviaQA `rc.nocontext`
  shuffle), architecture and RAW-activation contract matching `score()` exactly, and **verified on a
  held-out 200-item split** (indices 500-699): AUROC **0.837 [0.779, 0.890] on chat** (the 0.852
  table number is now inside the CI), **0.787 [0.722, 0.847] on raw**, correct polarity in every
  format. Few-shot stays ~chance (0.53) — use `EntropyGate(first_token_only=True)` there, as before.
  Procedure and numbers: `research/results/_published/v12_gen_time_probe_reship.json`; training
  script `modal_train_gen_time_probe.py`.
- `RELEASE_URL` now points at the **v0.3.2** release, which carries the new default probe alongside
  the unchanged input-time probe, alignment set, and cross-model projections. Pins to v0.3.0/v0.3.1
  keep their exact original bytes (the old checkpoint); upgrade to get the verified probe.

## [0.3.1] - 2026-07-16

### Verified
- **The released input-time artifact has been verified end-to-end through the public path**
  (`pip install` from GitHub → `load_base_probe(timing="input")` downloading the release asset →
  `score()` against a real Qwen 2.5 3B Instruct forward pass) on 200 **held-out** TriviaQA items:
  AUROC 0.738 / 0.778 / 0.791 (few-shot / raw / chat), each within the claimed number's CI, with
  well-spread scores and correct polarity in every format. Verification JSON:
  `research/results/_published/v8_deployment_verification.json` in the research repo.
- This tag exists chiefly so version pins work: `v0.3.0` was cut before
  `load_base_probe(timing="input")` landed, so pinning it misses the loader even though the asset
  hangs on the v0.3.0 release. Pin `v0.3.1` (or later) for the input-time probe.

### Added
- **The input-time probe now ships: `load_base_probe(timing="input")`.** New release asset
  `residual_layer_24_input_time.pt`. It reads the residual at the last **prompt** token, before
  generation, so you score the prompt alone. Measured raw-fed exactly as `score()` runs it
  (500 TriviaQA, Qwen 2.5 3B Instruct, question-grouped 5-fold OOF CV):

  | timing | few-shot | raw | chat | swing |
  |---|:--:|:--:|:--:|:--:|
  | generation-time (default) | 0.616 | 0.767 | 0.852 | **0.236** |
  | input-time (new) | **0.737** | 0.798 | 0.819 | **0.081** |

  Use it when prompt format varies or is few-shot, when you want to gate *before* paying for the
  generation, or when you want one fewer forward pass. Keep the default under a chat template
  (0.852 vs 0.819) and under adversarial context injection, where the ordering reverses (0.657 vs
  0.591) because the attack lives in the prompt and that is all the input-time probe reads. Under
  few-shot, note that free `EntropyGate(first_token_only=True)` (0.821) still beats both probes.

  Two caveats we would rather state than have you discover. **(1)** The shipped artifact is trained
  on all three prompt formats **pooled**, and that is load-bearing: a probe trained on a single
  format does *not* reliably transfer to another (train on raw, test on few-shot reads **0.597**,
  barely above chance). Pooling removes the dependence on transfer instead of hoping for it, and
  matches or beats per-format training on every format. If you train your own, train it on every
  format you serve. **(2)** The generation-time row above is *method-level*: it fits a
  `StandardScaler` inside its CV, which `score()` does not do, and that scaling is worth roughly 2
  to 6 points. The input-time row is the artifact itself, raw-fed. So the comparison is, if
  anything, flattering to the default probe.
- **`CalibrationProbe.timing`** records which activation a probe was trained to read, and
  **`SelfCorrector` now raises** if handed an input-time probe. Its two-pass loop scores the
  model's completed answer, so an input-time probe would be scored out of distribution: a
  mismatch that returns plausible numbers and fails silently. Better a loud error.
- **`EntropyGate(first_token_only=True)`** — read entropy at the first answer token instead
  of averaging across the answer. Use it for few-shot / completion prompts, where the model
  commits at that token (44.6%) and averaging *dilutes* the signal (0.821 → 0.707). The
  default stays answer-averaged, which is correct for chat templates, where the first token
  is preamble and reading it alone is worse than chance (0.380 → 0.696 aggregated). The rule
  is: **read where the model commits.**

### Changed
- **Corrected: the headline AUROC described a probe this package does not ship.** The
  previously advertised 0.84 (and "the probe swings only 0.07 across prompt formats") came
  from an **input-time** probe reading the last *prompt* token. Sottovoce scores the model's
  *completed answer* — a **generation-time** probe. Measured across all three formats
  (`research/results/probe_timing_format_sweep/`, 500 TriviaQA, honest 5-fold OOF CV):

  | Probe timing | few-shot | raw | chat | swing |
  |---|:--:|:--:|:--:|:--:|
  | **generation-time (ships)** | **0.616** | 0.767 | **0.852** | **0.236** |
  | input-time (research only) | 0.793 | 0.804 | 0.836 | **0.043** |

  Consequences, all now reflected in the README: the shipped probe is **not** format-robust;
  under **few-shot it loses decisively to free first-token entropy** (0.616 vs 0.821), so use
  `EntropyGate(first_token_only=True)` there; under **chat it wins clearly** (0.852 vs 0.696),
  which is most deployments. Shipping an input-time probe is under consideration — it ties
  under chat, is far better under few-shot, needs one fewer forward pass, and can gate before
  generation — but it is *worse under adversarial injection*, so it is a trade-off, not a
  free win. It also requires retraining the released `.pt` on prompt-only features.
- **Adversarial robustness: measured, no longer an open question — and the answer is mixed.**
  Under per-question misleading-context injection (chat, 500 items): **both entropy measures
  collapse to chance** (first-token 0.446, aggregated 0.455, CIs spanning 0.50 — no usable
  signal), while **the probe degrades but survives** (0.852 → 0.657) and flags 12–25pp fewer
  wrong answers as confident (all CIs excluding zero). That is the strongest argument for the
  trained probe. **But it is a mitigation, not a defence:** 70% of wrong answers under attack
  still read as confident (entropy: 89–95%). Do not deploy either as a security control.
- **Corrected: self-correction's size tracks how *often* the gate fires, not its precision.**
  The README previously said the reduction scales with gate precision and that "sharper
  gating" pushes it higher. Its own source (AQ15) shows the opposite: correcting everything
  gave 96.2%, a gate firing on 88% of items gave 92.4% (its 88% rate is an acknowledged
  *calibration mismatch*, i.e. overfiring, not sharpness), and the most selective gate — 2.7%
  trigger — gave 2%. Also now stated plainly: **the mechanism is hedging, not correction.**
  Of 51 wrong→hedge flips in that battery, **zero were wrong→correct**, at a −2.7pp accuracy
  cost. Self-correction makes the model honest about what it does not know; it does not make
  it know.
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
