# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
