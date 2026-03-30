"""
sottovoce: Your model already knows. This reads what it can't say
— and now, acts on it.

Confabulation detection and self-correction from the residual stream. A
lightweight probe reads uncertainty; the self-corrector acts on it.

Detection (v0.1):
    probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")
    score = probe.score(model, tokenizer, text)
    decision = probe.decide(score)  # external routing

Self-correction (v0.3):
    corrector = SelfCorrector(model, tokenizer, probe)
    result = corrector.generate("What year was the Eiffel Tower built?")
    # With CUDA bf16 probe (AUROC 0.989): CW 62.7% -> 9.3% (85% reduction)
    # With standard MLP probe (AUROC 0.84): CW ~10% reduction

The self-corrector is the primary intervention path. It generates a response,
probes the residual stream, and if the probe detects uncertainty, re-prompts
the model with an invitation to reconsider. The model revises selectively.

The reflex arc (v0.2) remains available as a prosthetic for sub-1B models
with bilateral SFT, where logit adjustment still provides modest benefit.
For all other models, use SelfCorrector.

Watson, N. (2026). "The Model Already Knows: Cross-Architecture
Uncertainty Signals in Language Model Residual Streams."
"""

from sottovoce.probe import CalibrationProbe, ProbeConfig, ProbeDecision
from sottovoce.alignment import load_alignment_set
from sottovoce.selfcorrect import SelfCorrector, SelfCorrectionResult, SelfCorrectorConfig
from sottovoce.reflex import ReflexArc, LogitAdjuster
from sottovoce.plucker import PluckerProbe

__version__ = "0.3.0"
__all__ = [
    "CalibrationProbe", "ProbeConfig", "ProbeDecision",
    "load_alignment_set",
    "SelfCorrector", "SelfCorrectionResult", "SelfCorrectorConfig",
    "ReflexArc", "LogitAdjuster",
    "PluckerProbe",
]
