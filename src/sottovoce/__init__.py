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
    # Reduction scales with probe precision: ~10% CW reduction reproduced
    # with a standard MLP probe (AUROC ~0.84); higher with geometry-gated gating.

The self-corrector is the primary intervention path. It generates a response,
probes the residual stream, and if the probe detects uncertainty, re-prompts
the model with an invitation to reconsider. The model revises selectively.

The reflex arc (v0.2) remains available as a prosthetic for sub-1B models
with bilateral SFT, where logit adjustment still provides modest benefit.
For all other models, use SelfCorrector.

Watson, N. (2026). "Where the Model Commits: Prompt Format Determines Whether a
Language Model's Uncertainty Is Legible From Its Output."
"""

from sottovoce.alignment import load_alignment_set
from sottovoce.entropy import EntropyGate
from sottovoce.hub import load_base_probe
from sottovoce.plucker import PluckerProbe
from sottovoce.probe import CalibrationProbe, Gate, ProbeConfig, ProbeDecision
from sottovoce.reflex import LogitAdjuster, ReflexArc
from sottovoce.selfcorrect import SelfCorrectionResult, SelfCorrector, SelfCorrectorConfig

__version__ = "0.3.0"
__all__ = [
    "CalibrationProbe", "ProbeConfig", "ProbeDecision", "Gate",
    "EntropyGate",
    "load_base_probe", "load_alignment_set",
    "SelfCorrector", "SelfCorrectionResult", "SelfCorrectorConfig",
    "ReflexArc", "LogitAdjuster",
    "PluckerProbe",
]
