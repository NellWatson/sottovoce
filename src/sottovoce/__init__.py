"""
sottovoce: Your model already knows. This reads what it can't say
— and now, acts on it.

Confabulation detection and intervention from the residual stream. A
lightweight probe reads uncertainty; the reflex arc routes it to behavior.

Detection (v0.1):
    probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")
    score = probe.score(model, tokenizer, text)
    decision = probe.decide(score)  # external routing

Intervention (v0.2):
    arc = ReflexArc(model, tokenizer, probe)
    output = arc.generate("What year was the Eiffel Tower built?")
    # Hedges when uncertain, answers directly when confident

The reflex arc is a prosthetic for models below ~1B parameters that lack
native interoceptive output. It reduces confident-wrong responses by
17.6pp on Qwen 2.5 0.5B. Models above the threshold (3B+) already
self-calibrate and should not use the reflex arc.

Watson, N. (2026). "The Model Already Knows: Cross-Architecture
Uncertainty Signals in Language Model Residual Streams."

Watson, N. (2026). "The Reflex Arc: A Prosthetic Architecture for
Uncertainty-Awareness in Language Models."
"""

from sottovoce.probe import CalibrationProbe, ProbeConfig, ProbeDecision
from sottovoce.alignment import load_alignment_set
from sottovoce.reflex import ReflexArc, LogitAdjuster
from sottovoce.plucker import PluckerProbe

__version__ = "0.2.0"
__all__ = [
    "CalibrationProbe", "ProbeConfig", "ProbeDecision",
    "load_alignment_set",
    "ReflexArc", "LogitAdjuster",
    "PluckerProbe",
]
