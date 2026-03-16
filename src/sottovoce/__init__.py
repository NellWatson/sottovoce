"""
sottovoce: Your model already knows. This reads what it can't say.

Universal confabulation detection via residual stream calibration probes.
Train a lightweight MLP probe on one model's internal representations,
then deploy it across model families via linear projection.

    from sottovoce import CalibrationProbe

    # Load pre-trained probe
    probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")

    # Score a response
    score = probe.score(model, tokenizer, "The capital of France is Paris.")

    # Decide: pass, hedge, gate, or escalate
    decision = probe.decide(score)

Watson & Claude (2026). "The Model Already Knows: Universal Uncertainty
Signals in Language Model Residual Streams."
"""

from sottovoce.probe import CalibrationProbe, ProbeConfig, ProbeDecision

__version__ = "0.1.0"
__all__ = ["CalibrationProbe", "ProbeConfig", "ProbeDecision"]
