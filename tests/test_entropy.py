"""CPU-only tests for the zero-training EntropyGate. No model or network required.

The entropy measurement itself needs a language model, so what is tested here is the
contract: that EntropyGate is a drop-in Gate, routes like the probe, and calibrates in the
right direction (low answer entropy => high confidence).
"""

import numpy as np
import pytest

from sottovoce import CalibrationProbe, EntropyGate, Gate, ProbeDecision


def test_entropy_gate_is_a_gate():
    """EntropyGate must be drop-in wherever CalibrationProbe is accepted."""
    assert isinstance(EntropyGate(), Gate)
    assert isinstance(CalibrationProbe(), Gate)


def test_decide_routing_matches_probe():
    gate = EntropyGate()
    assert gate.decide(0.95) is ProbeDecision.PASS
    assert gate.decide(0.60) is ProbeDecision.HEDGE
    assert gate.decide(0.35) is ProbeDecision.GATE
    assert gate.decide(0.10) is ProbeDecision.ESCALATE


def test_confidence_is_monotone_decreasing_in_entropy():
    """More entropy must never mean more confidence."""
    gate = EntropyGate()
    scores = [gate._to_confidence(h) for h in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_uncalibrated_by_default():
    assert EntropyGate().is_calibrated is False


def test_calibrate_learns_the_right_direction():
    """Low answer entropy on correct answers => calibration must map it to high confidence."""
    gate = EntropyGate()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=400)
    # Correct answers come out low-entropy; wrong answers high-entropy.
    entropies = np.where(
        labels == 1, rng.normal(0.5, 0.3, 400), rng.normal(2.0, 0.6, 400)
    ).clip(0)

    acc = gate.calibrate(entropies, labels)
    assert gate.is_calibrated
    assert acc > 0.8

    confident = gate._to_confidence(0.5)   # low entropy
    unsure = gate._to_confidence(2.0)      # high entropy
    assert confident > 0.5 > unsure
    assert gate.decide(confident) in (ProbeDecision.PASS, ProbeDecision.HEDGE)
    assert gate.decide(unsure) in (ProbeDecision.GATE, ProbeDecision.ESCALATE)


def test_calibrate_rejects_degenerate_labels():
    gate = EntropyGate()
    with pytest.raises(ValueError):
        gate.calibrate([0.1, 0.2, 0.3], [1, 1, 1])  # single class
