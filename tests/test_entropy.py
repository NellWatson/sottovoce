"""CPU-only tests for the zero-training EntropyGate. No model or network required.

The entropy measurement itself needs a language model, so what is tested here is the
contract: that EntropyGate is a drop-in Gate, routes like the probe, and calibrates in the
right direction (low answer entropy => high confidence).
"""

import math

import numpy as np
import pytest
import torch

from sottovoce import CalibrationProbe, EntropyGate, Gate, ProbeDecision


class _FakeTokenizer:
    """Whitespace tokenizer: one id per word. Enough to exercise answer_entropy."""

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = list(range(1, len(text.split()) + 1))
        if max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


class _FakeModel:
    """Returns logits whose per-position entropy is controlled by `ents`.

    Position i-1 predicts token i, so to set the entropy seen at answer token `start`
    we shape the logits at row start-1.
    """

    def __init__(self, n_positions, vocab=4, sharp_at=None):
        self.n, self.vocab, self.sharp_at = n_positions, vocab, sharp_at

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids=None, use_cache=False, **kw):
        rows = []
        for i in range(self.n):
            # sharp row -> near-zero entropy; flat row -> near-max entropy (ln vocab).
            rows.append(
                torch.tensor([20.0] + [0.0] * (self.vocab - 1))
                if i == self.sharp_at
                else torch.zeros(self.vocab)
            )
        return type("O", (), {"logits": torch.stack(rows).unsqueeze(0)})()


def test_first_token_only_reads_only_the_first_answer_token():
    """first_token_only must read the commitment token, not the mean over the answer.

    The text has 7 whitespace tokens; "Answer:" ends at token 4, so the answer tokens are
    [4, 7) and their entropies come from logit rows [3, 6). Row 3 is made sharp (~0 nats)
    and rows 4-5 flat (~ln 4 = 1.386), so first-token (0.0) and mean (~0.924) must differ.
    """
    text = "Q: capital ? Answer: Paris is nice"
    tok = _FakeTokenizer()
    n = len(text.split())
    model = _FakeModel(n_positions=n, vocab=4, sharp_at=3)

    mean_gate = EntropyGate()
    first_gate = EntropyGate(first_token_only=True)

    h_mean = mean_gate.answer_entropy(model, tok, text)
    h_first = first_gate.answer_entropy(model, tok, text)

    assert h_first == pytest.approx(0.0, abs=1e-3), "first answer token is the sharp one"
    assert h_mean > h_first, "mean must be dragged up by the flat trailing tokens"
    assert h_mean == pytest.approx(np.mean([0.0] + [math.log(4)] * 2), abs=1e-3)


def test_first_token_only_defaults_off_and_is_a_gate():
    """Default stays answer-averaged (correct for chat); the flag is opt-in."""
    assert EntropyGate().first_token_only is False
    assert EntropyGate(first_token_only=True).first_token_only is True
    assert isinstance(EntropyGate(first_token_only=True), Gate)


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
