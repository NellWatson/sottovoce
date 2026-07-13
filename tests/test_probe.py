"""CPU-only unit tests: no model or network required.

Covers the probe routing contract, MLP and JL save/load round-trips, the JL
probe's polarity and float32 handling, cross-model projection shapes, and the
base-probe registry. Run with `pytest`.
"""

import numpy as np
import pytest
import torch

from sottovoce import CalibrationProbe, ProbeDecision
from sottovoce.hub import BASE_PROBES, load_base_probe


def _synthetic(n=800, d=2048, seed=0):
    """Activations where 'correct' (label 1) examples are shifted +direction."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(d).astype(np.float32)
    direction /= np.linalg.norm(direction)
    labels = rng.integers(0, 2, size=n)
    signal = (labels * 2 - 1)[:, None] * direction[None, :] * 3.0
    acts = (rng.standard_normal((n, d)) + signal).astype(np.float32)
    return acts, labels


def test_decide_routing():
    probe = CalibrationProbe()  # defaults: pass .85, hedge .50, gate .30
    assert probe.decide(0.95) is ProbeDecision.PASS
    assert probe.decide(0.60) is ProbeDecision.HEDGE
    assert probe.decide(0.35) is ProbeDecision.GATE
    assert probe.decide(0.10) is ProbeDecision.ESCALATE


def test_mlp_probe_roundtrip(tmp_path):
    probe = CalibrationProbe()
    path = tmp_path / "probe.pt"
    probe.save(path)

    reloaded = CalibrationProbe.from_pretrained(path)
    x = torch.randn(1, 2048)
    original = probe._probe.predict_proba(x).item()
    restored = reloaded._probe.predict_proba(x).item()

    assert 0.0 <= original <= 1.0
    assert abs(original - restored) < 1e-6


def test_jl_probe_polarity_and_float32():
    acts, labels = _synthetic()
    jl = CalibrationProbe.from_jl_calibration(acts[:600], labels[:600], k=64, seed=42)

    # float32 activations must not raise (regression: R was silently float64).
    scores = jl._probe.predict_proba(
        torch.tensor(acts[600:], dtype=torch.float32)
    ).squeeze(-1).numpy()

    # Polarity: score is P(correct), so correct answers must score higher.
    te_labels = labels[600:]
    assert scores[te_labels == 1].mean() > scores[te_labels == 0].mean()


def test_jl_probe_save_load_roundtrip(tmp_path):
    acts, labels = _synthetic()
    jl = CalibrationProbe.from_jl_calibration(acts[:600], labels[:600], k=64, seed=42)
    path = tmp_path / "jl_probe.pt"
    jl.save(path)

    # from_pretrained must detect the JL checkpoint, not KeyError on 'net.0.weight'.
    reloaded = CalibrationProbe.from_pretrained(path)
    x = torch.tensor(acts[600:601], dtype=torch.float32)
    original = jl._probe.predict_proba(x).item()
    restored = reloaded._probe.predict_proba(x).item()
    assert abs(original - restored) < 1e-6


def test_train_projection_shape():
    probe = CalibrationProbe()  # source_dim 2048
    rng = np.random.default_rng(1)
    source = rng.standard_normal((64, 2048)).astype(np.float32)
    target = rng.standard_normal((64, 4096)).astype(np.float32)  # e.g. an 8B model

    probe.train_projection(source, target, n_epochs=5)
    projected = probe._projection(torch.tensor(target[:1]))
    assert projected.shape == (1, 2048)  # maps target_dim -> source_dim


def test_base_probe_registry():
    assert "qwen2.5-3b" in BASE_PROBES
    # Unknown model raises before any network access.
    with pytest.raises(ValueError):
        load_base_probe("nonexistent-model")
