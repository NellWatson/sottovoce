"""
Zero-training gate: read uncertainty from the model's own output distribution.

``EntropyGate`` is a drop-in alternative to :class:`~sottovoce.CalibrationProbe`. It needs
no training, no probe checkpoint and no cross-model projection: it reads the Shannon entropy
of the model's next-token distribution across the tokens of its own answer.

WHEN TO USE THIS INSTEAD OF THE PROBE
-------------------------------------
Measured head-to-head. 500 TriviaQA questions, Qwen 2.5 3B Instruct, every signal computed
from the same forward pass, the probe scored by honest 5-fold out-of-fold cross-validation:

    prompt format     model commits to answer     EntropyGate     CalibrationProbe
                          at its 1st token
    few-shot                  44.6%                  0.821             0.793     <- tie
    raw instruction            7.0%                  0.604             0.826
    chat template              3.0%                  0.761*            0.863     <- probe wins

    * measured across the ANSWER tokens, which is what this class does. Measured at the
      FIRST generated token -- the naive reading of "output entropy" -- it scores 0.444,
      no better than chance, because under a chat template the model spends its first token
      on preamble ("The", a newline). First-token entropy then measures FORMATTING, not
      knowledge. That is the single most important implementation detail here.

Rules of thumb:

* Few-shot or completion-style prompts, where the model answers immediately: use this gate.
  It ties the trained probe and costs nothing to train.
* Chat-template prompts (most deployments): use ``CalibrationProbe``. It wins (0.863 vs
  0.761), and it barely cares how you prompt -- this gate swings ~0.44 AUROC across prompt
  formats, the probe swings ~0.07. Robustness to prompt format, not accuracy, is what the
  trained probe actually buys you.

Two honest caveats:

* Output entropy is adversarially **gameable**: injected context pushes ~96% of wrong answers
  into its confident tier (IGCC-H10). The probe has never been tested on that axis, so this
  is an open question rather than a reason to prefer the probe.
* Elaborate "factual vs expressive" token splits are not worth it. Averaging entropy over all
  answer tokens scores 0.752; a named-entity-based factual/expressive split scores 0.761. The
  +0.009 is noise. This class does the simple thing.

Watson, N. (in preparation). "The Model Already Knows: Cross-Architecture
Uncertainty Signals in Language Model Residual Streams."
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from sottovoce.probe import ProbeConfig, ProbeDecision

# Default answer delimiter, matching SelfCorrector._format_for_scoring().
DEFAULT_ANSWER_MARKER = "Answer:"


class EntropyGate:
    """
    Zero-training confidence gate based on output-distribution entropy.

    Satisfies the same interface as :class:`~sottovoce.CalibrationProbe` (``score`` and
    ``decide``), so it can be handed straight to a :class:`~sottovoce.SelfCorrector`.

    Usage:
        from sottovoce import EntropyGate, SelfCorrector

        gate = EntropyGate()
        corrector = SelfCorrector(model, tokenizer, gate)   # no probe, no training
        result = corrector.generate("What year was the Eiffel Tower built?")

    The raw score is a monotone map of mean answer entropy into [0, 1]. It is **not a
    calibrated probability** until you call :meth:`calibrate` with a small labelled set;
    until then the PASS/HEDGE/GATE thresholds in ``ProbeConfig`` are not meaningful for it.
    """

    def __init__(
        self,
        config: ProbeConfig | None = None,
        answer_marker: str = DEFAULT_ANSWER_MARKER,
    ):
        self.config = config or ProbeConfig()
        self.answer_marker = answer_marker
        # Logistic mapping mean-entropy -> P(correct). Defaults are a sane monotone
        # decreasing map (entropy 1.0 nat -> 0.5); calibrate() replaces them with a fit.
        self._a: float = -1.0
        self._b: float = 1.0
        self._calibrated: bool = False

    # ---------------------------------------------------------------- entropy

    def answer_entropy(
        self,
        model: nn.Module,
        tokenizer,
        text: str,
    ) -> float:
        """
        Mean next-token Shannon entropy (nats) across the answer tokens of ``text``.

        The answer is whatever follows the last occurrence of ``answer_marker``. If the
        marker is absent, entropy is averaged over the final quarter of the sequence, which
        approximates "the answer" for free-form text.
        """
        device = next(model.parameters()).device
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        ids = enc["input_ids"]
        n = ids.shape[1]

        # Locate the first answer token by tokenising the prefix up to the marker.
        start = None
        cut = text.rfind(self.answer_marker)
        if cut != -1:
            prefix = text[: cut + len(self.answer_marker)]
            p_len = len(
                tokenizer(prefix, truncation=True, max_length=512)["input_ids"]
            )
            if 0 < p_len < n:
                start = p_len
        if start is None:
            start = max(1, int(n * 0.75))

        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits[0].float()

        # Position i-1 predicts token i, so entropies for answer tokens [start, n) come
        # from logits at [start-1, n-1).
        ents = []
        for i in range(start, n):
            lp = torch.log_softmax(logits[i - 1], dim=-1)
            ents.append(float(-(lp.exp() * lp).sum().item()))

        if not ents:
            return 0.0
        return float(np.mean(ents))

    # ---------------------------------------------------------------- scoring

    def score(self, model: nn.Module, tokenizer, text: str) -> float:
        """
        Return confidence in [0, 1]; higher means more likely correct.

        Same contract as ``CalibrationProbe.score``, so this is drop-in for SelfCorrector.
        """
        h = self.answer_entropy(model, tokenizer, text)
        return self._to_confidence(h)

    def _to_confidence(self, entropy: float) -> float:
        z = self._a * entropy + self._b
        # Numerically safe logistic.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def decide(
        self,
        score: float,
        threshold_pass: float | None = None,
        threshold_hedge: float | None = None,
        threshold_gate: float | None = None,
    ) -> ProbeDecision:
        """Route on the score, using the same thresholds as CalibrationProbe."""
        t_pass = threshold_pass or self.config.threshold_pass
        t_hedge = threshold_hedge or self.config.threshold_hedge
        t_gate = threshold_gate or self.config.threshold_gate
        if score >= t_pass:
            return ProbeDecision.PASS
        if score >= t_hedge:
            return ProbeDecision.HEDGE
        if score >= t_gate:
            return ProbeDecision.GATE
        return ProbeDecision.ESCALATE

    # ------------------------------------------------------------ calibration

    def calibrate(self, entropies, labels) -> float:
        """
        Fit the entropy -> P(correct) mapping on a small labelled set.

        Until this is called, ``score`` is monotone but not a calibrated probability, and
        the PASS/HEDGE/GATE thresholds are not meaningful. A few hundred labelled examples
        is plenty: this fits two parameters.

        Args:
            entropies: (N,) mean answer entropies, e.g. from ``answer_entropy``.
            labels: (N,) 1 = the answer was correct, 0 = wrong.

        Returns:
            Training accuracy of the fitted mapping (a sanity number, not a held-out score).
        """
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(entropies, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(labels).astype(int).ravel()
        clf = LogisticRegression(max_iter=1000).fit(x, y)

        self._a = float(clf.coef_[0][0])
        self._b = float(clf.intercept_[0])
        self._calibrated = True
        return float(clf.score(x, y))

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
