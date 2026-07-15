"""
Zero-training gate: read uncertainty from the model's own output distribution.

``EntropyGate`` is a drop-in alternative to :class:`~sottovoce.CalibrationProbe`. It needs
no training, no probe checkpoint and no cross-model projection: it reads the Shannon entropy
of the model's next-token distribution across the tokens of its own answer.

WHEN TO USE THIS INSTEAD OF THE PROBE
-------------------------------------
Measured head-to-head. 500 TriviaQA questions, Qwen 2.5 3B Instruct, every signal computed
from the same forward pass, the probe scored by honest 5-fold out-of-fold cross-validation:

    prompt format    commits at    entropy,      entropy,      CalibrationProbe
                      1st token    1st token    over answer    (as shipped)
    few-shot            44.6%        0.821         0.707            0.616     <- ENTROPY WINS
    raw instruction      7.0%        0.621         0.545            0.767     <- probe
    chat template        3.0%        0.380         0.696            0.852     <- probe

Read where the model commits. That one rule explains the whole table, in both directions:

* Under FEW-SHOT the model commits to its answer at the first generated token (44.6%), so
  that token is where the information is. Averaging over the rest of the answer DILUTES it
  (0.821 -> 0.707). Pass ``first_token_only=True``.
* Under a CHAT TEMPLATE it has committed to nothing at the first token -- it spends that
  token on preamble ("The", a newline) -- so first-token entropy measures FORMATTING, not
  knowledge, and lands below chance (0.380). Averaging across the answer RESCUES it
  (-> 0.696). That is this class's default, and it is the single most important
  implementation detail here.

Rules of thumb:

* Few-shot or completion-style prompts: use this gate with ``first_token_only=True``. It
  BEATS the shipped probe (0.821 vs 0.616) and costs nothing to train.
* Chat-template prompts (most deployments): use ``CalibrationProbe``. It wins (0.852 vs
  0.696). First-token entropy swings 0.44 AUROC across prompt formats; the shipped probe
  swings 0.24. What the probe buys is accuracy under chat-style prompts -- NOT robustness
  to how you prompt. (An input-time probe, reading the last PROMPT token, swings only 0.04
  and is format-robust, but is not what this package currently ships.)

Two honest caveats:

* Output entropy is adversarially **gameable**, and worse than the probe here. Under injected
  context (chat, 500 items) entropy collapses to CHANCE -- first-token 0.446, answer-averaged
  0.455, both CIs spanning 0.50 -- while the probe degrades but survives (0.852 -> 0.657) and
  flags 12-25pp fewer wrong answers as confident. This is the strongest reason to prefer the
  probe. It is still not a defence: 70% of wrong answers under attack read as confident
  (entropy: 89-95%). Deploy neither as a security control.
* Elaborate "factual vs expressive" token splits are not worth it. Averaging entropy over all
  answer tokens scores 0.752; a named-entity-based factual/expressive split scores 0.761. The
  +0.009 is noise. This class does the simple thing.

Watson, N. (in preparation). "Where the Model Commits: Prompt Format Determines
Whether a Language Model's Uncertainty Is Legible From Its Output."
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
        first_token_only: bool = False,
    ):
        """
        Args:
            first_token_only: read entropy at the **first answer token only**, instead of
                averaging across the answer. Set this for few-shot / completion-style
                prompts, where the model commits to its answer at that token (measured
                44.6% of the time) and averaging over the rest of the answer dilutes the
                signal: first-token 0.82 AUROC vs aggregated 0.71. Leave it False for
                chat-style prompts, where the first token is preamble and reading it alone
                is worse than chance (0.38 vs 0.70 aggregated). The rule is: read where the
                model commits. See `research/results/probe_timing_format_sweep/`.
        """
        self.config = config or ProbeConfig()
        self.answer_marker = answer_marker
        self.first_token_only = first_token_only
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

        If ``first_token_only`` was set, returns the entropy at the **first** answer token
        instead of the mean — the right choice for few-shot prompts (see ``__init__``).
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
            if self.first_token_only:
                break

        if not ents:
            return 0.0
        return float(ents[0]) if self.first_token_only else float(np.mean(ents))

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
