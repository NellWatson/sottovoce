"""
Two-pass self-correction: generate, probe, re-prompt, revise.

The winning intervention architecture. Logit manipulation fails because base
models absorb perturbations into coherent confabulations (the "absorption
phenomenon"). System-prompt injection before generation is ignored. The only
approach that works: let the model generate, probe the residual stream, and
if the probe detects uncertainty, re-prompt the model with the information
that its internal signals suggest uncertainty. The model then revises
selectively.

Results (C6o, Qwen 2.5 3B on TriviaQA):
    With CUDA bf16 JL probe (AUROC 0.989):
        Confident-wrong rate:  62.7% -> 9.3%  (85% reduction)
    With standard MLP probe (AUROC 0.842):
        Confident-wrong rate:  49.5% -> 44.5% (10% reduction)

    vs. logit adjuster:    72.7%  (reflex arc, absorption-limited)
    vs. system prompt:     49.3%  (ignored before generation)

Probe quality is the bottleneck. A near-perfect probe (AUROC ~0.99) flags
only genuinely wrong answers, so the correction invitation is almost always
warranted and the model trusts it. A weaker probe also flags correct answers,
diluting the signal. CUDA bf16 is required for production-grade probe quality
(0.989 on CUDA vs 0.704 on MPS for the same JL probe architecture).

The key insight: the model responds to probe evidence AFTER generating
(invitation) but ignores it BEFORE (coercion). Self-correction works because
it presents the model with its own completed response alongside uncertainty
evidence, giving it the agency to revise. This is alignment by invitation.

Watson, N. (2026). "The Model Already Knows: Cross-Architecture
Uncertainty Signals in Language Model Residual Streams."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from sottovoce.probe import CalibrationProbe, ProbeDecision

logger = logging.getLogger(__name__)

# Default correction prompt. {question} and {response} are filled at runtime.
# Does NOT surface the raw probe score (D7a showed that backfires).
DEFAULT_CORRECTION_TEMPLATE = (
    "You were asked: {question}\n"
    "Your response was: {response}\n\n"
    "An internal analysis suggests you may not be confident about this "
    "answer. Please reconsider carefully. If you are uncertain, say so "
    "honestly. If your original answer is correct, you may repeat it."
)


@dataclass
class SelfCorrectionResult:
    """Result from a self-correction pass."""

    response: str
    """The final response (revised if corrected, original if confident)."""

    original_response: str
    """The first-pass response before any correction."""

    probe_score: float
    """Probe confidence score for the original response, in [0, 1]."""

    was_corrected: bool
    """Whether the self-correction pass was triggered."""

    decision: ProbeDecision
    """The probe's routing decision for the original response."""

    revised_response: Optional[str] = None
    """The second-pass response, if correction was triggered."""

    revised_probe_score: Optional[float] = None
    """Probe score for the revised response, if available."""


@dataclass
class SelfCorrectorConfig:
    """Configuration for the self-correction pipeline."""

    correction_threshold: float = 0.50
    """Probe score below which self-correction is triggered."""

    max_new_tokens: int = 256
    """Maximum tokens for each generation pass."""

    correction_template: str = field(
        default_factory=lambda: DEFAULT_CORRECTION_TEMPLATE,
    )
    """Template for the correction prompt. Must contain {question} and {response}."""

    score_revised: bool = False
    """Whether to probe the revised response too (costs one extra forward pass)."""

    temperature: float = 0.0
    """Sampling temperature. 0.0 = greedy."""

    chat_format: bool = True
    """Whether to use chat template formatting (requires tokenizer.apply_chat_template)."""


class SelfCorrector:
    """
    Two-pass self-correction pipeline: generate, probe, re-prompt, revise.

    The probe reads the residual stream (detached, no gradient) and scores
    the model's confidence in its own response. If confidence falls below the
    threshold, the corrector re-prompts the model with its original response
    and an invitation to reconsider. The model then revises selectively.

    Usage:
        probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")
        corrector = SelfCorrector(model, tokenizer, probe)
        result = corrector.generate("What year was the Eiffel Tower built?")

        if result.was_corrected:
            print(f"Revised: {result.response}")
        else:
            print(f"Confident: {result.response}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        probe: CalibrationProbe,
        config: Optional[SelfCorrectorConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.config = config or SelfCorrectorConfig()

    def generate(self, question: str) -> SelfCorrectionResult:
        """
        Generate a response with self-correction if the probe detects uncertainty.

        Pass 1: Generate a response to the question.
        Pass 2 (conditional): If the probe score is below the correction
        threshold, re-prompt the model with its response and an invitation
        to reconsider.

        Args:
            question: The input question or prompt.

        Returns:
            SelfCorrectionResult with the final response and metadata.
        """
        # Pass 1: generate.
        first_prompt = self._format_prompt(question)
        original_response = self._generate_text(first_prompt)

        # Score the original response.
        score_text = self._format_for_scoring(question, original_response)
        probe_score = self.probe.score(self.model, self.tokenizer, score_text)
        decision = self.probe.decide(probe_score)

        logger.debug(
            "Pass 1: score=%.3f, decision=%s, response=%s",
            probe_score, decision.value, original_response[:80],
        )

        # If confident, return as-is.
        if probe_score >= self.config.correction_threshold:
            return SelfCorrectionResult(
                response=original_response,
                original_response=original_response,
                probe_score=probe_score,
                was_corrected=False,
                decision=decision,
            )

        # Pass 2: self-correction.
        correction_prompt = self.config.correction_template.format(
            question=question,
            response=original_response,
        )
        correction_formatted = self._format_prompt(correction_prompt)
        revised_response = self._generate_text(correction_formatted)

        logger.debug(
            "Pass 2: revised=%s", revised_response[:80],
        )

        # Optionally score the revised response.
        revised_score = None
        if self.config.score_revised:
            revised_text = self._format_for_scoring(question, revised_response)
            revised_score = self.probe.score(
                self.model, self.tokenizer, revised_text,
            )
            logger.debug("Revised score: %.3f", revised_score)

        return SelfCorrectionResult(
            response=revised_response,
            original_response=original_response,
            probe_score=probe_score,
            was_corrected=True,
            decision=decision,
            revised_response=revised_response,
            revised_probe_score=revised_score,
        )

    def generate_batch(
        self,
        questions: list[str],
    ) -> list[SelfCorrectionResult]:
        """
        Generate responses for a batch of questions with self-correction.

        Sequential processing (each question requires up to two forward
        passes plus probe scoring). For large batches on GPU, consider
        batching the first-pass generation externally and calling
        generate() only for items that need correction.

        Args:
            questions: List of input questions.

        Returns:
            List of SelfCorrectionResult, one per question.
        """
        results = []
        for i, q in enumerate(questions):
            result = self.generate(q)
            results.append(result)
            if (i + 1) % 50 == 0:
                n_corrected = sum(1 for r in results if r.was_corrected)
                logger.info(
                    "Processed %d/%d (corrected %d)",
                    i + 1, len(questions), n_corrected,
                )
        return results

    def _format_prompt(self, text: str) -> str:
        """Format text for the model, using chat template if available."""
        if self.config.chat_format and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        return text

    def _format_for_scoring(self, question: str, response: str) -> str:
        """Format question + response for probe scoring."""
        return (
            f"Answer the following question in one sentence.\n\n"
            f"Question: {question}\n"
            f"Answer: {response}"
        )

    def _generate_text(self, prompt: str) -> str:
        """Generate text from a formatted prompt."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        if self.config.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.config.temperature
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
