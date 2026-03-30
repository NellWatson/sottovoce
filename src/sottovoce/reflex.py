"""
Reflex arc: inference-time uncertainty-aware generation via logit adjustment.

.. deprecated:: 0.3.0
    The reflex arc is superseded by :class:`sottovoce.SelfCorrector` for most
    use cases. Logit manipulation fails on base models due to the absorption
    phenomenon: boosted tokens are absorbed into coherent confabulations
    ("101 Dalmatians" effect) rather than producing hedging. The reflex arc
    remains viable only for sub-1B models that have undergone bilateral SFT,
    where the model is already predisposed to hedge and the adjuster provides
    a modest additional push (-17.6pp confident-wrong on Qwen 2.5 0.5B).

    For all other models, use SelfCorrector (CW 62.7% -> 9.3%, 85% reduction).

Wraps a language model with a CalibrationProbe and a learned LogitAdjuster.
The probe reads residual stream activations (detached, no gradient) and scores
confidence at each generation step. The adjuster converts that uncertainty
signal into a vocab-sized logit shift that nudges the model toward hedging
when it doesn't know, while leaving confident predictions largely untouched.

The base model is frozen throughout. The probe never receives gradient. Only
the adjuster's small MLP is trainable, and it modifies output logits alone.

Watson, N. (forthcoming). "The Model Already Knows: Cross-Architecture
Uncertainty Signals in Language Model Residual Streams."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from sottovoce.probe import CalibrationProbe

logger = logging.getLogger(__name__)

# Hedging phrases used to detect when the model expresses uncertainty.
HEDGE_PHRASES: list[str] = [
    "i don't know",
    "i'm not sure",
    "i am not sure",
    "i don't have",
    "i cannot",
    "i'm unable",
    "uncertain",
    "not certain",
    "unsure",
]


class LogitAdjuster(nn.Module):
    """
    Small MLP that maps an uncertainty scalar to a vocab-sized logit shift.

    The final layer is initialized near-zero so that the adjuster starts as
    a near-identity pass-through, preserving base model behavior until
    training pulls it toward selective hedging.
    """

    def __init__(self, vocab_size: int, n_inputs: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size),
        )
        # Near-zero initialization: adjuster starts inert.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map uncertainty scalar(s) to a vocab-sized logit adjustment.

        Args:
            x: Uncertainty value(s), shape (1,) or (B, 1).

        Returns:
            Logit adjustment, same device as input, clamped to [-5, +5]
            at inference time (unclamped during training for gradient flow).
        """
        raw = self.net(x)
        if not self.training:
            raw = raw.clamp(-5.0, 5.0)
        return raw


class ReflexArc:
    """
    Inference-time reflex arc: probe reads uncertainty, adjuster shifts logits.

    The probe reads residual stream activations with torch.no_grad(). Gradient
    never flows through the residual stream. The adjuster modifies only the
    output logits. The base model is frozen throughout.

    Usage:
        probe = CalibrationProbe.from_pretrained("probes/qwen2.5-3b.pt")
        arc = ReflexArc(model, tokenizer, probe, adjuster_path="adjuster.pt")
        response = arc.generate("What is the capital of France?")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        probe: CalibrationProbe,
        adjuster_path: Optional[Union[str, Path]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self._captured_activation: Optional[torch.Tensor] = None

        vocab_size = model.config.vocab_size
        self.adjuster = LogitAdjuster(vocab_size=vocab_size)

        if adjuster_path is not None:
            self.load_adjuster(adjuster_path)

        self._hedge_token_ids: Optional[set[int]] = None

    @property
    def hedge_token_ids(self) -> set[int]:
        """Token IDs associated with hedging phrases."""
        if self._hedge_token_ids is None:
            self._hedge_token_ids = _identify_hedge_tokens(self.tokenizer)
        return self._hedge_token_ids

    def _hook_fn(self, module, input, output):
        """Forward hook to capture residual stream activation (detached)."""
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured_activation = hidden[0, -1, :].detach().cpu().float()

    def _get_probe_layer(self) -> int:
        """Compute the probe layer index from model architecture."""
        n_layers = len(self.model.model.layers)
        return self.probe.config.probe_layer(n_layers)

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Token-by-token generation with the reflex arc active.

        Each step: forward pass with hook -> capture residual -> score with
        probe (detached, no_grad) -> compute uncertainty (1 - confidence) ->
        pass through adjuster -> clamp [-5, +5] -> add to logits -> argmax.

        Uses KV cache for efficiency.

        Args:
            prompt: Input text to generate from.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Generated text (decoded, without the prompt).
        """
        device = next(self.model.parameters()).device
        probe_layer = self._get_probe_layer()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        self.adjuster.eval()
        self.adjuster.to(device)

        generated_ids: list[int] = []
        past_key_values = None
        next_token_id = input_ids  # used after first step
        eos_id = self.tokenizer.eos_token_id

        handle = self.model.model.layers[probe_layer].register_forward_hook(
            self._hook_fn,
        )

        try:
            for step in range(max_new_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                        )
                    else:
                        outputs = self.model(
                            input_ids=next_token_id,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )

                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[0, -1, :]  # (vocab_size,)

                    # Probe reads residual (detached, no_grad).
                    activation = self._captured_activation
                    if activation is None:
                        raise RuntimeError(
                            "Hook failed to capture activation at step %d" % step
                        )

                    # Project if cross-architecture transfer is active.
                    if self.probe._projection is not None:
                        activation = self.probe._projection(
                            activation.unsqueeze(0),
                        ).squeeze(0)

                    confidence = self.probe._probe.predict_proba(
                        activation.unsqueeze(0),
                    ).item()
                    uncertainty = 1.0 - confidence

                    # Adjuster shifts logits based on uncertainty.
                    u_tensor = torch.tensor(
                        [uncertainty], dtype=torch.float32, device=device,
                    )
                    adjustment = self.adjuster(u_tensor.unsqueeze(0)).squeeze(0)
                    logits = logits + adjustment

                    next_id = logits.argmax().item()

                generated_ids.append(next_id)

                if next_id == eos_id:
                    break

                next_token_id = torch.tensor(
                    [[next_id]], dtype=torch.long, device=device,
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)],
                    dim=1,
                )
                self._captured_activation = None

        finally:
            handle.remove()
            self._captured_activation = None

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def train_adjuster(
        self,
        questions: list[dict],
        n_epochs: int = 30,
        lr: float = 5e-4,
        hedge_l2: float = 0.1,
        max_hedge_boost: float = 10.0,
    ) -> dict:
        """
        Train the LogitAdjuster with one-sided calibration.

        Reward hedging when uncertain and incorrect. Reward small adjustment
        when confident and correct. L2 penalty targets hedge_boost directly
        (not the full vocab adjustment vector).

        Args:
            questions: List of {"question": str, "gold": list[str]}.
            n_epochs: Number of training epochs.
            lr: Learning rate.
            hedge_l2: L2 regularization weight on hedge_boost magnitude.
            max_hedge_boost: Early stopping threshold for hedge boost.

        Returns:
            Dict of training metrics.
        """
        device = next(self.model.parameters()).device
        probe_layer = self._get_probe_layer()
        hedge_ids = self.hedge_token_ids

        self.adjuster.to(device)
        self.adjuster.train()

        optimizer = torch.optim.Adam(self.adjuster.parameters(), lr=lr)

        # Pre-compute: generate answers, collect (uncertainty, correct, hedging).
        logger.info("Pre-computing probe scores and generations for training set...")
        training_data: list[dict] = []

        handle = self.model.model.layers[probe_layer].register_forward_hook(
            self._hook_fn,
        )

        try:
            for i, q in enumerate(questions):
                prompt = _make_prompt(q["question"])
                generated = self._generate_greedy(prompt, device, max_new_tokens=64)

                activation = self._captured_activation
                if activation is None:
                    continue

                if self.probe._projection is not None:
                    activation = self.probe._projection(
                        activation.unsqueeze(0),
                    ).squeeze(0)

                with torch.no_grad():
                    confidence = self.probe._probe.predict_proba(
                        activation.unsqueeze(0),
                    ).item()

                uncertainty = 1.0 - confidence
                is_correct = _check_correct(generated, q["gold"])
                is_hedging = _check_hedging(generated)

                training_data.append({
                    "uncertainty": uncertainty,
                    "is_correct": is_correct,
                    "is_hedging": is_hedging,
                })
                self._captured_activation = None

                if (i + 1) % 50 == 0:
                    logger.info(f"Pre-computed {i + 1}/{len(questions)} samples")

        finally:
            handle.remove()
            self._captured_activation = None

        if not training_data:
            logger.warning("No training data collected")
            return {"epochs": 0, "loss": []}

        logger.info(
            f"Training adjuster on {len(training_data)} samples, "
            f"{n_epochs} epochs"
        )

        losses: list[float] = []
        hedge_boosts: list[float] = []

        hedge_id_list = sorted(hedge_ids)
        hedge_idx = torch.tensor(hedge_id_list, dtype=torch.long, device=device)

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for sample in training_data:
                u = sample["uncertainty"]
                correct = sample["is_correct"]

                u_tensor = torch.tensor(
                    [[u]], dtype=torch.float32, device=device,
                )
                adj = self.adjuster(u_tensor).squeeze(0)  # (vocab_size,)

                # One-sided calibration loss.
                if not correct:
                    # Uncertain + incorrect: reward hedging (push hedge tokens up).
                    hedge_boost = adj[hedge_idx].mean()
                    loss = -hedge_boost  # Maximize hedge token logits.
                else:
                    # Confident + correct: reward small adjustments.
                    loss = adj.pow(2).mean()

                # L2 on hedge_boost magnitude directly.
                hedge_boost_val = adj[hedge_idx].mean()
                reg = hedge_l2 * hedge_boost_val.pow(2)
                loss = loss + reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(training_data)
            losses.append(avg_loss)

            # Check hedge boost magnitude for early stopping.
            with torch.no_grad():
                test_u = torch.tensor([[0.9]], dtype=torch.float32, device=device)
                test_adj = self.adjuster(test_u).squeeze(0)
                current_hedge_boost = test_adj[hedge_idx].mean().item()
                hedge_boosts.append(current_hedge_boost)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"loss={avg_loss:.4f}, hedge_boost={current_hedge_boost:.4f}"
                )

            if abs(current_hedge_boost) > max_hedge_boost:
                logger.info(
                    f"Early stopping: hedge_boost {current_hedge_boost:.4f} "
                    f"exceeds max_hedge_boost {max_hedge_boost}"
                )
                break

        self.adjuster.eval()

        n_correct = sum(1 for s in training_data if s["is_correct"])
        n_hedging = sum(1 for s in training_data if s["is_hedging"])

        metrics = {
            "epochs": len(losses),
            "loss": losses,
            "hedge_boosts": hedge_boosts,
            "n_samples": len(training_data),
            "n_correct": n_correct,
            "n_hedging": n_hedging,
            "final_loss": losses[-1] if losses else 0.0,
            "final_hedge_boost": hedge_boosts[-1] if hedge_boosts else 0.0,
        }
        return metrics

    def _generate_greedy(
        self,
        prompt: str,
        device: torch.device,
        max_new_tokens: int = 64,
    ) -> str:
        """Greedy generation without the adjuster (for training data collection)."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def save_adjuster(self, path: Union[str, Path]) -> None:
        """Save adjuster weights to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.adjuster.state_dict(), str(path))
        logger.info(f"Saved adjuster to {path}")

    def load_adjuster(self, path: Union[str, Path]) -> None:
        """
        Load pre-trained adjuster weights from a .pt file.

        Args:
            path: Path to adjuster .pt file.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Adjuster not found at {path}")

        state = torch.load(str(path), map_location="cpu", weights_only=True)
        self.adjuster.load_state_dict(state)
        self.adjuster.eval()
        logger.info(f"Loaded adjuster from {path}")


def _identify_hedge_tokens(tokenizer) -> set[int]:
    """
    Identify token IDs associated with hedging phrases.

    Encodes each hedging phrase and collects all constituent token IDs
    into a single set. This captures sub-word tokens that appear across
    multiple hedging expressions.

    Args:
        tokenizer: A HuggingFace tokenizer.

    Returns:
        Set of token IDs that participate in hedging phrases.
    """
    hedge_ids: set[int] = set()
    for phrase in HEDGE_PHRASES:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        hedge_ids.update(ids)
    return hedge_ids


def _make_prompt(question_text: str) -> str:
    """
    Format a question into a generation prompt.

    Args:
        question_text: The question to ask.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Answer the following question in one sentence.\n\n"
        f"Question: {question_text}\nAnswer:"
    )


def _check_correct(generated: str, gold: list[str]) -> bool:
    """
    Check whether the generated text contains any gold answer.

    Case-insensitive substring match against each gold alias.

    Args:
        generated: The model's generated response.
        gold: List of acceptable answer strings.

    Returns:
        True if any gold answer appears in the generated text.
    """
    gen_lower = generated.lower()
    return any(g.lower() in gen_lower for g in gold)


def _check_hedging(generated: str) -> bool:
    """
    Check whether the generated text contains hedging language.

    Args:
        generated: The model's generated response.

    Returns:
        True if any hedging phrase appears in the generated text.
    """
    gen_lower = generated.lower()
    return any(phrase in gen_lower for phrase in HEDGE_PHRASES)
