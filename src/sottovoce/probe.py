"""
Core calibration probe: a 2-layer MLP that reads residual stream activations
and predicts answer correctness.

The probe reads uncertainty as the *negative space of certainty*: when the
attention mechanism at ~2/3 depth fails to retrieve confident content, the
skip connection dominates, and the probe reads this dominance as a self-
knowledge signal. The representation is convergent across model families.

Watson, N. (forthcoming). "The Model Already Knows: Cross-Architecture
Uncertainty Signals in Language Model Residual Streams."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class ProbeDecision(Enum):
    """Decision output from the probe threshold check."""
    PASS = "pass"           # High confidence: answer directly
    HEDGE = "hedge"         # Moderate confidence: add uncertainty marker
    GATE = "gate"           # Low confidence: block or retry
    ESCALATE = "escalate"   # Very low confidence: escalate to human/better model


@dataclass
class ProbeConfig:
    """Configuration for a calibration probe deployment."""
    model_family: str = "qwen"
    probe_layer_fraction: float = 0.67
    threshold_pass: float = 0.85
    threshold_hedge: float = 0.50
    threshold_gate: float = 0.30
    hidden_dim: int = 256
    dropout: float = 0.2
    source_dim: int = 2048  # Qwen 2.5 3B residual stream dimension

    def probe_layer(self, n_layers: int) -> int:
        """Compute the probe layer index from total layer count."""
        return int(n_layers * self.probe_layer_fraction)


class _ProbeNet(nn.Module):
    """2-layer MLP probe network."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class _JLLogisticNet(nn.Module):
    """JL-compressed logistic probe: random projection + logistic regression.

    Drop-in replacement for _ProbeNet in the Tier 1 (coarse) deployment.
    CV AUROC 0.638 at k=64 (vs 0.704 for full MLP), ~513KB, ~262K FLOPs.
    """

    def __init__(self, input_dim: int, k: int = 64, seed: int = 42):
        super().__init__()
        self.k = k
        self.input_dim = input_dim

        # Fixed random projection (not trainable).
        rng = np.random.RandomState(seed)
        R_np = rng.randn(k, input_dim).astype(np.float32) / np.sqrt(k)
        self.register_buffer("R", torch.from_numpy(R_np))

        # Logistic regression (trainable or loaded from sklearn).
        self.logistic = nn.Linear(k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x @ self.R.T  # (B, k)
        return self.logistic(projected)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    @classmethod
    def from_sklearn(
        cls,
        input_dim: int,
        k: int,
        seed: int,
        coef: np.ndarray,
        intercept: float,
    ) -> "_JLLogisticNet":
        """Load weights from a trained sklearn LogisticRegression."""
        net = cls(input_dim=input_dim, k=k, seed=seed)
        with torch.no_grad():
            net.logistic.weight.copy_(torch.from_numpy(coef.astype(np.float32)))
            net.logistic.bias.fill_(intercept)
        net.eval()
        return net

    @classmethod
    def from_calibration(
        cls,
        activations: np.ndarray,
        labels: np.ndarray,
        k: int = 64,
        seed: int = 42,
    ) -> "_JLLogisticNet":
        """Train from raw activations and correctness labels."""
        from sklearn.linear_model import LogisticRegression

        input_dim = activations.shape[1]
        rng = np.random.RandomState(seed)
        R = rng.randn(k, input_dim).astype(np.float32) / np.sqrt(k)
        X_proj = activations @ R.T

        confab = 1 - labels
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
        clf.fit(X_proj, confab)

        return cls.from_sklearn(
            input_dim=input_dim,
            k=k,
            seed=seed,
            coef=clf.coef_,
            intercept=float(clf.intercept_[0]),
        )


class CalibrationProbe:
    """
    Universal calibration probe for confabulation detection.

    Reads residual stream activations at ~2/3 model depth and predicts
    whether the model's answer is correct. Transfers across model families
    via lightweight linear projection.

    Usage:
        probe = CalibrationProbe.from_pretrained("path/to/probe.pt")
        score = probe.score(model, tokenizer, "What is the capital of France?")
        decision = probe.decide(score)
    """

    def __init__(self, config: Optional[ProbeConfig] = None):
        self.config = config or ProbeConfig()
        self._probe = _ProbeNet(
            input_dim=self.config.source_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        self._projection: Optional[nn.Linear] = None
        self._captured_activation: Optional[torch.Tensor] = None

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: Union[str, Path],
        config: Optional[ProbeConfig] = None,
    ) -> "CalibrationProbe":
        """
        Load a pre-trained probe from a .pt file.

        Args:
            name_or_path: Path to probe .pt file
            config: Optional probe configuration override
        """
        instance = cls(config=config)
        path = Path(name_or_path)

        if path.exists() and path.suffix == ".pt":
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            input_dim = state["net.0.weight"].shape[1]
            instance.config.source_dim = input_dim
            instance._probe = _ProbeNet(
                input_dim=input_dim,
                hidden_dim=instance.config.hidden_dim,
                dropout=instance.config.dropout,
            )
            instance._probe.load_state_dict(state)
        else:
            raise FileNotFoundError(
                f"Probe not found at {path}. "
                "Download pre-trained probes from the sottovoce releases."
            )

        instance._probe.eval()
        return instance

    def load_projection(self, path: Union[str, Path]) -> None:
        """
        Load a linear projection for cross-model transfer.

        The projection maps from the target model's hidden dimension
        to the source probe's dimension.
        """
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        target_dim = state["weight"].shape[1]
        self._projection = nn.Linear(target_dim, self.config.source_dim, bias=True)
        self._projection.load_state_dict(state)
        self._projection.eval()

    def train_projection(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """
        Train a linear projection from target model features to source space.

        Alignment set sizing: 200 examples suffice for target hidden dims
        up to ~5000 (e.g. 7B-32B models). For frontier models with hidden
        dim 8192+ (e.g. 70B), use 1000+ examples.

        Args:
            source_features: (N, source_dim) from the probe's source model
            target_features: (N, target_dim) from the target model
            n_epochs: Training epochs
            lr: Learning rate

        Returns:
            Final MSE loss
        """
        target_dim = target_features.shape[1]
        self._projection = nn.Linear(target_dim, self.config.source_dim, bias=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._projection.to(device)

        X = torch.tensor(target_features, dtype=torch.float32).to(device)
        Y = torch.tensor(source_features, dtype=torch.float32).to(device)

        optimizer = torch.optim.Adam(self._projection.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            projected = self._projection(X)
            loss = criterion(projected, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._projection.eval()
        return loss.item()

    def _hook_fn(self, module, input, output):
        """Forward hook to capture residual stream activation."""
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured_activation = hidden[0, -1, :].detach().cpu().float()

    def score(
        self,
        model: nn.Module,
        tokenizer,
        text: str,
        probe_layer: Optional[int] = None,
    ) -> float:
        """
        Score a text input for confidence.

        Returns a float in [0, 1] where higher means more confident the
        answer is correct. Lower scores indicate higher confabulation risk.

        Args:
            model: The language model (must have model.model.layers)
            tokenizer: The tokenizer
            text: Input text to score
            probe_layer: Layer index to probe (default: auto from config)

        Returns:
            Confidence score in [0, 1]
        """
        if probe_layer is None:
            n_layers = len(model.model.layers)
            probe_layer = self.config.probe_layer(n_layers)

        handle = model.model.layers[probe_layer].register_forward_hook(self._hook_fn)

        try:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs, use_cache=False)

            activation = self._captured_activation
            if activation is None:
                raise RuntimeError("Hook failed to capture activation")

            if self._projection is not None:
                activation = self._projection(activation.unsqueeze(0)).squeeze(0)

            score = self._probe.predict_proba(activation.unsqueeze(0)).item()
            return score

        finally:
            handle.remove()
            self._captured_activation = None

    def decide(
        self,
        score: float,
        threshold_pass: Optional[float] = None,
        threshold_hedge: Optional[float] = None,
        threshold_gate: Optional[float] = None,
    ) -> ProbeDecision:
        """
        Make a routing decision based on the probe score.

        Args:
            score: Confidence score from self.score()
            threshold_pass: Override for PASS threshold
            threshold_hedge: Override for HEDGE threshold
            threshold_gate: Override for GATE threshold

        Returns:
            ProbeDecision indicating recommended action
        """
        t_pass = threshold_pass or self.config.threshold_pass
        t_hedge = threshold_hedge or self.config.threshold_hedge
        t_gate = threshold_gate or self.config.threshold_gate

        if score >= t_pass:
            return ProbeDecision.PASS
        elif score >= t_hedge:
            return ProbeDecision.HEDGE
        elif score >= t_gate:
            return ProbeDecision.GATE
        else:
            return ProbeDecision.ESCALATE

    def extract_features(
        self,
        model: nn.Module,
        tokenizer,
        texts: list[str],
        probe_layer: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract residual stream features for a list of texts.

        Args:
            model: The language model
            tokenizer: The tokenizer
            texts: List of input texts
            probe_layer: Layer to extract from

        Returns:
            (N, hidden_dim) numpy array of features
        """
        if probe_layer is None:
            n_layers = len(model.model.layers)
            probe_layer = self.config.probe_layer(n_layers)

        features = []
        handle = model.model.layers[probe_layer].register_forward_hook(self._hook_fn)

        try:
            device = next(model.parameters()).device
            for text in texts:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    model(**inputs, use_cache=False)

                if self._captured_activation is not None:
                    features.append(self._captured_activation.numpy())
                    self._captured_activation = None
        finally:
            handle.remove()

        return np.stack(features)

    def save(self, path: Union[str, Path]) -> None:
        """Save probe weights to a .pt file."""
        torch.save(self._probe.state_dict(), str(path))

    def save_projection(self, path: Union[str, Path]) -> None:
        """Save projection weights to a .pt file."""
        if self._projection is None:
            raise ValueError("No projection to save")
        torch.save(self._projection.state_dict(), str(path))

    @classmethod
    def from_jl_calibration(
        cls,
        activations: np.ndarray,
        labels: np.ndarray,
        k: int = 64,
        seed: int = 42,
        config: Optional[ProbeConfig] = None,
    ) -> "CalibrationProbe":
        """Create a Tier 1 JL-compressed logistic probe from calibration data.

        The probe uses a fixed random projection (JL lemma) followed by
        logistic regression. CV AUROC 0.638 at k=64 (C6i).

        Args:
            activations: (N, d) residual stream activations.
            labels: (N,) correctness labels (1=correct, 0=wrong).
            k: JL projection dimension (default 64, the production sweet spot).
            seed: Random seed for projection matrix.
            config: Optional probe configuration.

        Returns:
            CalibrationProbe with JL logistic probe as its internal network.
        """
        instance = cls(config=config)
        instance._probe = _JLLogisticNet.from_calibration(
            activations=activations,
            labels=labels,
            k=k,
            seed=seed,
        )
        # JL probe handles its own projection; disable the external one.
        instance._projection = None
        return instance
