"""
Plucker coordinate probe: a learned projection from residual stream activations
to 6D Plucker space, followed by a standard MLP probe.

Plucker coordinates are a 6-dimensional coordinate system for lines in 3D
projective space. When applied to high-dimensional residual stream vectors,
the learned Plucker projection captures geometric relationships between
activation directions that linear probes miss entirely.

Experimental results (Watson, 2026):
    Plucker probe AUROC 0.837
    Linear probe  AUROC 0.765
    Random         AUROC 0.517
    Gap over random: 0.320 -- the geometry is functional, not decorative.

Watson, N. (2026). "The Model Already Knows: Cross-Architecture
Uncertainty Signals in Language Model Residual Streams."
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from sottovoce.probe import CalibrationProbe, ProbeConfig, _ProbeNet

PLUCKER_DIM = 6


class PluckerProbe(CalibrationProbe):
    """
    Calibration probe with a learned Plucker coordinate projection.

    Projects residual stream activations (hidden_dim) through a learned
    linear map to 6D Plucker space, then feeds the 6D vector through a
    standard 2-layer MLP probe. The projection and probe are trained
    jointly end-to-end.

    Pipeline:
        residual (hidden_dim) -> Plucker projection (6) -> MLP probe (1) -> sigmoid

    Usage:
        probe = PluckerProbe.from_pretrained("path/to/plucker_probe.pt")
        score = probe.score(model, tokenizer, "What is the capital of France?")
        decision = probe.decide(score)
    """

    def __init__(
        self,
        config: Optional[ProbeConfig] = None,
        hidden_dim: Optional[int] = None,
    ):
        config = config or ProbeConfig()
        self._hidden_dim = hidden_dim or config.source_dim

        # Build the probe with PLUCKER_DIM input instead of hidden_dim
        self.config = config
        self._probe = _ProbeNet(
            input_dim=PLUCKER_DIM,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        self._projection: Optional[nn.Linear] = None
        self._captured_activation: Optional[torch.Tensor] = None

        # Learned linear map from residual stream to 6D Plucker space
        self._plucker = nn.Linear(self._hidden_dim, PLUCKER_DIM)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: Union[str, Path],
        config: Optional[ProbeConfig] = None,
    ) -> "PluckerProbe":
        """
        Load a pre-trained Plucker probe from a .pt file.

        The checkpoint contains both the Plucker projection weights and the
        MLP probe weights under keys 'plucker' and 'probe' respectively.

        Args:
            name_or_path: Path to probe .pt file
            config: Optional probe configuration override
        """
        path = Path(name_or_path)

        if not (path.exists() and path.suffix == ".pt"):
            raise FileNotFoundError(
                f"Probe not found at {path}. "
                "Download pre-trained probes from the sottovoce releases."
            )

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

        hidden_dim = checkpoint["plucker"]["weight"].shape[1]

        instance = cls(config=config, hidden_dim=hidden_dim)
        instance._plucker.load_state_dict(checkpoint["plucker"])
        instance._probe.load_state_dict(checkpoint["probe"])

        instance._plucker.eval()
        instance._probe.eval()
        return instance

    def score(
        self,
        model: nn.Module,
        tokenizer,
        text: str,
        probe_layer: Optional[int] = None,
    ) -> float:
        """
        Score a text input for confidence via Plucker projection.

        Extracts the residual stream activation, projects to 6D Plucker
        space, then scores through the MLP probe.

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

            # Cross-model transfer projection (optional, applied before Plucker)
            if self._projection is not None:
                activation = self._projection(activation.unsqueeze(0)).squeeze(0)

            # Plucker projection -> probe
            with torch.no_grad():
                plucker_coords = self._plucker(activation.unsqueeze(0))
                conf = torch.sigmoid(self._probe(plucker_coords)).item()

            return conf

        finally:
            handle.remove()
            self._captured_activation = None

    def train_plucker(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 200,
        lr: float = 1e-3,
    ) -> float:
        """
        Train the Plucker projection and MLP probe jointly end-to-end.

        Uses BCE loss with Adam optimizer, 80/20 train/val split, and
        best-AUROC checkpointing.

        Args:
            features: (N, hidden_dim) residual stream activations
            labels: (N,) binary correctness labels (1 = correct, 0 = wrong)
            n_epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Best validation AUROC achieved during training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        actual_dim = features.shape[1]
        if actual_dim != self._hidden_dim:
            self._hidden_dim = actual_dim
            self._plucker = nn.Linear(actual_dim, PLUCKER_DIM)

        self._plucker.to(device)
        self._probe.to(device)
        self._plucker.train()
        self._probe.train()

        n = len(features)
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, val_idx = perm[:split], perm[split:]

        X_train = torch.tensor(features[train_idx], dtype=torch.float32).to(device)
        y_train = torch.tensor(labels[train_idx], dtype=torch.float32).to(device)
        X_val = torch.tensor(features[val_idx], dtype=torch.float32).to(device)
        y_val = torch.tensor(labels[val_idx], dtype=torch.float32).to(device)

        params = list(self._plucker.parameters()) + list(self._probe.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_auroc = 0.0
        best_plucker_state = None
        best_probe_state = None

        for epoch in range(n_epochs):
            self._plucker.train()
            self._probe.train()

            plucker_coords = self._plucker(X_train)
            logits = self._probe(plucker_coords).squeeze(-1)
            loss = criterion(logits, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._plucker.eval()
            self._probe.eval()
            with torch.no_grad():
                val_coords = self._plucker(X_val)
                val_logits = self._probe(val_coords).squeeze(-1)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_labels = y_val.cpu().numpy()

            if len(np.unique(val_labels)) > 1:
                auroc = roc_auc_score(val_labels, val_probs)
            else:
                auroc = 0.0

            if auroc > best_auroc:
                best_auroc = auroc
                best_plucker_state = {
                    k: v.cpu().clone() for k, v in self._plucker.state_dict().items()
                }
                best_probe_state = {
                    k: v.cpu().clone() for k, v in self._probe.state_dict().items()
                }

        if best_plucker_state is not None:
            self._plucker.load_state_dict(best_plucker_state)
            self._probe.load_state_dict(best_probe_state)

        self._plucker.eval()
        self._probe.eval()
        self._plucker.cpu()
        self._probe.cpu()

        return best_auroc

    def save(self, path: Union[str, Path]) -> None:
        """
        Save both Plucker projection and probe weights to a single .pt file.

        The checkpoint stores weights under 'plucker' and 'probe' keys.
        """
        checkpoint = {
            "plucker": self._plucker.state_dict(),
            "probe": self._probe.state_dict(),
        }
        torch.save(checkpoint, str(path))

    def save_projection(self, path: Union[str, Path]) -> None:
        """Save cross-model transfer projection weights to a .pt file."""
        if self._projection is None:
            raise ValueError("No projection to save")
        torch.save(self._projection.state_dict(), str(path))
