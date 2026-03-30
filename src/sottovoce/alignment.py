"""
Pre-computed alignment sets for cross-model probe transfer.

Provides a fixed set of TriviaQA questions with pre-extracted Qwen 2.5 3B
residual stream features. Users only need to run their target model on
these questions, extract features, and call train_projection().

The "curated" set is geometrically diverse: questions are selected to span
the full probe uncertainty range (10 equal-width bins, 0.0-1.0), ensuring
the projection sees both confident and uncertain examples. This produces
better projections from fewer examples than random selection.

Usage:
    from sottovoce.alignment import load_alignment_set

    questions, source_features = load_alignment_set()
    target_features = probe.extract_features(target_model, target_tok, questions)
    probe.train_projection(source_features, target_features)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_DIR = Path(os.environ.get(
    "SOTTOVOCE_CACHE",
    Path.home() / ".cache" / "sottovoce",
))

RELEASE_URL = (
    "https://github.com/NellWatson/sottovoce/releases/download/v0.3.0"
)


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to dest path."""
    import urllib.request
    print(f"  Downloading {dest.name}...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def load_alignment_set(
    n: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> tuple[list[str], np.ndarray]:
    """
    Load the curated alignment set for cross-model transfer.

    Returns a set of TriviaQA questions with pre-extracted Qwen 2.5 3B
    layer-24 residual stream features. The questions are selected for
    geometric diversity across the probe's uncertainty range.

    Alignment set sizing (empirical):
        - 200 questions: sufficient for targets up to ~32B within-family
          or ~8B cross-family (hidden_dim up to ~5000)
        - 500 questions: recommended default for most use cases
        - 1000 questions: needed for 70B+ cross-family targets
          (hidden_dim 8192+)

    Args:
        n: Number of alignment questions to return. If None, returns
           the full curated set. If less than the full set, samples
           uniformly across uncertainty bins to preserve geometric
           diversity.
        cache_dir: Override cache directory (default: ~/.cache/sottovoce/)

    Returns:
        (questions, source_features): list of question strings and
        (N, 2048) numpy array of Qwen 3B layer-24 features.
    """
    cache = Path(cache_dir) if cache_dir else _ensure_cache_dir()

    questions_path = cache / "alignment_questions.json"
    features_path = cache / "alignment_features.npz"

    # Download if not cached
    if not questions_path.exists():
        _download_file(
            f"{RELEASE_URL}/alignment_questions.json", questions_path,
        )
    if not features_path.exists():
        _download_file(
            f"{RELEASE_URL}/alignment_features.npz", features_path,
        )

    with open(questions_path) as f:
        data = json.load(f)

    questions = data["questions"]
    bins = data.get("bins")  # probe score bin for each question
    features = np.load(str(features_path))["features"]

    if n is not None and n < len(questions):
        # Sample uniformly across bins to preserve geometric diversity
        if bins is not None:
            indices = _stratified_sample(bins, n)
        else:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(questions), size=n, replace=False)
            indices.sort()

        questions = [questions[i] for i in indices]
        features = features[indices]

    return questions, features


def _stratified_sample(bins: list[int], n: int) -> np.ndarray:
    """Sample uniformly across uncertainty bins."""
    rng = np.random.default_rng(42)
    unique_bins = sorted(set(bins))
    per_bin = n // len(unique_bins)
    remainder = n % len(unique_bins)

    indices = []
    for i, b in enumerate(unique_bins):
        bin_indices = [j for j, bb in enumerate(bins) if bb == b]
        take = per_bin + (1 if i < remainder else 0)
        take = min(take, len(bin_indices))
        chosen = rng.choice(bin_indices, size=take, replace=False)
        indices.extend(chosen.tolist())

    indices.sort()
    return np.array(indices)


def alignment_set_info() -> dict:
    """Return metadata about the bundled alignment set."""
    return {
        "source_model": "Qwen/Qwen2.5-3B-Instruct",
        "source_layer": 24,
        "source_hidden_dim": 2048,
        "dataset": "TriviaQA (rc, validation split)",
        "selection": "geometric-coverage (10 bins across probe uncertainty range)",
        "sizing_guide": {
            "up_to_32B_within_family": "200 questions",
            "up_to_8B_cross_family": "200 questions",
            "70B_cross_family": "1000 questions",
            "405B_cross_family": "2000-3000 questions (estimated)",
        },
    }
