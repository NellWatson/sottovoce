"""
Release-asset hub: cache directory, download helpers, and one-line loaders for
the assets shipped with each sottovoce GitHub release.

Two loaders are provided:

- :func:`load_base_probe` — download and load a pre-trained base probe.
- :func:`load_alignment_set` (re-exported from :mod:`sottovoce.alignment`) —
  download the curated cross-model alignment set.

Assets are cached under ``~/.cache/sottovoce`` (override with the
``SOTTOVOCE_CACHE`` environment variable).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from sottovoce.probe import CalibrationProbe

CACHE_DIR = Path(os.environ.get(
    "SOTTOVOCE_CACHE",
    Path.home() / ".cache" / "sottovoce",
))

RELEASE_URL = (
    "https://github.com/NellWatson/sottovoce/releases/download/v0.3.0"
)

# Pre-trained base probes shipped as release assets, keyed by source model.
# Each is a 2-layer MLP CalibrationProbe over the model's residual stream at
# ~2/3 depth. The Qwen 2.5 3B probe (residual_layer_24.pt) probes layer 24
# (hidden_dim 2048) and is the transfer source for every other architecture.
#
# Two TIMINGS are shipped, and the choice matters more than any other setting:
#
#   generation-time  scores prompt + the model's completed answer (the default,
#                    and what SelfCorrector's two-pass loop needs).
#   input-time       scores the prompt alone, before the model generates.
#
# Measured across three prompt formats (500 TriviaQA, Qwen 2.5 3B Instruct,
# question-grouped 5-fold out-of-fold CV). Two measurement regimes, kept apart
# because mixing them would flatter the artifacts:
#
#   timing                          few-shot   raw    chat   swing
#   generation-time  (method*)        0.616   0.767  0.852   0.236
#   input-time       (method*)        0.793   0.804  0.836   0.043
#   input-time       (THIS ARTIFACT)  0.737   0.798  0.819   0.081
#
#   * method-level numbers fit a StandardScaler inside the CV. score() feeds the
#     RAW activation to the MLP, so a shipped artifact does not get that scaling;
#     it costs roughly 2 to 6 points. The artifact row is what you actually get.
#     The generation-time artifact has not been re-measured raw, so its row is
#     method-level and is, if anything, flattering to it.
#
# The ordering survives both regimes: input-time is ~3x more robust to how you
# prompt (swing 0.081 vs 0.236) and much better under few-shot, while
# generation-time is better under chat and under adversarial context injection
# (0.657 vs 0.591), because the attack lives in the prompt and that is all the
# input-time probe reads. Neither dominates. Pick per deployment.
#
# Under few-shot, note that free output entropy (EntropyGate with
# first_token_only=True, 0.821) still beats BOTH probes. Reach for a probe under
# chat-style prompts, or when you need attack resistance.
BASE_PROBES = {
    "qwen2.5-3b": "residual_layer_24.pt",
}

# Input-time probes. Trained on raw prompt-only activations POOLED ACROSS all
# three prompt formats, which is load-bearing: a probe trained on a single format
# does not reliably transfer to another (train-on-raw, test-on-few-shot reads
# 0.597, barely above chance). Pooled training removes the problem rather than
# relying on transfer, and matches or beats per-format training on every format.
# If you train your own input-time probe, train it on every format you serve.
INPUT_TIME_PROBES = {
    "qwen2.5-3b": "residual_layer_24_input_time.pt",
}


def _ensure_cache_dir(cache_dir: Path | None = None) -> Path:
    """Create (if needed) and return the asset cache directory."""
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_file(url: str, dest: Path) -> None:
    """Download a file from ``url`` to ``dest``."""
    import urllib.request

    print(f"  Downloading {dest.name}...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def load_base_probe(
    model: str = "qwen2.5-3b",
    cache_dir: Path | None = None,
    timing: str = "generation",
) -> CalibrationProbe:
    """
    Download (if needed) and load a pre-trained base probe.

    Returns a ready-to-use :class:`~sottovoce.CalibrationProbe`. Pair it with a
    :class:`~sottovoce.SelfCorrector`, score responses directly, or transfer it
    to another architecture with ``train_projection`` / ``load_projection``.

    Args:
        model: Source model key. See :data:`BASE_PROBES` for available probes
            (currently ``"qwen2.5-3b"``).
        cache_dir: Override the asset cache directory
            (default: ``~/.cache/sottovoce/``).
        timing: Which probe to load, and it changes what you must score.

            ``"generation"`` (default) reads the residual after the model's
            completed answer. Score it on ``prompt + answer``; this is what
            :class:`~sottovoce.SelfCorrector` does. Best under a chat template
            (0.852) and under adversarial context injection (0.657), and weak
            under few-shot prompting (0.616), where free ``EntropyGate`` beats
            it outright.

            ``"input"`` reads the residual at the last prompt token, before
            generation. Score it on the **prompt alone**, and do not hand it to
            ``SelfCorrector`` (which would score it out of distribution; it
            raises). This artifact reads 0.737 / 0.798 / 0.819 across few-shot /
            raw / chat, so it is ~3x less sensitive to prompt format than the
            generation-time probe (swing 0.081 against 0.236) and much better
            under few-shot. It needs one fewer forward pass and can gate *before*
            the model generates. It is worse under attack (0.591 against 0.657).

    Returns:
        A loaded ``CalibrationProbe`` with a ``.timing`` attribute set.

    Raises:
        ValueError: If no base probe is available for ``model``/``timing``.
    """
    from sottovoce.probe import CalibrationProbe

    if timing not in ("generation", "input"):
        raise ValueError(
            f"timing must be 'generation' or 'input', got {timing!r}."
        )

    table = BASE_PROBES if timing == "generation" else INPUT_TIME_PROBES
    if model not in table:
        available = ", ".join(sorted(table))
        raise ValueError(
            f"No {timing}-time base probe available for {model!r}. "
            f"Available: {available}. "
            "Train your own with `python -m sottovoce.train`."
        )

    asset = table[model]
    cache = _ensure_cache_dir(cache_dir)
    path = cache / asset
    if not path.exists():
        _download_file(f"{RELEASE_URL}/{asset}", path)

    probe = CalibrationProbe.from_pretrained(path)
    probe.timing = timing
    return probe
