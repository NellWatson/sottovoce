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
    "https://github.com/NellWatson/sottovoce/releases/download/v0.3.2"
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
# question-grouped 5-fold out-of-fold CV). These are the same reference numbers as
# the README table; the generation-time row is method-level and the input-time row
# is the raw-fed shipped artifact (score() feeds raw activations, so that is what
# the input-time probe actually gets):
#
#   timing                            few-shot   raw    chat   swing
#   generation-time (default)           0.616   0.767  0.852   0.236
#   input-time (timing="input")         0.737   0.798  0.819   0.081   raw-fed artifact
#
# Both shipped probes are verified end-to-end on a held-out 200-item split
# (indices 500-699) -- the number you actually get from load_base_probe():
#
#   held-out deployment               few-shot   raw    chat
#   generation-time (v0.3.2, v12)        0.53    0.79   0.84
#   input-time      (v8)                 0.74    0.78   0.79
#
# The generation-time default was retrained for v0.3.2 with a documented, committed
# procedure (research/results/_published/v12_gen_time_probe_reship.json) after the
# previous, undocumented checkpoint read only 0.74-0.77 on chat in deployment; the
# new one reproduces its chat number within CI (held-out 0.837 [0.779, 0.890]).
# Re-measuring generation-time raw-fed (v10) also shows it pays no scaler tax --
# within format it reads 0.612/0.772/0.870 -- so the method-level row above is a
# fair reflection of the deployed probe, not a flattering one.
#
# The ordering: generation-time is better under chat (0.852 vs 0.819) and under
# adversarial context injection (0.657 vs 0.591), because the attack lives in the
# prompt and that is all the input-time probe reads; input-time is ~3x more robust
# to how you prompt (swing 0.081 vs 0.236) and much better under few-shot. Neither
# dominates. Pick per deployment.
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
