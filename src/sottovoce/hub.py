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
BASE_PROBES = {
    "qwen2.5-3b": "residual_layer_24.pt",
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

    Returns:
        A loaded ``CalibrationProbe``.

    Raises:
        ValueError: If no base probe is available for ``model``.
    """
    from sottovoce.probe import CalibrationProbe

    if model not in BASE_PROBES:
        available = ", ".join(sorted(BASE_PROBES))
        raise ValueError(
            f"No base probe available for {model!r}. Available: {available}. "
            "Train your own with `python -m sottovoce.train`."
        )

    asset = BASE_PROBES[model]
    cache = _ensure_cache_dir(cache_dir)
    path = cache / asset
    if not path.exists():
        _download_file(f"{RELEASE_URL}/{asset}", path)

    return CalibrationProbe.from_pretrained(path)
