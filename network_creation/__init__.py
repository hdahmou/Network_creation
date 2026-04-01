"""Package initialization for network_creation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def _configure_matplotlib_cache() -> None:
    """Point Matplotlib at a persistent writable cache for short-lived workers."""

    if "MPLCONFIGDIR" in os.environ:
        return

    cache_dir = Path(tempfile.gettempdir()) / "network_creation-mpl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


_configure_matplotlib_cache()
