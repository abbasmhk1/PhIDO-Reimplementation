from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Data folder: $PHIDO_DATA if set, else <repo>/data (works for editable installs)."""
    env = os.environ.get("PHIDO_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"
