from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from vani_persona import CONFIG_DIR


@lru_cache(maxsize=4)
def load_thresholds(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG_DIR / "thresholds.yaml").read_text())
