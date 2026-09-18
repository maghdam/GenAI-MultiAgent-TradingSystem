from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def resolve_db_path() -> Path:
    configured = (os.getenv("TRADEAGENT_DB_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DATA_DIR / "tradeagent.db"


DB_PATH = resolve_db_path()


@dataclass(frozen=True)
class AppSettings:
    version: str = "1.0.0"
    db_path: Path = DB_PATH


SETTINGS = AppSettings()
