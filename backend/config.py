from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "tradeagent.db"


@dataclass(frozen=True)
class AppSettings:
    version: str = "1.0.0"
    db_path: Path = DB_PATH


SETTINGS = AppSettings()
