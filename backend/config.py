from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LEGACY_DB_PATH = DATA_DIR / "tradeagent.db"


def resolve_db_path() -> Path:
    configured = (os.getenv("TRADEAGENT_DB_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    local_app_data = (os.getenv("LOCALAPPDATA") or "").strip()
    if local_app_data:
        return (Path(local_app_data) / "TradeAgent" / "data" / "tradeagent.db").expanduser().resolve()

    xdg_state_home = (os.getenv("XDG_STATE_HOME") or "").strip()
    if xdg_state_home:
        return (Path(xdg_state_home) / "tradeagent" / "tradeagent.db").expanduser().resolve()

    return (Path.home() / ".local" / "state" / "tradeagent" / "tradeagent.db").expanduser().resolve()


DB_PATH = resolve_db_path()


@dataclass(frozen=True)
class AppSettings:
    version: str = "1.0.0"
    db_path: Path = DB_PATH


SETTINGS = AppSettings()
