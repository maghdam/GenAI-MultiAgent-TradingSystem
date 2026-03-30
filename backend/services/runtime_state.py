from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExternalDependencyState:
    ctrader_started: bool = False
    ctrader_reason: str = ""
    ollama_warmed: bool = False
    ollama_reason: str = ""
    notes: list[str] = field(default_factory=list)

    def snapshot_notes(self) -> list[str]:
        notes = list(self.notes)
        if self.ctrader_reason:
            notes.append(f"ctrader: {self.ctrader_reason}")
        if self.ollama_reason:
            notes.append(f"ollama: {self.ollama_reason}")
        return notes


@dataclass
class MarketDataDependencyState:
    last_checked_at: datetime | None = None
    last_symbol: str = ""
    last_timeframe: str = ""
    last_success: bool | None = None
    last_success_at: datetime | None = None
    last_reason: str = ""
    market_data_ready: bool = False

    def snapshot(self) -> dict[str, object]:
        return {
            "last_checked_at": self.last_checked_at.isoformat() if self.last_checked_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_symbol": self.last_symbol or None,
            "last_timeframe": self.last_timeframe or None,
            "last_success": self.last_success,
            "last_reason": self.last_reason,
            "market_data_ready": self.market_data_ready,
        }


external_dependency_state = ExternalDependencyState()
market_data_dependency_state = MarketDataDependencyState()
