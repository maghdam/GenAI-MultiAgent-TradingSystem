from __future__ import annotations

import asyncio

from backend.domain.models import EngineConfig, WatchlistItem
from backend.services import market_data_monitor
from backend.services.runtime_state import market_data_dependency_state


def test_probe_target_prefers_first_enabled_watch_item() -> None:
    config = EngineConfig(
        default_symbol="EURUSD",
        default_timeframe="H1",
        watchlist=[
            WatchlistItem(symbol="NAS100", timeframe="M1", enabled=False),
            WatchlistItem(symbol="XAUUSD", timeframe="M5", enabled=True),
        ],
    )

    assert market_data_monitor.resolve_market_data_probe_target(config) == ("XAUUSD", "M5")


def test_probe_market_data_once_runs_blocking_adapter_off_event_loop(monkeypatch) -> None:
    config = EngineConfig(default_symbol="XAUUSD", default_timeframe="M5")
    monkeypatch.setattr(market_data_monitor, "load_engine_config", lambda _: config)

    def slow_probe(symbol: str, timeframe: str) -> dict[str, object]:
        import time

        time.sleep(0.15)
        return {"ok": True, "symbol": symbol, "timeframe": timeframe}

    monkeypatch.setattr(market_data_monitor, "get_market_data_status", slow_probe)

    async def run_probe() -> float:
        import time

        started = time.monotonic()
        task = asyncio.create_task(market_data_monitor.probe_market_data_once())
        await asyncio.sleep(0.02)
        event_loop_delay = time.monotonic() - started
        await task
        return event_loop_delay

    assert asyncio.run(run_probe()) < 0.1


def test_probe_failure_is_visible_in_readiness_state(monkeypatch) -> None:
    config = EngineConfig(default_symbol="XAUUSD", default_timeframe="M5")
    monkeypatch.setattr(market_data_monitor, "load_engine_config", lambda _: config)
    monkeypatch.setattr(
        market_data_monitor,
        "get_market_data_status",
        lambda *_: (_ for _ in ()).throw(RuntimeError("feed timeout")),
    )

    result = asyncio.run(market_data_monitor.probe_market_data_once())

    assert result["ok"] is False
    assert market_data_dependency_state.market_data_ready is False
    assert "feed timeout" in market_data_dependency_state.last_reason


def test_short_trendbar_probe_looks_back_across_weekends() -> None:
    from backend.ctrader_client import _trendbar_lookback_minutes

    assert _trendbar_lookback_minutes("M5", 5) == 7 * 24 * 60
    assert _trendbar_lookback_minutes("D1", 20) > 7 * 24 * 60


def test_large_ctrader_metadata_event_skips_expensive_serialization(monkeypatch, capsys) -> None:
    from backend import ctrader_client as ctd

    class ProtoOASymbolsListRes:
        symbol = range(6_456)

    monkeypatch.setattr(
        ctd,
        "MessageToDict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not serialize")),
    )
    ctd._log_event(ProtoOASymbolsListRes())
