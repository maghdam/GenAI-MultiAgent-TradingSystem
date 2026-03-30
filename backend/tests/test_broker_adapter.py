from __future__ import annotations

from backend.adapters.ctrader import CTraderBrokerAdapter


def test_start_transport_is_idempotent(monkeypatch) -> None:
    started_targets: list[object] = []

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            started_targets.append(self.target)

    monkeypatch.setattr("backend.adapters.ctrader.threading.Thread", FakeThread)

    adapter = CTraderBrokerAdapter()

    assert adapter.start_transport() is True
    assert adapter.transport_started() is True
    assert adapter.start_transport() is False
    assert len(started_targets) == 1


def test_list_symbols_returns_sorted_symbol_names(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.adapters.ctrader.ctd.symbol_name_to_id",
        {"XAUUSD": 2, "AUDUSD": 3, "EURUSD": 1},
    )

    adapter = CTraderBrokerAdapter()

    assert adapter.list_symbols() == ["AUDUSD", "EURUSD", "XAUUSD"]


def test_get_bars_normalizes_rows_and_truncates_to_requested_count(monkeypatch) -> None:
    monkeypatch.setattr("backend.adapters.ctrader.ctd.is_connected", lambda: True)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.is_authorized", lambda: True)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_name_to_id", {"XAUUSD": 1})
    monkeypatch.setattr(
        "backend.adapters.ctrader.ctd.get_ohlc_data",
        lambda symbol, tf, n: [
            {"time": "invalid", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "10"},
            {"time": "2026-03-19T10:10:00Z", "open": "100.5", "high": "101.5", "low": "99.5", "close": "101.0", "volume": "12"},
            {"time": "2026-03-19T10:00:00Z", "open": "99.0", "high": "100.0", "low": "98.5", "close": "99.5", "volume": "8"},
            {"time": "2026-03-19T10:05:00Z", "open": "99.5", "high": "101.0", "low": "99.0", "close": "100.5", "volume": "9"},
        ],
    )

    adapter = CTraderBrokerAdapter()
    bars, live_price = adapter.get_bars("xauusd", "m5", 2)

    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == 2
    assert list(bars.index.strftime("%Y-%m-%dT%H:%M:%SZ")) == ["2026-03-19T10:05:00Z", "2026-03-19T10:10:00Z"]
    assert bars["open"].dtype.kind in {"f", "i"}
    assert bars.iloc[0]["close"] == 100.5
    assert live_price == 101.0


def test_get_status_reports_degraded_reads_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr("backend.adapters.ctrader.ctd.is_connected", lambda: True)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.is_authorized", lambda: True)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.get_auth_error", lambda: None)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.get_last_auth_attempt", lambda: None)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_name_to_id", {"EURUSD": 1})
    monkeypatch.setattr("backend.adapters.ctrader.external_dependency_state.snapshot_notes", lambda: ["startup skipped"])

    monkeypatch.setattr(
        "backend.adapters.ctrader.ctd.get_reconcile_snapshot",
        lambda: {"positions": [], "orders": [], "error": "reconcile unavailable"},
    )
    monkeypatch.setattr("backend.adapters.ctrader.ctd.HOST_TYPE", "demo")
    monkeypatch.setattr("backend.adapters.ctrader.ctd.ACCOUNT_ID", "acct-123")

    adapter = CTraderBrokerAdapter()
    status = adapter.get_status()

    assert status.connected is True
    assert status.symbols_loaded == 1
    assert status.ready is True
    assert status.open_positions == 0
    assert status.pending_orders == 0
    assert status.account_id is None
    assert "reconcile_unavailable: reconcile unavailable" in status.notes
    assert "startup skipped" in status.notes


def test_get_market_data_status_reports_empty_feed(monkeypatch) -> None:
    monkeypatch.setattr("backend.adapters.ctrader.ctd.is_connected", lambda: True)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.is_authorized", lambda: True)
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_name_to_id", {"XAUUSD": 1})
    monkeypatch.setattr(
        "backend.adapters.ctrader.ctd.get_ohlc_data",
        lambda symbol, tf, n: [],
    )

    adapter = CTraderBrokerAdapter()
    status = adapter.get_market_data_status("XAUUSD", "M5")

    assert status["ok"] is False
    assert status["reason"] == "No market data available for XAUUSD:M5"


def test_get_symbol_limits_uses_broker_metadata(monkeypatch) -> None:
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_min_volume_map", {7: 250})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_step_volume_map", {7: 50})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_max_volume_map", {7: 250000})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_min_verified", {7: True})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_step_verified", {7: False})

    adapter = CTraderBrokerAdapter()
    limits = adapter.get_symbol_limits("xauusd")

    assert limits.symbol == "XAUUSD"
    assert limits.source == "broker"
    assert limits.min_api_units == 250
    assert limits.step_api_units == 50
    assert limits.max_api_units == 250000
    assert limits.min_lots == 0.025
    assert limits.step_lots == 0.005
    assert limits.max_lots == 25.0
    assert limits.hard_min is True
    assert limits.hard_step is False


def test_get_symbol_limits_falls_back_when_symbol_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_name_to_id", {"EURUSD": 1})

    adapter = CTraderBrokerAdapter()
    limits = adapter.get_symbol_limits("XAUUSD")

    assert limits.symbol == "XAUUSD"
    assert limits.source == "fallback"
    assert limits.min_lots == 0.01
    assert limits.step_lots == 0.01
    assert limits.max_lots == 100.0
