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
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_lot_size_map", {7: 100.0})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_min_volume_map", {7: 100})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_step_volume_map", {7: 100})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_max_volume_map", {7: 1_000_000})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_min_verified", {7: True})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_step_verified", {7: False})

    adapter = CTraderBrokerAdapter()
    limits = adapter.get_symbol_limits("xauusd")

    assert limits.symbol == "XAUUSD"
    assert limits.source == "broker"
    assert limits.min_api_units == 100
    assert limits.step_api_units == 100
    assert limits.max_api_units == 1_000_000
    assert limits.min_lots == 0.01
    assert limits.step_lots == 0.01
    assert limits.max_lots == 100.0
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


def test_get_instrument_spec_uses_broker_contract_for_usd_account(monkeypatch) -> None:
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_name_to_id", {"US100": 7})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_lot_size_map", {7: 1.0})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_digits_map", {7: 2})

    spec = CTraderBrokerAdapter().get_instrument_spec("us100", "USD")

    assert spec.valuation_ready is True
    assert spec.verified is True
    assert spec.quote_currency == "USD"
    assert spec.cash_per_price_unit_per_lot == 1.0
    assert spec.tick_size == 0.01
    assert spec.tick_value_per_lot == 0.01


def test_get_instrument_spec_blocks_unavailable_currency_conversion(monkeypatch) -> None:
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_name_to_id", {"EURJPY": 9})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_lot_size_map", {9: 100_000.0})
    monkeypatch.setattr("backend.adapters.ctrader.ctd.symbol_digits_map", {9: 3})

    spec = CTraderBrokerAdapter().get_instrument_spec("EURJPY", "USD")

    assert spec.valuation_ready is False
    assert spec.cash_per_price_unit_per_lot is None
    assert any("JPY/USD" in note for note in spec.notes)


def test_ctrader_volume_conversion_uses_symbol_contract_size(monkeypatch) -> None:
    from backend import ctrader_client as ctd

    monkeypatch.setattr(ctd, "symbol_lot_size_map", {7: 100.0, 8: 100_000.0})
    monkeypatch.setattr(ctd, "symbol_min_volume_map", {7: 100, 8: 100_000})
    monkeypatch.setattr(ctd, "symbol_step_volume_map", {7: 100, 8: 100_000})
    monkeypatch.setattr(ctd, "symbol_max_volume_map", {7: 1_000_000, 8: 100_000_000})

    # XAUUSD-style 100-unit lot: 0.01 lot = 1 measurement unit = protocol volume 100.
    assert ctd.volume_lots_to_units(7, 0.01) == 100
    assert ctd.protocol_volume_to_lots(7, 100) == 0.01

    # FX-style 100,000-unit lot: 0.01 lot = 1,000 units = protocol volume 100,000.
    assert ctd.volume_lots_to_units(8, 0.01) == 100_000
    assert ctd.protocol_volume_to_lots(8, 100_000) == 0.01


def test_symbol_details_preserve_protocol_volume_metadata(monkeypatch) -> None:
    from types import SimpleNamespace
    from backend import ctrader_client as ctd

    monkeypatch.setattr(ctd, "symbol_lot_size_map", {})
    monkeypatch.setattr(ctd, "symbol_min_volume_map", {})
    monkeypatch.setattr(ctd, "symbol_step_volume_map", {})
    monkeypatch.setattr(ctd, "symbol_max_volume_map", {})
    monkeypatch.setattr(ctd, "symbol_min_verified", {})
    monkeypatch.setattr(ctd, "symbol_step_verified", {})
    monkeypatch.setattr(ctd, "symbol_digits_map", {})
    monkeypatch.setattr(
        ctd.Protobuf,
        "extract",
        lambda _: SimpleNamespace(
            symbol=[
                SimpleNamespace(
                    symbolId=7,
                    digits=2,
                    lotSize=10_000,      # 100.00 measurement units per lot
                    minVolume=100,       # 1.00 measurement unit = 0.01 lot
                    stepVolume=100,
                    maxVolume=1_000_000,
                )
            ]
        ),
    )

    ctd.symbol_details_response_cb(object())

    assert ctd.symbol_lot_size_map[7] == 100.0
    assert ctd.symbol_min_volume_map[7] == 100
    assert ctd.symbol_step_volume_map[7] == 100
    assert ctd.symbol_max_volume_map[7] == 1_000_000
    assert ctd.volume_lots_to_units(7, 0.01) == 100


def test_demo_symbol_execution_readiness_waits_for_full_contract(monkeypatch) -> None:
    from backend import ctrader_client as ctd

    monkeypatch.setattr(ctd, "is_demo_account_confirmed", lambda: True)
    monkeypatch.setattr(ctd, "get_account_verification_error", lambda: None)
    monkeypatch.setattr(ctd, "symbol_name_to_id", {"XAUUSD": 7})
    monkeypatch.setattr(ctd, "symbol_lot_size_map", {})
    monkeypatch.setattr(ctd, "symbol_min_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_step_volume_map", {7: 100})
    monkeypatch.setattr(ctd, "symbol_max_volume_map", {7: 500_000})

    adapter = CTraderBrokerAdapter()
    ready, reason = adapter.demo_symbol_execution_readiness("XAUUSD")
    assert ready is False
    assert "lotSize" in reason

    monkeypatch.setattr(ctd, "symbol_lot_size_map", {7: 100.0})
    ready, reason = adapter.demo_symbol_execution_readiness("XAUUSD")
    assert ready is True
    assert "ready" in reason.lower()
