from __future__ import annotations

from typing import Any

import pytest

from scripts import verify_runtime_strategy_matrix as verifier


class _Response:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected GET: {url}")
        return self.responses.pop(0)


def _demo_status() -> dict[str, Any]:
    return {
        "broker": {
            "account_type": "demo",
            "account_verified": True,
            "demo_account_confirmed": True,
            "socket_connected": True,
            "account_authorized": True,
            "market_data_ready": True,
            "account_id": 123,
        }
    }


def _audit_payload(strategy: str, symbol: str, timeframe: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "Fetched Bars": 1500,
        "Full Backtest": {
            "Total Return [%]": 1.25,
            "Number of Trades": 12.0,
            "Win Rate [%]": 58.33,
            "Expectancy [%]": 0.11,
            "Max Drawdown [%]": -2.2,
            "Trades/Day": 0.8,
        },
        "Out of Sample": {
            "Total Return [%]": 0.25,
            "Number of Trades": 4.0,
            "Expectancy [%]": 0.06,
        },
        "Costs": {"Cost Drag [%]": 0.15},
        "Regime Analysis": [{}, {}, {}],
        "Parameter Sensitivity": {"Cases": [{}, {}, {}, {}, {}]},
    }


def test_enabled_targets_uses_only_enabled_unique_watchlist_rows() -> None:
    config = {
        "watchlist": [
            {"symbol": "xauusd", "timeframe": "m5", "enabled": True},
            {"symbol": "XAUUSD", "timeframe": "M5", "enabled": True},
            {"symbol": "NAS100", "timeframe": "M5", "enabled": False},
            {"symbol": "US30", "timeframe": "m15", "enabled": True},
            {"symbol": "", "timeframe": "M5", "enabled": True},
        ]
    }

    assert verifier._enabled_targets(config) == [("XAUUSD", "M5"), ("US30", "M15")]


def test_collect_matrix_requires_authenticated_active_account() -> None:
    status = _demo_status()
    status["broker"]["account_verified"] = False
    status["broker"]["demo_account_confirmed"] = False
    session = _Session([_Response(200, status)])

    with pytest.raises(RuntimeError, match="authenticated active account"):
        verifier.collect_matrix(session=session, base_url="http://127.0.0.1:4000")

    assert len(session.calls) == 1
    assert session.calls[0]["url"].endswith("/api/status")


def test_collect_matrix_queries_every_strategy_for_enabled_targets() -> None:
    config = {
        "watchlist": [
            {"symbol": "XAUUSD", "timeframe": "M5", "enabled": True},
            {"symbol": "US30", "timeframe": "M5", "enabled": True},
        ]
    }
    audits = [
        _audit_payload(strategy, symbol, "M5")
        for symbol in ("XAUUSD", "US30")
        for strategy in verifier.RUNTIME_STRATEGIES
    ]
    session = _Session(
        [
            _Response(200, _demo_status()),
            _Response(200, config),
            *[_Response(200, item) for item in audits],
        ]
    )

    result = verifier.collect_matrix(
        session=session,
        base_url="http://127.0.0.1:4000/",
        api_key="secret",
        num_bars=1500,
        fee_bps=1.0,
        slippage_bps=1.0,
        spread_bps=2.0,
    )

    assert result["all_ok"] is True
    assert result["available_rows"] == 6
    assert result["total_rows"] == 6
    assert result["targets"] == [
        {"symbol": "XAUUSD", "timeframe": "M5"},
        {"symbol": "US30", "timeframe": "M5"},
    ]
    audit_calls = [call for call in session.calls if call["url"].endswith("/api/studio/runtime-strategy-audit")]
    assert len(audit_calls) == 6
    assert {call["params"]["strategy"] for call in audit_calls} == set(verifier.RUNTIME_STRATEGIES)
    assert all(call["headers"] == {"x-api-key": "secret"} for call in session.calls)
    assert all(call["params"]["num_bars"] == 1500 for call in audit_calls)
    assert all(call["params"]["fee_bps"] == 1.0 for call in audit_calls)
    assert all(call["params"]["slippage_bps"] == 1.0 for call in audit_calls)
    assert all(call["params"]["spread_bps"] == 2.0 for call in audit_calls)


def test_collect_matrix_preserves_unavailable_rows_and_continues() -> None:
    config = {"watchlist": [{"symbol": "XAUUSD", "timeframe": "M5", "enabled": True}]}
    session = _Session(
        [
            _Response(200, _demo_status()),
            _Response(200, config),
            _Response(503, {"detail": "No market data available for XAUUSD:M5."}),
            _Response(200, _audit_payload("rsi_reversal", "XAUUSD", "M5")),
            _Response(200, _audit_payload("breakout", "XAUUSD", "M5")),
        ]
    )

    result = verifier.collect_matrix(session=session, base_url="http://127.0.0.1:4000")

    assert result["all_ok"] is False
    assert result["available_rows"] == 2
    assert result["total_rows"] == 3
    assert result["results"][0]["status"] == "unavailable"
    assert "HTTP 503" in result["results"][0]["error"]
    assert [row["status"] for row in result["results"][1:]] == ["ok", "ok"]
