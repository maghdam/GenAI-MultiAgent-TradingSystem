from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from backend.services import alpaca_market_data, market_data


def _bar(at: datetime, *, price: float, volume: float = 10.0) -> dict[str, object]:
    return {
        "t": at.isoformat().replace("+00:00", "Z"),
        "o": price - 0.2,
        "h": price + 0.3,
        "l": price - 0.4,
        "c": price,
        "v": volume,
    }


def test_stock_bars_use_explicit_iex_feed_and_return_sorted_frame(monkeypatch) -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    captured: dict[str, object] = {}

    def _request(path: str, *, params: dict[str, object]):
        captured["path"] = path
        captured["params"] = dict(params)
        return {
            "bars": {
                "SPY": [
                    _bar(now, price=101.0),
                    _bar(now - timedelta(minutes=5), price=100.0),
                ]
            },
            "next_page_token": None,
        }

    monkeypatch.delenv("ALPACA_STOCK_DATA_FEED", raising=False)
    monkeypatch.setattr(alpaca_market_data, "_request_json", _request)

    frame = alpaca_market_data.get_bars(
        symbol="spy",
        timeframe="M5",
        num_bars=2,
        asset_class="us_equity",
    )

    assert captured["path"] == "/v2/stocks/bars"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["symbols"] == "SPY"
    assert params["timeframe"] == "5Min"
    assert params["feed"] == "iex"
    assert params["sort"] == "desc"
    assert params["adjustment"] == "raw"
    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame["close"].tolist() == [100.0, 101.0]


def test_crypto_bars_follow_pagination_without_symbol_inference(monkeypatch) -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    responses = [
        {
            "bars": {"BTC/USD": [_bar(now, price=80_000.0)]},
            "next_page_token": "older",
        },
        {
            "bars": {"BTC/USD": [_bar(now - timedelta(minutes=5), price=79_900.0)]},
            "next_page_token": None,
        },
    ]
    calls: list[tuple[str, dict[str, object]]] = []

    def _request(path: str, *, params: dict[str, object]):
        calls.append((path, dict(params)))
        return responses.pop(0)

    monkeypatch.setattr(alpaca_market_data, "_request_json", _request)

    frame = alpaca_market_data.get_bars(
        symbol="BTC/USD",
        timeframe="M5",
        num_bars=2,
        asset_class="crypto",
    )

    assert [path for path, _ in calls] == [
        "/v1beta3/crypto/us/bars",
        "/v1beta3/crypto/us/bars",
    ]
    assert calls[0][1]["symbols"] == "BTC/USD"
    assert calls[0][1]["timeframe"] == "5Min"
    assert "feed" not in calls[0][1]
    assert calls[1][1]["page_token"] == "older"
    assert frame["close"].tolist() == [79_900.0, 80_000.0]


def test_malformed_alpaca_bar_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        alpaca_market_data,
        "_request_json",
        lambda *args, **kwargs: {
            "bars": {
                "SPY": [
                    {
                        "t": datetime.now(UTC).isoformat(),
                        "o": 100,
                        "h": 101,
                        "l": 99,
                        "c": "bad",
                        "v": 10,
                    }
                ]
            }
        },
    )

    with pytest.raises(alpaca_market_data.AlpacaMarketDataError, match="malformed"):
        alpaca_market_data.get_bars(
            symbol="SPY",
            timeframe="M5",
            num_bars=1,
            asset_class="us_equity",
        )


def test_shared_market_data_routes_alpaca_without_ctrader_or_persistent_cache(monkeypatch) -> None:
    market_data._bars_cache.clear()
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    frame = pd.DataFrame(
        [
            {
                "open": 99.8,
                "high": 100.3,
                "low": 99.5,
                "close": 100.0,
                "volume": 1000.0,
            }
        ],
        index=pd.DatetimeIndex([now]),
    )
    calls = {"alpaca": 0}

    def _alpaca(**kwargs):
        calls["alpaca"] += 1
        assert kwargs == {
            "symbol": "SPY",
            "timeframe": "M5",
            "num_bars": 1,
            "asset_class": "us_equity",
        }
        return frame

    monkeypatch.setattr(market_data, "get_alpaca_bars", _alpaca)
    monkeypatch.setattr(
        market_data.adapter,
        "get_bars",
        lambda **kwargs: pytest.fail("Alpaca venue must not call cTrader market data"),
    )
    monkeypatch.setattr(
        market_data,
        "load_cached_market_bars",
        lambda *args, **kwargs: pytest.fail("Alpaca venue must not read the cTrader persistent cache"),
    )
    monkeypatch.setattr(
        market_data,
        "upsert_market_bars",
        lambda *args, **kwargs: pytest.fail("Alpaca venue must not write the cTrader persistent cache"),
    )

    first = market_data.get_bars(
        "SPY",
        "M5",
        1,
        venue="alpaca",
        alpaca_asset_class="us_equity",
    )
    second = market_data.get_bars(
        "SPY",
        "M5",
        1,
        venue="alpaca",
        alpaca_asset_class="us_equity",
    )

    assert calls["alpaca"] == 1
    assert first.equals(second)
    assert ("ALPACA:US_EQUITY:SPY", "M5") in market_data._bars_cache


def test_shared_market_data_normalizes_alpaca_adapter_errors(monkeypatch) -> None:
    market_data._bars_cache.clear()
    monkeypatch.setattr(
        market_data,
        "get_alpaca_bars",
        lambda **kwargs: (_ for _ in ()).throw(
            alpaca_market_data.AlpacaMarketDataError("Alpaca feed unavailable.")
        ),
    )

    with pytest.raises(market_data.MarketDataError, match="Alpaca feed unavailable"):
        market_data.get_bars(
            "SPY",
            "M5",
            10,
            venue="alpaca",
            alpaca_asset_class="us_equity",
        )


def test_paper_stock_calendar_uses_only_get_with_bounded_trading_dates(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return [{"date": "2026-11-27", "open": "09:30", "close": "13:00"}]

    class Client:
        def __init__(self, *, base_url, timeout, headers):
            assert base_url == "https://paper-api.alpaca.markets"
            assert headers["APCA-API-KEY-ID"] == "key"
            assert headers["APCA-API-SECRET-KEY"] == "secret"
            assert timeout == 15.0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, path, *, params):
            calls.append((path, params))
            return Response()

    monkeypatch.delenv("ALPACA_PAPER_BASE_URL", raising=False)
    monkeypatch.setattr(alpaca_market_data, "_credentials", lambda: ("key", "secret"))
    monkeypatch.setattr(alpaca_market_data.httpx, "Client", Client)

    result = alpaca_market_data.get_stock_trading_calendar(
        start_date="2026-11-25", end_date="2026-11-30"
    )

    assert result[0]["close"] == "13:00"
    assert calls == [
        ("/v2/calendar", {"start": "2026-11-25", "end": "2026-11-30", "date_type": "TRADING"})
    ]


def test_stock_calendar_invalid_bounds_fail_before_credential_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        alpaca_market_data, "_credentials",
        lambda: pytest.fail("Bad dates must fail before reading credentials"),
    )
    for first, last in (
        ("2026-11-30", "2026-11-25"),
        ("invalid", "2026-11-25"),
    ):
        with pytest.raises(ValueError):
            alpaca_market_data.get_stock_trading_calendar(
                start_date=first, end_date=last
            )
