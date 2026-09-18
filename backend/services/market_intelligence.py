from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from typing import Iterable

from backend.domain.models import (
    EngineConfig,
    MarketIntelligenceDriver,
    MarketIntelligenceInstrument,
    MarketIntelligenceMacroEvent,
    MarketIntelligenceResponse,
    MarketEventRecord,
    StrategyAnalysis,
    WatchlistItem,
)
from backend.calendar import get_next_event
from backend.services.market_data import MarketDataError, get_bars
from backend.storage.repositories import list_event_alerts, list_market_events, list_recent_analyses


_SYMBOL_META = {
    "US30": ("Dow Jones CFD", "index"),
    "YM": ("Dow futures", "futures"),
    "NAS100": ("Nasdaq 100 CFD", "index"),
    "NDX": ("Nasdaq 100", "index"),
    "US500": ("S&P 500 CFD", "index"),
    "VIX": ("Volatility index", "volatility"),
    "XAUUSD": ("Gold spot", "commodity"),
    "XAGUSD": ("Silver spot", "commodity"),
    "BTCUSD": ("Bitcoin", "crypto"),
    "ETHUSD": ("Ethereum", "crypto"),
    "BRENT": ("Brent crude", "commodity"),
    "WTI": ("WTI crude", "commodity"),
    "EURUSD": ("Euro / US dollar", "fx"),
    "GBPUSD": ("British pound / US dollar", "fx"),
    "USDJPY": ("US dollar / Japanese yen", "fx"),
    "USDOLLAR": ("US dollar index proxy", "fx"),
}


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _watchlist_from_config(config: EngineConfig) -> list[WatchlistItem]:
    return [item for item in config.watchlist if item.enabled][:12]


def _analysis_by_symbol(analyses: Iterable[StrategyAnalysis]) -> dict[str, StrategyAnalysis]:
    latest: dict[str, StrategyAnalysis] = {}
    for item in analyses:
        symbol = item.symbol.upper()
        if symbol not in latest or item.created_at > latest[symbol].created_at:
            latest[symbol] = item
    return latest


def _bias_from(change_pct: float | None, analysis: StrategyAnalysis | None) -> tuple[str, float]:
    if analysis and analysis.signal in {"long", "short"} and analysis.confidence >= 0.55:
        return ("bullish" if analysis.signal == "long" else "bearish", min(0.95, analysis.confidence))
    if change_pct is None:
        return "unknown", 0.0
    if change_pct >= 0.25:
        return "bullish", min(0.82, 0.45 + abs(change_pct) / 3)
    if change_pct <= -0.25:
        return "bearish", min(0.82, 0.45 + abs(change_pct) / 3)
    return "neutral", 0.4


def _driver_set(
    symbol: str,
    change_pct: float | None,
    analysis: StrategyAnalysis | None,
    events: list[MarketEventRecord] | None = None,
) -> list[MarketIntelligenceDriver]:
    drivers: list[MarketIntelligenceDriver] = []
    upper = symbol.upper()
    if change_pct is not None:
        impact = "bullish" if change_pct > 0 else "bearish" if change_pct < 0 else "neutral"
        drivers.append(
            MarketIntelligenceDriver(
                label="Live price impulse",
                detail=f"{upper} is moving {change_pct:+.2f}% over the current sample window.",
                impact=impact,
            )
        )
    if analysis and analysis.reasons:
        signal_impact = "bullish" if analysis.signal == "long" else "bearish" if analysis.signal == "short" else "neutral"
        drivers.append(
            MarketIntelligenceDriver(
                label=f"{analysis.strategy} signal",
                detail=analysis.reasons[0],
                impact=signal_impact,
            )
        )
    for event in (events or [])[:2]:
        event_impact = (
            "bullish" if event.sentiment == "bullish" else "bearish" if event.sentiment == "bearish" else "neutral"
        )
        drivers.append(
            MarketIntelligenceDriver(
                label=f"{event.event_type.replace('_', ' ').title()} · {event.source}",
                detail=f"{event.title} (credibility {event.credibility_score:.0%}, {event.impact} impact)",
                impact=event_impact,
            )
        )
    if upper in {"XAUUSD", "XAGUSD"}:
        drivers.append(MarketIntelligenceDriver(label="Safe-haven sensitivity", detail="Gold and silver remain sensitive to real yields, dollar strength, and risk-off flows.", impact="neutral"))
    elif upper in {"US30", "NAS100", "US500", "NDX", "YM"}:
        drivers.append(MarketIntelligenceDriver(label="Equity risk appetite", detail="Index direction is most exposed to rate expectations, breadth, and volatility regime.", impact="neutral"))
    elif upper in {"BRENT", "WTI"}:
        drivers.append(MarketIntelligenceDriver(label="Energy supply premium", detail="Crude is sensitive to inventory, OPEC, transport disruption, and geopolitical headlines.", impact="neutral"))
    elif upper in {"BTCUSD", "ETHUSD"}:
        drivers.append(MarketIntelligenceDriver(label="Liquidity proxy", detail="Crypto often amplifies broad liquidity, ETF flow, and risk sentiment shifts.", impact="neutral"))
    return drivers[:4]


def _instrument_from_watch_item(
    item: WatchlistItem,
    analysis: StrategyAnalysis | None,
    events: list[MarketEventRecord] | None = None,
) -> MarketIntelligenceInstrument:
    symbol = item.symbol.upper()
    timeframe = item.timeframe.upper()
    name, category = _SYMBOL_META.get(symbol, (symbol, "market"))
    generated_at = _now()
    try:
        df = get_bars(symbol, timeframe, 240, prefer_live=True)
        close = df["close"].dropna()
        if len(close) < 2:
            raise MarketDataError(f"Not enough bars available for {symbol}:{timeframe}")
        price = _finite(close.iloc[-1])
        previous = _finite(close.iloc[-2])
        first = _finite(close.iloc[0])
        if price is None or previous is None:
            raise MarketDataError(f"Invalid close price for {symbol}:{timeframe}")
        change = price - previous
        change_pct = ((price - first) / first * 100.0) if first else None
        low_range = _finite(df["low"].tail(180).min())
        high_range = _finite(df["high"].tail(180).max())
        support = [_finite(df["low"].tail(60).min()), _finite(df["low"].tail(20).min())]
        resistance = [_finite(df["high"].tail(20).max()), _finite(df["high"].tail(60).max())]
        bias, confidence = _bias_from(change_pct, analysis)
        situation = f"{symbol} is trading around {price:.5g} on {timeframe}, with a {change_pct:+.2f}% sample-window move." if change_pct is not None else f"{symbol} has live bars available on {timeframe}."
        if analysis and analysis.signal != "no_trade":
            direction_note = f"Latest strategy analysis leans {analysis.signal} with {analysis.confidence:.0%} confidence."
        elif bias == "neutral":
            direction_note = "Current movement is balanced; treat this as context until a stronger strategy signal appears."
        else:
            direction_note = f"Short-term bias is {bias}, based on recent live market movement."
        return MarketIntelligenceInstrument(
            symbol=symbol,
            timeframe=timeframe,
            name=name,
            category=category,
            price=price,
            change=change,
            change_pct=change_pct,
            low_range=low_range,
            high_range=high_range,
            bias=bias,  # type: ignore[arg-type]
            confidence=confidence,
            situation=situation,
            direction_note=direction_note,
            drivers=_driver_set(symbol, change_pct, analysis, events),
            support=[value for value in support if value is not None],
            resistance=[value for value in resistance if value is not None],
            last_updated=generated_at,
            data_status="live",
        )
    except Exception as exc:
        return MarketIntelligenceInstrument(
            symbol=symbol,
            timeframe=timeframe,
            name=name,
            category=category,
            situation=f"{symbol} is configured for intelligence monitoring, but market data is unavailable.",
            direction_note="No directional context generated until bars are available.",
            drivers=_driver_set(symbol, None, analysis, events),
            last_updated=generated_at,
            data_status="unavailable",
            error=str(exc),
        )


def _macro_events() -> list[MarketIntelligenceMacroEvent]:
    try:
        event = get_next_event()
    except Exception:
        return []
    if not event:
        return []
    impact = str(event.get("impact") or "unknown").lower()
    if impact not in {"high", "medium", "low", "unknown"}:
        impact = "unknown"
    title = event.get("title") or "No scheduled macro event"
    source = event.get("source") or "calendar"
    return [
        MarketIntelligenceMacroEvent(
            title=str(title),
            impact=impact,  # type: ignore[arg-type]
            source=str(source),
            ts=event.get("ts"),
        )
    ]


def _regime(instruments: list[MarketIntelligenceInstrument]) -> str:
    equity = [item for item in instruments if item.symbol in {"US30", "NAS100", "US500", "NDX", "YM"}]
    havens = [item for item in instruments if item.symbol in {"XAUUSD", "XAGUSD", "VIX"}]
    equity_score = sum((item.change_pct or 0.0) for item in equity)
    haven_score = sum((item.change_pct or 0.0) for item in havens)
    if equity_score > 0.4 and haven_score <= 0.4:
        return "risk_on"
    if equity_score < -0.4 or haven_score > 0.6:
        return "risk_off"
    return "mixed" if instruments else "unknown"


def build_market_intelligence(config: EngineConfig) -> MarketIntelligenceResponse:
    analyses = _analysis_by_symbol(list_recent_analyses(50))
    market_events = list_market_events(30)
    events_by_symbol: dict[str, list[MarketEventRecord]] = {}
    for event in market_events:
        for symbol in event.symbols:
            events_by_symbol.setdefault(symbol.upper(), []).append(event)
    instruments = [
        _instrument_from_watch_item(
            item,
            analyses.get(item.symbol.upper()),
            events_by_symbol.get(item.symbol.upper(), []),
        )
        for item in _watchlist_from_config(config)
    ]
    available = [item for item in instruments if item.data_status != "unavailable"]
    bullish = sum(1 for item in available if item.bias == "bullish")
    bearish = sum(1 for item in available if item.bias == "bearish")
    regime = _regime(available)
    headline = {
        "risk_on": "Risk appetite is constructive across the monitored watchlist.",
        "risk_off": "Defensive pressure is visible across the monitored watchlist.",
        "mixed": "The monitored watchlist is mixed and needs instrument-level confirmation.",
        "unknown": "Market regime is unavailable until live data resolves.",
    }[regime]
    summary = f"{len(available)} of {len(instruments)} instruments have usable market data. Bias count: {bullish} bullish, {bearish} bearish."
    return MarketIntelligenceResponse(
        generated_at=_now(),
        regime=regime,  # type: ignore[arg-type]
        headline=headline,
        summary=summary,
        instruments=instruments,
        macro_events=_macro_events(),
        market_events=market_events[:20],
        event_alerts=list_event_alerts(10),
        source_notes=[
            "Live price context uses the existing cTrader market-data service.",
            "News events are deduplicated, credibility-scored, classified, and linked to affected symbols.",
            "Event sentiment is contextual evidence only; it cannot place trades or bypass deterministic risk rules.",
        ],
    )
