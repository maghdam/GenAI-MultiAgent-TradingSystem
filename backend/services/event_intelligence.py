from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import unescape
import os
import re
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

from backend.domain.models import EventRefreshResponse, MarketEventInput, MarketEventRecord
from backend.storage.repositories import (
    add_event_alert,
    add_event_source_run,
    insert_market_event,
    list_market_events,
    log_incident,
)


CLASSIFICATION_VERSION = "deterministic-v1"
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "NAS100": ("nasdaq 100", "nasdaq-100", "nasdaq100", "nasdaq futures", "ndx", "nq futures"),
    "NVDA": ("nvda", "nvidia"),
    "MSFT": ("msft", "microsoft"),
    "AAPL": ("aapl", "apple"),
    "AMZN": ("amzn", "amazon"),
    "META": ("meta platforms", "facebook"),
    "GOOGL": ("googl", "google", "alphabet"),
    "TSLA": ("tsla", "tesla"),
    "AVGO": ("avgo", "broadcom"),
    "AMD": ("amd", "advanced micro devices"),
    "NFLX": ("nflx", "netflix"),
    "QCOM": ("qcom", "qualcomm"),
    "INTC": ("intc", "intel"),
    "US500": ("s&p 500", "sp500", "s&p futures"),
    "US30": ("dow jones", "dow futures", "djia"),
    "XAUUSD": ("gold price", "spot gold", "xauusd"),
    "BTCUSD": ("bitcoin", "btc-usd", "btcusd"),
}

_NASDAQ_COMPONENTS = {"NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX", "QCOM", "INTC"}

_EVENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("earnings", ("earnings", "quarterly results", "revenue", "eps", "profit")),
    ("guidance", ("guidance", "outlook", "forecast", "raises full-year", "cuts full-year")),
    ("monetary_policy", ("federal reserve", "fed rate", "interest rate", "fomc", "powell", "inflation", "cpi")),
    ("regulation", ("regulator", "antitrust", "sec investigation", "lawsuit", "ban", "tariff", "sanction")),
    ("merger_acquisition", ("acquisition", "acquire", "merger", "takeover", "buyout")),
    ("product", ("launches", "unveils", "new product", "chip", "ai model", "data center")),
    ("cybersecurity", ("cyberattack", "data breach", "outage", "ransomware", "security incident")),
    ("analyst_action", ("upgrade", "downgrade", "price target", "initiates coverage")),
    ("management", ("ceo resigns", "chief executive", "cfo resigns", "appoints ceo")),
)

_BULLISH = (
    "beats estimates", "beat expectations", "raises guidance", "record revenue", "upgrade", "surges",
    "strong demand", "approval", "partnership", "buyback", "dividend increase", "rate cut",
)
_BEARISH = (
    "misses estimates", "missed expectations", "cuts guidance", "downgrade", "investigation", "lawsuit",
    "data breach", "outage", "recall", "layoffs", "rate hike", "tariff", "ban", "warning",
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _clean(value: str | None) -> str:
    return _SPACE_RE.sub(" ", unescape(_TAG_RE.sub(" ", value or ""))).strip()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _content_hash(item: MarketEventInput) -> str:
    canonical = "|".join(
        (
            item.source.strip().lower(),
            _clean(item.title).lower(),
            item.url.strip().lower(),
            item.published_at.isoformat() if item.published_at else "",
        )
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _identify_symbols(text: str, supplied: list[str]) -> list[str]:
    lowered = f" {_clean(text).lower()} "
    found = {symbol.strip().upper() for symbol in supplied if symbol.strip()}
    for symbol, aliases in _SYMBOL_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias.lower())
            if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered):
                found.add(symbol)
                break
    if found.intersection(_NASDAQ_COMPONENTS):
        found.add("NAS100")
    return sorted(found)


def _event_type(text: str) -> str:
    lowered = text.lower()
    for event_type, patterns in _EVENT_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return event_type
    return "general"


def _sentiment(text: str) -> tuple[str, float]:
    lowered = text.lower()
    positive = sum(1 for term in _BULLISH if term in lowered)
    negative = sum(1 for term in _BEARISH if term in lowered)
    total = positive + negative
    if total == 0:
        return "neutral", 0.0
    score = max(-1.0, min(1.0, (positive - negative) / total))
    if positive and negative:
        return "mixed", score
    return ("bullish", score) if score > 0 else ("bearish", score)


def _credibility(source: str, url: str) -> float:
    domain = (urlparse(url).hostname or "").lower()
    label = source.lower()
    if domain.endswith(".gov") or "federal reserve" in label or "sec.gov" in domain:
        return 0.98
    if any(name in domain or name in label for name in ("reuters", "apnews", "bloomberg", "financial times")):
        return 0.9
    if any(name in domain or name in label for name in ("nasdaq", "cnbc", "marketwatch", "yahoo finance")):
        return 0.76
    if any(name in domain or name in label for name in ("reddit", "stocktwits", "x.com", "twitter")):
        return 0.35
    return 0.55


def _impact(event_type: str, credibility: float, symbols: list[str]) -> str:
    if event_type in {"earnings", "guidance", "monetary_policy", "regulation", "merger_acquisition", "cybersecurity"}:
        return "high" if credibility >= 0.7 else "medium"
    if symbols and credibility >= 0.65:
        return "medium"
    return "low"


def _horizon(event_type: str) -> str:
    if event_type in {"cybersecurity", "monetary_policy", "analyst_action"}:
        return "immediate"
    if event_type in {"earnings", "guidance", "regulation", "merger_acquisition", "management"}:
        return "swing"
    if event_type == "product":
        return "long_term"
    return "intraday"


def classify_event(item: MarketEventInput) -> MarketEventRecord:
    title = _clean(item.title)
    summary = _clean(item.summary)
    combined = f"{title}. {summary}"
    symbols = _identify_symbols(combined, item.symbols)
    event_type = _event_type(combined)
    sentiment, sentiment_score = _sentiment(combined)
    credibility = _credibility(item.source, item.url)
    return MarketEventRecord(
        id=0,
        content_hash=_content_hash(item),
        source=item.source.strip(),
        title=title,
        summary=summary,
        url=item.url.strip(),
        published_at=item.published_at,
        ingested_at=_now(),
        symbols=symbols,
        event_type=event_type,
        sentiment=sentiment,  # type: ignore[arg-type]
        sentiment_score=sentiment_score,
        impact=_impact(event_type, credibility, symbols),  # type: ignore[arg-type]
        horizon=_horizon(event_type),  # type: ignore[arg-type]
        credibility_score=credibility,
        classification_version=CLASSIFICATION_VERSION,
        raw=item.raw,
    )


def ingest_events(items: list[MarketEventInput]) -> tuple[list[MarketEventRecord], int]:
    inserted: list[MarketEventRecord] = []
    duplicates = 0
    for item in items:
        event, created = insert_market_event(classify_event(item))
        if created:
            inserted.append(event)
        else:
            duplicates += 1
    return inserted, duplicates


def _feed_name(url: str) -> str:
    return urlparse(url).hostname or "rss"


def _text(node, names: tuple[str, ...]) -> str:
    for name in names:
        child = node.find(name)
        if child is not None and child.text:
            return child.text
        for candidate in list(node):
            if candidate.tag.rsplit("}", 1)[-1] == name and candidate.text:
                return candidate.text
    return ""


def fetch_rss(url: str, timeout: float = 8.0) -> list[MarketEventInput]:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Event source must be an absolute HTTP(S) URL.")
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "TradeAgent-V2/2.0 market-intelligence"},
    )
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("Event source response exceeds the 2 MB safety limit.")
    root = ElementTree.fromstring(response.content)
    nodes = root.findall(".//item")
    if not nodes:
        nodes = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "entry"]
    source = _feed_name(url)
    items: list[MarketEventInput] = []
    for node in nodes[:100]:
        title = _text(node, ("title",))
        if not _clean(title):
            continue
        link = _text(node, ("link", "guid"))
        if not link:
            link_node = next((child for child in list(node) if child.tag.rsplit("}", 1)[-1] == "link"), None)
            link = str(link_node.attrib.get("href", "")) if link_node is not None else ""
        items.append(
            MarketEventInput(
                source=source,
                title=title,
                summary=_text(node, ("description", "summary", "content")),
                url=link,
                published_at=_parse_datetime(_text(node, ("pubDate", "published", "updated"))),
                raw={"feed_url": url},
            )
        )
    return items


def configured_feed_urls() -> list[str]:
    raw = os.getenv("MARKET_NEWS_RSS_URLS", "")
    return list(dict.fromkeys(url.strip() for url in raw.split(";") if url.strip()))[:8]


def detect_abnormal_events(inserted: list[MarketEventRecord]) -> int:
    if not inserted:
        return 0
    recent = list_market_events(500)
    now = _now()
    last_15m: Counter[str] = Counter()
    prior_2h: Counter[str] = Counter()
    directions: dict[str, set[str]] = defaultdict(set)
    for event in recent:
        age = now - event.ingested_at
        for symbol in event.symbols:
            if age <= timedelta(minutes=15):
                last_15m[symbol] += 1
                if event.credibility_score >= 0.7 and event.sentiment in {"bullish", "bearish"}:
                    directions[symbol].add(event.sentiment)
            elif age <= timedelta(hours=2):
                prior_2h[symbol] += 1

    created_count = 0
    bucket = now.strftime("%Y%m%d%H") + str(now.minute // 15)
    for symbol in sorted({symbol for event in inserted for symbol in event.symbols}):
        current = last_15m[symbol]
        baseline_per_15m = prior_2h[symbol] / 7.0
        if current >= 3 and current >= max(3.0, baseline_per_15m * 2.5):
            _, created = add_event_alert(
                alert_key=f"news_velocity:{symbol}:{bucket}",
                alert_type="news_velocity",
                symbol=symbol,
                severity="warning",
                summary=f"Abnormal news velocity detected for {symbol}: {current} events in 15 minutes.",
                details={"events_15m": current, "baseline_per_15m": baseline_per_15m},
            )
            if created:
                created_count += 1
                log_incident(
                    "warning",
                    "abnormal_news_velocity",
                    f"Abnormal news velocity detected for {symbol}",
                    {"events_15m": current, "baseline_per_15m": baseline_per_15m},
                )
        if directions[symbol] == {"bullish", "bearish"}:
            _, created = add_event_alert(
                alert_key=f"conflicting_news:{symbol}:{bucket}",
                alert_type="conflicting_evidence",
                symbol=symbol,
                severity="warning",
                summary=f"Conflicting high-credibility event sentiment detected for {symbol}.",
                details={"sentiments": sorted(directions[symbol])},
            )
            created_count += int(created)
    return created_count


def refresh_configured_feeds() -> EventRefreshResponse:
    urls = configured_feed_urls()
    result = EventRefreshResponse(ok=True, configured_sources=len(urls))
    try:
        timeout_value = float(os.getenv("MARKET_NEWS_TIMEOUT_SEC", "8"))
    except ValueError:
        timeout_value = 8.0
    timeout = max(2.0, min(30.0, timeout_value))
    inserted_all: list[MarketEventRecord] = []
    for url in urls:
        started = _now()
        fetched = 0
        inserted_count = 0
        error = ""
        try:
            items = fetch_rss(url, timeout=timeout)
            fetched = len(items)
            inserted, duplicates = ingest_events(items)
            inserted_count = len(inserted)
            inserted_all.extend(inserted)
            result.fetched_items += fetched
            result.inserted_events += inserted_count
            result.duplicate_events += duplicates
        except Exception as exc:
            error = str(exc)
            result.ok = False
            result.errors.append(f"{_feed_name(url)}: {error}")
        finally:
            add_event_source_run(
                started_at=started,
                completed_at=_now(),
                source=url,
                fetched_items=fetched,
                inserted_events=inserted_count,
                error=error,
            )
    result.alerts_created = detect_abnormal_events(inserted_all)
    return result
