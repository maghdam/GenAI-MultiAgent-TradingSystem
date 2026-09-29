from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from backend.domain.models import MarketEventInput
from backend.services.event_intelligence import (
    build_event_source_status,
    classify_event,
    detect_abnormal_events,
    fetch_rss,
    ingest_events,
)
from backend.storage.repositories import add_event_source_run, list_event_alerts, list_market_events


def _item(title: str, *, url: str = "https://www.reuters.com/technology/story", summary: str = "") -> MarketEventInput:
    return MarketEventInput(
        source="Reuters",
        title=title,
        summary=summary,
        url=url,
        published_at=datetime.now(UTC),
    )


def test_classify_event_links_nasdaq_component_and_context() -> None:
    event = classify_event(
        _item("Nvidia beats estimates and raises guidance", summary="Strong data center demand drives record revenue.")
    )

    assert event.symbols == ["NAS100", "NVDA"]
    assert event.event_type == "earnings"
    assert event.sentiment == "bullish"
    assert event.sentiment_score > 0
    assert event.impact == "high"
    assert event.credibility_score == 0.9


def test_ingestion_is_content_hash_idempotent() -> None:
    item = _item("Microsoft launches new AI product")

    first, first_duplicates = ingest_events([item])
    second, second_duplicates = ingest_events([item])

    assert len(first) == 1
    assert first_duplicates == 0
    assert second == []
    assert second_duplicates == 1
    assert len(list_market_events(10)) == 1


def test_abnormal_velocity_and_conflicting_evidence_create_deduplicated_alerts() -> None:
    inserted, _ = ingest_events(
        [
            _item("Nvidia beats estimates and raises guidance", url="https://www.reuters.com/a"),
            _item("Nvidia faces investigation warning", url="https://www.reuters.com/b"),
            _item("NVDA upgrade follows strong demand", url="https://www.reuters.com/c"),
        ]
    )

    created = detect_abnormal_events(inserted)
    repeated = detect_abnormal_events(inserted)
    alerts = list_event_alerts(10)

    assert created == 4
    assert repeated == 0
    assert {alert.alert_type for alert in alerts} == {"news_velocity", "conflicting_evidence"}
    assert all(alert.symbol in {"NVDA", "NAS100"} for alert in alerts)


def test_fetch_rss_normalizes_items(monkeypatch) -> None:
    xml = b"""<?xml version='1.0'?>
    <rss><channel><item>
      <title>AMD raises guidance</title>
      <description><![CDATA[<p>Strong demand.</p>]]></description>
      <link>https://example.com/amd</link>
      <pubDate>Sun, 19 Jul 2026 10:30:00 GMT</pubDate>
    </item></channel></rss>"""

    class Response:
        content = xml

        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr("backend.services.event_intelligence.requests.get", lambda *args, **kwargs: Response())

    items = fetch_rss("https://example.com/feed.xml")

    assert len(items) == 1
    assert items[0].source == "example.com"
    assert items[0].title == "AMD raises guidance"
    assert "Strong demand" in items[0].summary
    assert items[0].published_at is not None

def test_event_source_status_is_explicit_when_no_feeds_are_configured(monkeypatch) -> None:
    monkeypatch.delenv("MARKET_NEWS_RSS_URLS", raising=False)

    status = build_event_source_status()

    assert status["status"] == "not_configured"
    assert status["research_only"] is True
    assert status["configured_sources"] == 0
    assert status["sources"] == []


def test_event_source_status_reports_never_refreshed_configured_source(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_NEWS_RSS_URLS", "https://example.com/feed.xml")

    status = build_event_source_status()

    assert status["status"] == "missing"
    assert status["configured_sources"] == 1
    assert status["sources"][0]["status"] == "never_refreshed"
    assert status["sources"][0]["last_completed_at"] is None


def test_event_source_status_distinguishes_fresh_stale_and_failed_runs(monkeypatch) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    fresh_url = "https://fresh.example/feed.xml"
    stale_url = "https://stale.example/feed.xml"
    failed_url = "https://failed.example/feed.xml"
    monkeypatch.setenv("MARKET_NEWS_RSS_URLS", f"{fresh_url};{stale_url};{failed_url}")
    monkeypatch.setenv("MARKET_NEWS_STALE_AFTER_SEC", "3600")

    add_event_source_run(
        started_at=now - timedelta(minutes=5),
        completed_at=now - timedelta(minutes=4),
        source=fresh_url,
        fetched_items=4,
        inserted_events=2,
    )
    add_event_source_run(
        started_at=now - timedelta(hours=3),
        completed_at=now - timedelta(hours=2),
        source=stale_url,
        fetched_items=3,
        inserted_events=1,
    )
    add_event_source_run(
        started_at=now - timedelta(minutes=3),
        completed_at=now - timedelta(minutes=2),
        source=failed_url,
        fetched_items=0,
        inserted_events=0,
        error="upstream timeout",
    )

    status = build_event_source_status()
    by_source = {item["source"]: item for item in status["sources"]}

    assert status["status"] == "degraded"
    assert by_source[fresh_url]["status"] == "healthy"
    assert by_source[stale_url]["status"] == "stale"
    assert by_source[failed_url]["status"] == "failed"
    assert by_source[failed_url]["error"] == "upstream timeout"


def test_event_source_status_endpoint_is_read_only_and_truthful(monkeypatch) -> None:
    monkeypatch.setenv("APP_START_CTRADER_ON_BOOT", "0")
    monkeypatch.setenv("APP_WARM_OLLAMA_ON_BOOT", "0")
    monkeypatch.setenv("APP_START_LEGACY_CONTROLLER_ON_BOOT", "0")
    monkeypatch.setattr(
        "backend.api.router.build_event_source_status",
        lambda: {
            "status": "stale",
            "research_only": True,
            "configured_sources": 1,
            "stale_after_seconds": 3600,
            "sources": [{"source": "https://example.com/feed.xml", "status": "stale"}],
        },
    )
    from backend.app import app

    with TestClient(app) as client:
        response = client.get("/api/market/events/status")

    assert response.status_code == 200
    assert response.json()["status"] == "stale"
    assert response.json()["research_only"] is True

