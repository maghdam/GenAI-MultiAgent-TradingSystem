from __future__ import annotations

from datetime import UTC, datetime

from backend.domain.models import MarketEventInput
from backend.services.event_intelligence import classify_event, detect_abnormal_events, fetch_rss, ingest_events
from backend.storage.repositories import list_event_alerts, list_market_events


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
