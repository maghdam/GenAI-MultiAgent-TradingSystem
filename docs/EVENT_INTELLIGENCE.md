# Event intelligence

The event-intelligence subsystem is a read-only evidence pipeline. It cannot create orders or bypass the deterministic strategy, sizing, risk, and execution boundaries.

## Pipeline

```text
RSS/Atom or API ingest
  -> text normalization
  -> SHA-256 content deduplication
  -> source credibility scoring
  -> ticker and Nasdaq-context mapping
  -> event taxonomy
  -> sentiment, impact, and horizon classification
  -> persistent evidence
  -> abnormal velocity/conflict alerts
  -> Market Intelligence drivers
```

The first classifier is deliberately deterministic and versioned as `deterministic-v1`. This makes replay and evaluation possible before an LLM classifier is introduced.

## Configure feeds

Add semicolon-separated RSS or Atom URLs to `backend/.env`:

```dotenv
MARKET_NEWS_RSS_URLS=https://source-one.example/feed.xml;https://source-two.example/markets.atom
MARKET_NEWS_TIMEOUT_SEC=8
MARKET_NEWS_AUTO_REFRESH_SEC=300
APP_START_EVENT_INTELLIGENCE_ON_BOOT=1
EVENT_CALIBRATION_AUTO=1
EVENT_CALIBRATION_MIN_SAMPLES=30
```

No external source is contacted when the list is empty. The UI reports that no feeds are configured.
When sources are configured, the backend refreshes them automatically. The interval is clamped between 30 seconds and 24 hours. Set `APP_START_EVENT_INTELLIGENCE_ON_BOOT=0` to keep manual refresh only.

## APIs

```text
POST /api/market/events/refresh
GET  /api/market/events?limit=50&symbol=NAS100
GET  /api/market/event-alerts?limit=20
POST /api/market/events/ingest
GET  /api/market/intelligence
```

The ingest endpoint accepts a JSON array containing `source`, `title`, and optional `summary`, `url`, `published_at`, `symbols`, and `raw` fields. Duplicate content returns as a duplicate rather than creating another event.

## Alerts

The deterministic detector currently produces:

- `news_velocity`: at least three symbol-linked events in 15 minutes and a material increase over the preceding baseline.
- `conflicting_evidence`: credible bullish and bearish evidence for the same symbol in the current window.

Alerts use stable time-bucket keys, so repeated refreshes cannot create duplicate notifications. They are also surfaced through the existing incident system.

## Nasdaq context

Events mentioning major Nasdaq-100 constituents are linked to the constituent and `NAS100`. This provides index context while preserving the originating stock evidence.

Event sentiment is displayed as contextual evidence. It does not modify strategy confidence or qualify a trade yet. That fusion should be added only after event labels are evaluated against subsequent market moves.
