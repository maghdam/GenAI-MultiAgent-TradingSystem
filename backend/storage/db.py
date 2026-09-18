from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import local
from typing import Iterator

from backend.config import SETTINGS


_LOCAL = local()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _get_connection() -> sqlite3.Connection:
    conn = getattr(_LOCAL, "connection", None)
    if conn is None:
        _ensure_parent(SETTINGS.db_path)
        conn = sqlite3.connect(SETTINGS.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _LOCAL.connection = conn
    return conn


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    yield _get_connection()


def init_db() -> None:
    with get_db() as db:
        cur = db.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                level TEXT NOT NULL,
                code TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                reasons_json TEXT NOT NULL,
                context_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                quantity REAL NOT NULL,
                status TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                stop_loss REAL,
                take_profit REAL,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                exit_price REAL,
                realized_pnl REAL NOT NULL DEFAULT 0,
                unrealized_pnl REAL NOT NULL DEFAULT 0,
                close_reason TEXT,
                account_currency TEXT NOT NULL DEFAULT 'USD',
                cash_per_price_unit_per_lot REAL NOT NULL DEFAULT 1.0,
                instrument_spec_source TEXT NOT NULL DEFAULT 'legacy'
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS order_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                intent_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                quantity REAL,
                rationale TEXT NOT NULL,
                details_json TEXT NOT NULL,
                decision_id INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                correlation_id TEXT NOT NULL UNIQUE,
                decision_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                outcome TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS order_intent_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                details_json TEXT NOT NULL,
                FOREIGN KEY(intent_id) REFERENCES order_intents(id)
            )
            """
        )
        existing_intent_columns = {row[1] for row in cur.execute("PRAGMA table_info(order_intents)").fetchall()}
        if "decision_id" not in existing_intent_columns:
            cur.execute("ALTER TABLE order_intents ADD COLUMN decision_id INTEGER")
        existing_position_columns = {row[1] for row in cur.execute("PRAGMA table_info(paper_positions)").fetchall()}
        if "account_currency" not in existing_position_columns:
            cur.execute("ALTER TABLE paper_positions ADD COLUMN account_currency TEXT NOT NULL DEFAULT 'USD'")
        if "cash_per_price_unit_per_lot" not in existing_position_columns:
            cur.execute("ALTER TABLE paper_positions ADD COLUMN cash_per_price_unit_per_lot REAL NOT NULL DEFAULT 1.0")
        if "instrument_spec_source" not in existing_position_columns:
            cur.execute("ALTER TABLE paper_positions ADD COLUMN instrument_spec_source TEXT NOT NULL DEFAULT 'legacy'")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_decision_records_symbol_created
            ON decision_records(symbol, created_at DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intent_transitions_intent
            ON order_intent_transitions(intent_id, id)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                url TEXT NOT NULL,
                published_at TEXT,
                ingested_at TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                impact TEXT NOT NULL,
                horizon TEXT NOT NULL,
                credibility_score REAL NOT NULL,
                classification_version TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS event_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                severity TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS event_source_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_items INTEGER NOT NULL,
                inserted_events INTEGER NOT NULL,
                error TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_events_published
            ON market_events(published_at DESC, id DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_event_alerts_created
            ON event_alerts(created_at DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS event_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                market_symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                status TEXT NOT NULL,
                reference_at TEXT,
                target_at TEXT,
                reference_price REAL,
                target_price REAL,
                forward_return_pct REAL,
                max_favorable_excursion_pct REAL,
                max_adverse_excursion_pct REAL,
                predicted_direction TEXT NOT NULL,
                realized_direction TEXT NOT NULL,
                direction_hit INTEGER,
                brier_score REAL,
                threshold_pct REAL NOT NULL,
                reason TEXT NOT NULL,
                computed_at TEXT NOT NULL,
                UNIQUE(event_id, symbol, horizon),
                FOREIGN KEY(event_id) REFERENCES market_events(id)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_event_outcomes_status
            ON event_outcomes(status, event_id, symbol)
            """
        )
        existing_outcome_columns = {row[1] for row in cur.execute("PRAGMA table_info(event_outcomes)").fetchall()}
        if "market_symbol" not in existing_outcome_columns:
            cur.execute("ALTER TABLE event_outcomes ADD COLUMN market_symbol TEXT NOT NULL DEFAULT ''")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS confluence_shadow_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                analysis_created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                original_signal TEXT NOT NULL,
                original_confidence REAL NOT NULL,
                shadow_signal TEXT NOT NULL,
                shadow_confidence REAL NOT NULL,
                confidence_adjustment REAL NOT NULL,
                action TEXT NOT NULL,
                target_horizon TEXT NOT NULL,
                event_score REAL NOT NULL,
                eligible_event_count INTEGER NOT NULL,
                event_ids_json TEXT NOT NULL,
                original_would_pass INTEGER NOT NULL,
                shadow_would_pass INTEGER NOT NULL,
                rationale TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                execution_unchanged INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_confluence_shadow_symbol_created
            ON confluence_shadow_records(symbol, created_at DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                position_id INTEGER,
                intent_id INTEGER,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_bars (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                bar_time TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY(symbol, timeframe, bar_time)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_bars_symbol_tf_time
            ON market_bars(symbol, timeframe, bar_time DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_lifecycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                version INTEGER NOT NULL,
                version_hash TEXT NOT NULL,
                stage TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                gates_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(strategy, version_hash),
                UNIQUE(strategy, version)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_lifecycle_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lifecycle_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                passed INTEGER NOT NULL,
                summary TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(lifecycle_id) REFERENCES strategy_lifecycles(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_lifecycle_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lifecycle_id INTEGER NOT NULL,
                from_stage TEXT NOT NULL,
                to_stage TEXT NOT NULL,
                operator TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(lifecycle_id) REFERENCES strategy_lifecycles(id)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_latest
            ON strategy_lifecycles(strategy, version DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_evidence
            ON strategy_lifecycle_evidence(lifecycle_id, id DESC)
            """
        )
        db.commit()


init_db()
