"""Durable exit identity and one-shot actions; no expiring locks or retries."""
import json
from datetime import datetime, timezone

from backend.storage.db import get_db


def bind_exit_session(intent_id, account_id, identity):
    if not account_id:
        raise ValueError("Exact Paper account identity required")
    encoded = json.dumps(identity, sort_keys=True)
    with get_db() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("INSERT OR IGNORE INTO alpaca_bracket_exit_sessions VALUES (?, ?, ?, ?)",
                       (intent_id, account_id, encoded, datetime.now(timezone.utc).isoformat()))
            row = db.execute("SELECT * FROM alpaca_bracket_exit_sessions WHERE intent_id = ?",
                             (intent_id,)).fetchone()
            if row["account_id"] != account_id or row["identity_json"] != encoded:
                raise ValueError("Exit account or bracket identity changed since first claim")
            db.commit()
        except BaseException:
            db.rollback()
            raise


def exit_actions(intent_id):
    with get_db() as db:
        rows = db.execute("SELECT action_key, request_json FROM alpaca_bracket_exit_actions WHERE intent_id = ?",
                          (intent_id,)).fetchall()
    return {row["action_key"]: json.loads(row["request_json"]) for row in rows}


def claim_exit_action(intent_id, key, request):
    """Commit before HTTP; an uncertain/crashed send is NEVER reclaimed."""
    with get_db() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            if db.execute("SELECT 1 FROM alpaca_bracket_exit_sessions WHERE intent_id = ?",
                          (intent_id,)).fetchone() is None:
                raise ValueError("Exit session has no durable identity")
            cursor = db.execute("INSERT OR IGNORE INTO alpaca_bracket_exit_actions VALUES (?, ?, ?, ?)",
                                (intent_id, key, json.dumps(request, sort_keys=True),
                                 datetime.now(timezone.utc).isoformat()))
            claimed = cursor.rowcount == 1
            db.commit()
            return claimed
        except BaseException:
            db.rollback()
            raise
