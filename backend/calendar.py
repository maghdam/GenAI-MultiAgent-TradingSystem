import json
import os
import time
from pathlib import Path
from typing import Optional, Literal

Impact = Literal["high", "medium", "low", "unknown"]


def _env_event():
    try:
        ts = os.getenv("CALENDAR_NEXT_TS")
        if not ts:
            return None
        ts_val = float(ts)
        title = os.getenv("CALENDAR_NEXT_TITLE") or "High-impact event"
        impact = os.getenv("CALENDAR_NEXT_IMPACT") or "high"
        return {
            "ts": ts_val,
            "title": title,
            "impact": impact if impact in ("high", "medium", "low") else "high",
            "source": "env",
        }
    except Exception:
        return None


def _file_event() -> Optional[dict]:
    try:
        path = Path(__file__).parent / "data" / "calendar.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        now = time.time()
        candidates = [
            e for e in data if isinstance(e, dict) and e.get("ts") and float(e["ts"]) > now and e.get("impact") in ("high", "medium", "low")
        ]
        if not candidates:
            return None
        nxt = sorted(candidates, key=lambda e: e["ts"])[0]
        return {
            "ts": float(nxt["ts"]),
            "title": nxt.get("title") or "Event",
            "impact": nxt.get("impact", "high"),
            "source": nxt.get("source") or "file",
        }
    except Exception:
        return None


def get_next_event() -> Optional[dict]:
    """Return the next upcoming event (ts unix seconds). Sources: env, then optional data/calendar.json."""
    evt = _env_event()
    if evt:
        return evt
    return _file_event()
