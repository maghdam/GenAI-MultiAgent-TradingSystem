# backend/agent_state.py
from collections import deque
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, Dict, List
import json
import os

_STORE_PATH = Path(os.getenv("AGENT_STATE_PATH", "backend/data/agent_state.json"))
_STORE_LOCK = Lock()

_SIGNALS: deque[Dict[str, Any]] = deque(maxlen=200)
_LAST_BAR_TS: dict[tuple[str, str], int] = {}
_TASK_STATUS: dict[tuple[str, str], Dict[str, Any]] = {}
_STATUS_LOCK = Lock()


def _key(sym: str, tf: str) -> tuple[str, str]:
    return (sym.upper(), tf.upper())


def _serialize_state() -> dict[str, Any]:
    return {
        "signals": list(_SIGNALS),
        "last_bar_ts": {
            f"{sym}|{tf}": ts for (sym, tf), ts in _LAST_BAR_TS.items()
        },
    }


def _deserialize_state(payload: dict[str, Any]) -> None:
    signals = payload.get("signals") or []
    _SIGNALS.clear()
    for sig in signals[: _SIGNALS.maxlen]:
        _SIGNALS.append(sig)

    _LAST_BAR_TS.clear()
    for key, ts in (payload.get("last_bar_ts") or {}).items():
        if not isinstance(key, str) or "|" not in key:
            continue
        sym, tf = key.split("|", 1)
        try:
            _LAST_BAR_TS[(sym, tf)] = int(ts)
        except (TypeError, ValueError):
            continue


def _load_state() -> None:
    if not _STORE_PATH.exists():
        return
    try:
        with _STORE_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return
    _deserialize_state(payload if isinstance(payload, dict) else {})


def _ensure_dir_exists() -> None:
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _save_state() -> None:
    data = _serialize_state()
    _ensure_dir_exists()
    tmp_path = _STORE_PATH.with_suffix(_STORE_PATH.suffix + ".tmp")
    try:
        with _STORE_LOCK:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(_STORE_PATH)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


_load_state()


def push_signal(sig: Dict[str, Any]) -> None:
    _SIGNALS.appendleft(sig)
    _save_state()


def recent_signals(n: int = 50) -> List[Dict[str, Any]]:
    return list(_SIGNALS)[:n]


# --- bar-close bookkeeping -------------------------------------------------

def get_last_bar_ts(sym: str, tf: str) -> int | None:
    return _LAST_BAR_TS.get(_key(sym, tf))

def set_last_bar_ts(sym: str, tf: str, ts: int) -> None:
    _LAST_BAR_TS[_key(sym, tf)] = int(ts)
    _save_state()

def clear_last_bar_ts(sym: str, tf: str) -> None:
    """Force a re-scan on next agent loop by clearing the last-seen bar."""
    k = _key(sym, tf)
    if k in _LAST_BAR_TS:
        del _LAST_BAR_TS[k]
        _save_state()


# --- task status tracking --------------------------------------------------

def update_task_status(sym: str, tf: str, **fields: Any) -> None:
    key = _key(sym, tf)
    now = time()
    with _STATUS_LOCK:
        current = _TASK_STATUS.get(key, {})
        current.update(fields)
        current["symbol"], current["timeframe"] = key
        current.setdefault("first_seen_ts", now)
        current["updated_ts"] = now
        _TASK_STATUS[key] = current


def clear_task_status(sym: str, tf: str) -> None:
    key = _key(sym, tf)
    with _STATUS_LOCK:
        _TASK_STATUS.pop(key, None)


def all_task_status() -> List[Dict[str, Any]]:
    with _STATUS_LOCK:
        return [dict(v) for v in _TASK_STATUS.values()]
