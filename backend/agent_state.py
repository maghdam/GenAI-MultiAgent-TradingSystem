# backend/agent_state.py
from collections import deque
from typing import List, Dict, Any

_SIGNALS: deque[Dict[str, Any]] = deque(maxlen=200)

def push_signal(sig: Dict[str, Any]) -> None:
    _SIGNALS.appendleft(sig)

def recent_signals(n: int = 50) -> List[Dict[str, Any]]:
    return list(_SIGNALS)[:n]


# --- bar-close bookkeeping -------------------------------------------------
_LAST_BAR_TS: dict[tuple[str, str], int] = {}

def _key(sym: str, tf: str) -> tuple[str, str]:
    return (sym.upper(), tf.upper())

def get_last_bar_ts(sym: str, tf: str) -> int | None:
    return _LAST_BAR_TS.get(_key(sym, tf))

def set_last_bar_ts(sym: str, tf: str, ts: int) -> None:
    _LAST_BAR_TS[_key(sym, tf)] = int(ts)
