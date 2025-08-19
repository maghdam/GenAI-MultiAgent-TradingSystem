# backend/agent_state.py
from collections import deque
from typing import List, Dict, Any

_SIGNALS: deque[Dict[str, Any]] = deque(maxlen=200)

def push_signal(sig: Dict[str, Any]) -> None:
    _SIGNALS.appendleft(sig)

def recent_signals(n: int = 50) -> List[Dict[str, Any]]:
    return list(_SIGNALS)[:n]
