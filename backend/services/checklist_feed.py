from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from backend.calendar import get_next_event
from backend.checklist import compute_auto_checklist
from backend.data_fetcher import ALLOWED_TF


def get_auto_checklist(*, tf: str = "M5", structure_tf: str = "H1") -> dict[str, Any]:
    tf_upper = (tf or "M5").upper()
    structure_tf_upper = (structure_tf or "H1").upper()
    if tf_upper not in ALLOWED_TF:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe '{tf_upper}'. Allowed: {sorted(ALLOWED_TF)}")
    if structure_tf_upper not in ALLOWED_TF:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid structure timeframe '{structure_tf_upper}'. Allowed: {sorted(ALLOWED_TF)}",
        )
    payload = compute_auto_checklist(bias_tf=tf_upper, structure_tf=structure_tf_upper, volume_tf=tf_upper)
    return payload.dict()


def get_calendar_next() -> dict[str, Any]:
    try:
        event = get_next_event()
        if not event:
            return {"ts": None, "title": None, "impact": "unknown", "source": None}
        return event
    except Exception as exc:
        return {"ts": None, "title": None, "impact": "unknown", "source": f"error: {exc}"}
