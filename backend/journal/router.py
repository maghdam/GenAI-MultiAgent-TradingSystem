from fastapi import APIRouter, HTTPException, Query
from . import db as journal_db

router = APIRouter()

@router.get("/trades")
async def get_trades(limit: int = Query(100, gt=0, le=1000)):
    """Returns a list of the most recent trades from the journal."""
    try:
        trades = journal_db.get_all_trades(limit=limit)
        return trades
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve journal entries: {e}")
