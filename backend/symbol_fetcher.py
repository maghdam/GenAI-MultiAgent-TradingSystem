# backend/symbol_fetcher.py
import backend.ctrader_client as ctd

def get_available_symbols():
    """Return list of symbol names (e.g., 'XAUUSD'). Empty until cTrader connects."""
    try:
        return list(ctd.symbol_map.values())
    except Exception as e:
        print("[WARN] get_available_symbols:", e)
        return []
