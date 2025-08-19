## symbol_fetcher.py

from backend.ctrader_client import symbol_map

def get_available_symbols():
    """Fetch all available trading symbols from cTrader."""
    return list(symbol_map.values())
