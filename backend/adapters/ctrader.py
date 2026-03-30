from __future__ import annotations

from datetime import UTC, datetime
import threading
from typing import Any, Dict, List

import backend.ctrader_client as ctd
import pandas as pd

from backend.domain.models import BrokerStatus, SymbolLimits
from backend.services.runtime_state import external_dependency_state, market_data_dependency_state


class CTraderBrokerAdapter:
    @staticmethod
    def _default_symbol_limits(symbol: str) -> SymbolLimits:
        min_api = 100
        step_api = 100
        max_api = 1_000_000
        return SymbolLimits(
            symbol=(symbol or "").strip().upper(),
            source="fallback",
            min_lots=min_api / 10_000,
            step_lots=step_api / 10_000,
            max_lots=max_api / 10_000,
            min_api_units=min_api,
            step_api_units=step_api,
            max_api_units=max_api,
            hard_min=False,
            hard_step=False,
        )

    def __init__(self) -> None:
        self._thread_lock = threading.Lock()
        self._thread_started = False
        self._symbol_cache: List[str] = []
        self._symbol_cache_count: int = 0

    def start_transport(self) -> bool:
        with self._thread_lock:
            if self._thread_started:
                return False
            threading.Thread(target=ctd.init_client, daemon=True).start()
            self._thread_started = True
            return True

    def transport_started(self) -> bool:
        return self._thread_started

    @staticmethod
    def connected() -> bool:
        try:
            return bool(ctd.is_connected())
        except Exception:
            return False

    @staticmethod
    def _account_id_value() -> int | None:
        raw = getattr(ctd, "ACCOUNT_ID", None)
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_time_value(value: Any):
        if value is None or pd.isna(value):
            return pd.NaT
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if ts.tzinfo is None:
            return ts.tz_localize(UTC)
        return ts.tz_convert(UTC)

    @classmethod
    def _normalize_time_column(cls, values):
        source = pd.Series(values)
        parsed = pd.Series(pd.to_datetime(source, utc=True, errors="coerce", format="ISO8601"), index=source.index)
        missing = parsed.isna() & source.notna()
        if missing.any():
            merged = parsed.astype("object")
            merged.loc[missing] = source.loc[missing].map(cls._parse_time_value)
            parsed = pd.Series(pd.to_datetime(merged, utc=True, errors="coerce"), index=source.index)
        return parsed

    def get_status(self) -> BrokerStatus:
        notes: List[str] = []
        connected = self.connected()
        authorized = bool(ctd.is_authorized())
        symbols_loaded = len(ctd.symbol_name_to_id or {})

        positions = []
        pending_orders = []
        if connected and authorized:
            try:
                snapshot = ctd.get_reconcile_snapshot()
                positions = snapshot.get("positions") or []
                pending_orders = snapshot.get("orders") or []
                if snapshot.get("error"):
                    notes.append(f"reconcile_unavailable: {snapshot['error']}")
            except Exception as exc:
                notes.append(f"reconcile_unavailable: {exc}")

        ready = connected and symbols_loaded > 0 and authorized
        market_data_ready = bool(market_data_dependency_state.market_data_ready and ready)

        if not connected:
            notes.append("cTrader transport is not connected.")
        if not authorized:
            notes.append("cTrader account is not authorized.")
        if symbols_loaded == 0:
            notes.append("No broker symbols are loaded.")
        
        notes.extend(external_dependency_state.snapshot_notes())

        return BrokerStatus(
            connected=connected,
            socket_connected=connected,
            account_authorized=authorized,
            auth_error=ctd.get_auth_error(),
            last_auth_attempt_at=ctd.get_last_auth_attempt(),
            symbols_loaded=symbols_loaded,
            open_positions=len(positions),
            pending_orders=len(pending_orders),
            ready=ready,
            market_data_ready=market_data_ready,
            broker_mode=str(getattr(ctd, "HOST_TYPE", "unknown")),
            account_id=self._account_id_value(),
            notes=notes,
        )

    def list_positions(self) -> List[Dict[str, Any]]:
        rows = ctd.get_open_positions() or []
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "symbol": row.get("symbol_name"),
                    "direction": row.get("direction"),
                    "volume_lots": row.get("volume_lots"),
                    "entry_price": row.get("entry_price"),
                    "stop_loss": row.get("stop_loss"),
                    "take_profit": row.get("take_profit"),
                    "position_id": row.get("position_id"),
                }
            )
        return out

    def list_symbols(self) -> List[str]:
        # Quick check if symbols changed via length
        # (Usually enough for broker symbol list changes)
        current_id_map = ctd.symbol_name_to_id or {}
        count = len(current_id_map)
        
        if self._symbol_cache and self._symbol_cache_count == count and count > 0:
            return self._symbol_cache

        symbols = sorted(current_id_map.keys())
        if not symbols:
            # Fallback symbols don't need caching
            return sorted(ctd.FALLBACK_SYMBOLS)

        self._symbol_cache = symbols
        self._symbol_cache_count = count
        return symbols

    def get_symbol_limits(self, symbol: str) -> SymbolLimits:
        sym = (symbol or "").strip().upper()
        if not sym:
            return self._default_symbol_limits(sym)

        symbol_id = (ctd.symbol_name_to_id or {}).get(sym)
        if symbol_id is None:
            return self._default_symbol_limits(sym)

        min_api = int(ctd.symbol_min_volume_map.get(symbol_id) or 100)
        step_api = int(ctd.symbol_step_volume_map.get(symbol_id) or min_api or 100)
        max_api = int(ctd.symbol_max_volume_map.get(symbol_id) or 1_000_000)
        max_api = max(max_api, min_api)
        step_api = max(step_api, 1)

        return SymbolLimits(
            symbol=sym,
            source="broker",
            min_lots=min_api / 10_000,
            step_lots=step_api / 10_000,
            max_lots=max_api / 10_000,
            min_api_units=min_api,
            step_api_units=step_api,
            max_api_units=max_api,
            hard_min=bool(ctd.symbol_min_verified.get(symbol_id)),
            hard_step=bool(ctd.symbol_step_verified.get(symbol_id)),
        )

    def _generate_mock_bars(self, symbol: str, timeframe: str, num_bars: int):
        import numpy as np
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC).replace(tzinfo=None)
        intervals = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        minutes = intervals.get(timeframe, 5)

        # Base price based on symbol
        base_price = 2500.0 if "XAU" in symbol else 1.1000 if "EUR" in symbol else 150.0
        volatility = base_price * 0.002

        times = [now - timedelta(minutes=minutes * i) for i in range(num_bars)]
        times.reverse()

        prices = [base_price]
        for _ in range(num_bars - 1):
            change = np.random.normal(0, volatility)
            prices.append(prices[-1] + change)

        data = []
        for i, t in enumerate(times):
            p = prices[i]
            # Random OHLC
            noise = volatility * 0.5
            o = p + np.random.uniform(-noise, noise)
            c = p + np.random.uniform(-noise, noise)
            h = max(o, c) + np.random.uniform(0, noise)
            l = min(o, c) - np.random.uniform(0, noise)
            data.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": np.random.randint(100, 1000)})
        
        df = pd.DataFrame(data).set_index("time")
        return df, float(df["close"].iloc[-1])

    def _get_real_bars(self, symbol: str, timeframe: str, num_bars: int):
        sym = (symbol or "").strip().upper()
        tf = (timeframe or "M5").strip().upper()
        if not sym:
            raise ValueError("Symbol is required")
        if tf not in {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}:
            raise ValueError(f"Invalid timeframe '{tf}'")

        if not ctd.is_connected():
            raise RuntimeError("cTrader transport is not connected.")

        if not ctd.is_authorized():
            raise RuntimeError("cTrader account is not authorized.")

        if sym not in (ctd.symbol_name_to_id or {}):
            raise RuntimeError(f"Symbol '{sym}' is not loaded from broker.")

        rows = ctd.get_ohlc_data(symbol=sym, tf=tf, n=num_bars)
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"No market data available for {sym}:{tf}")

        df["time"] = self._normalize_time_column(df["time"])
        df = df.dropna(subset=["time"]).set_index("time").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            raise RuntimeError(f"No market data available for {sym}:{tf}")
        if num_bars and len(df) > num_bars:
            df = df.iloc[-num_bars:]
        live_price = float(df["close"].iloc[-1]) if not df.empty else None
        return df, live_price

    def get_bars(self, symbol: str, timeframe: str, num_bars: int):
        return self._get_real_bars(symbol, timeframe, num_bars)

    def get_market_data_status(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        sym = (symbol or "").strip().upper()
        tf = (timeframe or "M5").strip().upper()
        checked_at = datetime.now(UTC).replace(tzinfo=None).isoformat()
        status: Dict[str, Any] = {
            "checked_at": checked_at,
            "symbol": sym,
            "timeframe": tf,
            "ok": False,
            "reason": "",
        }

        try:
            df, _ = self._get_real_bars(sym, tf, 5)
            if df is not None and not df.empty:
                status["ok"] = True
                status["reason"] = f"Fetched {len(df)} bars"
                return status
        except Exception as exc:
            status["reason"] = str(exc)
            return status

        status["reason"] = f"No market data available for {sym}:{tf}"
        return status


adapter = CTraderBrokerAdapter()
