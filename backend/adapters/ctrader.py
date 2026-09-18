from __future__ import annotations

from datetime import UTC, datetime
import threading
import time
from typing import Any, Dict, List

import backend.ctrader_client as ctd
import pandas as pd

from backend.domain.models import BrokerStatus, InstrumentSpec, SymbolLimits
from backend.services.runtime_state import external_dependency_state, market_data_dependency_state


class CTraderBrokerAdapter:
    _USD_INDEX_ALIASES = ("US100", "NAS100", "USTEC", "NAS", "US500", "SPX", "US30", "DJ30")

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

    def demo_symbol_execution_readiness(self, symbol: str) -> tuple[bool, str]:
        """Return whether broker metadata is ready to safely size a demo order."""
        if not ctd.is_demo_account_confirmed():
            reason = ctd.get_account_verification_error() or "Connected cTrader account is not confirmed as demo."
            return False, reason

        sym = (symbol or "").strip().upper()
        symbol_id = (ctd.symbol_name_to_id or {}).get(sym)
        if symbol_id is None:
            return False, f"Broker symbol {sym!r} is not loaded yet."

        try:
            lot_size = float((ctd.symbol_lot_size_map or {}).get(symbol_id) or 0.0)
        except (TypeError, ValueError):
            lot_size = 0.0
        if lot_size <= 0:
            return False, f"Broker lotSize metadata is not loaded yet for {sym}."

        min_volume = (ctd.symbol_min_volume_map or {}).get(symbol_id)
        step_volume = (ctd.symbol_step_volume_map or {}).get(symbol_id)
        max_volume = (ctd.symbol_max_volume_map or {}).get(symbol_id)
        if min_volume is None or step_volume is None or max_volume is None:
            return False, f"Broker volume limits are not loaded yet for {sym}."

        return True, "Broker symbol contract metadata is ready."

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
        auth_error = ctd.get_auth_error()
        market_reason = str(market_data_dependency_state.last_reason or "")
        if authorized and "not authorized" in market_reason.lower():
            authorized = False
            auth_error = market_reason
            notes.append(market_reason)
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

        if not connected:
            notes.append("cTrader transport is not connected.")
        if not authorized:
            notes.append("cTrader account is not authorized.")
        if symbols_loaded == 0:
            notes.append("No broker symbols are loaded.")
        demo_confirmed = bool(ctd.is_demo_account_confirmed())
        account_type = (
            "demo"
            if ctd.ACCOUNT_IS_DEMO is True
            else ("live" if ctd.ACCOUNT_IS_DEMO is False else "unknown")
        )
        verification_error = ctd.get_account_verification_error()
        if verification_error and verification_error not in notes:
            notes.append(verification_error)
        if connected and not demo_confirmed:
            notes.append("Order execution is blocked until the connected account is confirmed as demo.")
        
        notes.extend(external_dependency_state.snapshot_notes())
        auth_note = next((note for note in notes if "not authorized" in note.lower()), "")
        if authorized and auth_note:
            authorized = False
            auth_error = auth_note

        ready = connected and symbols_loaded > 0 and authorized
        market_data_ready = bool(market_data_dependency_state.market_data_ready and ready)
        execution_ready = bool(ready and demo_confirmed)

        return BrokerStatus(
            connected=connected,
            socket_connected=connected,
            account_authorized=authorized,
            auth_error=auth_error,
            last_auth_attempt_at=ctd.get_last_auth_attempt(),
            symbols_loaded=symbols_loaded,
            open_positions=len(positions),
            pending_orders=len(pending_orders),
            ready=ready,
            market_data_ready=market_data_ready,
            broker_mode=str(getattr(ctd, "HOST_TYPE", "unknown")),
            account_id=self._account_id_value(),
            account_type=account_type,
            demo_account_confirmed=demo_confirmed,
            execution_ready=execution_ready,
            notes=notes,
        )

    def place_demo_market_order(
        self,
        *,
        symbol: str,
        direction: str,
        quantity_lots: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        client_msg_id: str | None = None,
    ) -> Dict[str, Any]:
        if not ctd.is_demo_account_confirmed():
            reason = ctd.get_account_verification_error() or "Connected cTrader account is not confirmed as demo."
            raise RuntimeError(f"Demo order blocked: {reason}")

        sym = (symbol or "").strip().upper()
        symbol_id = (ctd.symbol_name_to_id or {}).get(sym)
        if symbol_id is None:
            raise RuntimeError(f"Demo order blocked: broker symbol {sym!r} is unavailable.")

        side = "BUY" if direction == "long" else ("SELL" if direction == "short" else "")
        if not side:
            raise RuntimeError(f"Demo order blocked: unsupported direction {direction!r}.")

        volume = ctd.volume_lots_to_units(symbol_id, quantity_lots)
        deferred = ctd.place_order(
            client=ctd.client,
            account_id=ctd.ACCOUNT_ID,
            symbol_id=symbol_id,
            order_type="MARKET",
            side=side,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            client_msg_id=client_msg_id,
        )
        result = ctd.wait_for_deferred(deferred, timeout=20)
        if isinstance(result, dict) and result.get("status") in {"failed", "order_rejected"}:
            reason = result.get("error") or result.get("reject_reason") or result["status"]
            raise RuntimeError(f"cTrader demo order failed: {reason}")
        if not isinstance(result, dict):
            raise RuntimeError("cTrader demo order returned an unrecognized acknowledgement.")
        return {
            "status": "executed",
            "account_id": self._account_id_value(),
            "account_type": "demo",
            "symbol": sym,
            "symbol_id": symbol_id,
            "direction": direction,
            "quantity_lots": float(quantity_lots),
            "volume_api_units": volume,
            "position_id": result.get("position_id"),
            "ack": result.get("ack", {}),
        }

    def sync_demo_position_targets(
        self,
        *,
        symbol: str,
        direction: str,
        stop_loss: float | None,
        take_profit: float | None,
        position_id: int | None = None,
        reference_price: float | None = None,
    ) -> Dict[str, Any]:
        """Synchronize protective levels to exactly one cTrader demo position.

        This is deliberately synchronous: local targets are considered broker-synced
        only after reconcile confirms the requested SL/TP values.
        """
        if not ctd.is_demo_account_confirmed():
            reason = ctd.get_account_verification_error() or "Connected cTrader account is not confirmed as demo."
            raise RuntimeError(f"Demo target sync blocked: {reason}")

        sym = (symbol or "").strip().upper()
        side = "buy" if direction == "long" else ("sell" if direction == "short" else "")
        if not side:
            raise RuntimeError(f"Demo target sync blocked: unsupported direction {direction!r}.")

        symbol_id = (ctd.symbol_name_to_id or {}).get(sym)
        if symbol_id is None:
            raise RuntimeError(f"Demo target sync blocked: broker symbol {sym!r} is unavailable.")

        def _matching_position() -> Dict[str, Any] | None:
            rows = ctd.get_open_positions() or []
            if position_id is not None:
                for row in rows:
                    if int(row.get("position_id") or 0) == int(position_id):
                        return row
                return None
            matches = [
                row
                for row in rows
                if str(row.get("symbol_name") or "").upper() == sym
                and str(row.get("direction") or "").lower() == side
            ]
            if len(matches) > 1:
                raise RuntimeError(
                    f"Demo target sync is ambiguous: {len(matches)} broker positions match {sym} {side}."
                )
            return matches[0] if matches else None

        broker_position = None
        for attempt in range(3):
            broker_position = _matching_position()
            if broker_position is not None:
                break
            if attempt < 2:
                time.sleep(0.4)
        if broker_position is None:
            raise RuntimeError(f"Demo target sync could not find broker position for {sym} {side}.")

        broker_position_id = int(broker_position.get("position_id") or 0)
        if broker_position_id <= 0:
            raise RuntimeError(f"Demo target sync found {sym} {side} without a valid position id.")

        # Do not move a stale protective target farther away after the market
        # has already crossed it. Signal the caller to close the broker position
        # instead, preserving the strategy's intended protective exit.
        if reference_price is not None:
            ref = float(reference_price)
            sl = float(stop_loss) if stop_loss is not None else None
            tp = float(take_profit) if take_profit is not None else None
            if direction == "long":
                if sl is not None and ref <= sl:
                    return {
                        "status": "exit_due_stop_loss",
                        "symbol": sym,
                        "position_id": broker_position_id,
                        "quantity_lots": broker_position.get("volume_lots"),
                        "reference_price": ref,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "verified": False,
                    }
                if tp is not None and ref >= tp:
                    return {
                        "status": "exit_due_take_profit",
                        "symbol": sym,
                        "position_id": broker_position_id,
                        "quantity_lots": broker_position.get("volume_lots"),
                        "reference_price": ref,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "verified": False,
                    }
            else:
                if sl is not None and ref >= sl:
                    return {
                        "status": "exit_due_stop_loss",
                        "symbol": sym,
                        "position_id": broker_position_id,
                        "quantity_lots": broker_position.get("volume_lots"),
                        "reference_price": ref,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "verified": False,
                    }
                if tp is not None and ref <= tp:
                    return {
                        "status": "exit_due_take_profit",
                        "symbol": sym,
                        "position_id": broker_position_id,
                        "quantity_lots": broker_position.get("volume_lots"),
                        "reference_price": ref,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "verified": False,
                    }

        tick_size = None
        try:
            digits = int((ctd.symbol_digits_map or {}).get(symbol_id))
            tick_size = 10.0 ** -digits
        except (TypeError, ValueError):
            tick_size = None
        tolerance = max(float(tick_size or 0.0) * 1.5, 1e-6)

        def _same(actual: Any, expected: float | None) -> bool:
            if expected is None:
                return actual in (None, 0, 0.0)
            if actual in (None, 0, 0.0):
                return False
            try:
                return abs(float(actual) - float(expected)) <= tolerance
            except (TypeError, ValueError):
                return False

        if _same(broker_position.get("stop_loss"), stop_loss) and _same(
            broker_position.get("take_profit"), take_profit
        ):
            return {
                "status": "already_synced",
                "symbol": sym,
                "position_id": broker_position_id,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "verified": True,
            }

        deferred = ctd.modify_position_sltp(
            client=ctd.client,
            account_id=ctd.ACCOUNT_ID,
            position_id=broker_position_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            symbol_id=symbol_id,
        )
        ack = ctd.wait_for_deferred(deferred, timeout=25)
        ack_payload: Dict[str, Any] = {}
        if isinstance(ack, dict):
            ack_payload = dict(ack)
            if ack.get("status") in {"failed", "order_rejected"}:
                reason = ack.get("error") or ack.get("reject_reason") or ack["status"]
                raise RuntimeError(f"cTrader demo target sync failed: {reason}; ack={ack_payload}")
        else:
            try:
                event = ctd.Protobuf.extract(ack)
                ack_payload = ctd.MessageToDict(event, preserving_proto_field_name=True)
                error_code = getattr(event, "errorCode", None)
                description = getattr(event, "description", None)
                reject_reason = getattr(event, "rejectReason", None)
                execution_type = getattr(event, "executionType", None)
                if error_code:
                    raise RuntimeError(
                        f"cTrader demo target sync rejected: errorCode={error_code} "
                        f"description={description or ''}; ack={ack_payload}"
                    )
                if reject_reason:
                    raise RuntimeError(
                        f"cTrader demo target sync rejected: rejectReason={reject_reason}; ack={ack_payload}"
                    )
                # ProtoOAExecutionType.ORDER_REJECTED == 7.
                if execution_type is not None and int(execution_type) == 7:
                    raise RuntimeError(f"cTrader demo target sync rejected; ack={ack_payload}")
            except RuntimeError:
                raise
            except Exception as exc:
                ack_payload = {"parse_error": str(exc), "raw_type": type(ack).__name__}

        verified_row = None
        last_observed = None
        for attempt in range(8):
            if attempt:
                time.sleep(0.5)
            candidate = _matching_position()
            if candidate is None:
                continue
            last_observed = candidate
            if _same(candidate.get("stop_loss"), stop_loss) and _same(
                candidate.get("take_profit"), take_profit
            ):
                verified_row = candidate
                break

        if verified_row is None:
            observed_sl = (last_observed or {}).get("stop_loss")
            observed_tp = (last_observed or {}).get("take_profit")
            raise RuntimeError(
                "cTrader demo target sync could not verify SL/TP on "
                f"position {broker_position_id}; requested_sl={stop_loss} requested_tp={take_profit} "
                f"observed_sl={observed_sl} observed_tp={observed_tp} ack={ack_payload}"
            )

        return {
            "status": "synced",
            "symbol": sym,
            "position_id": broker_position_id,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "verified": True,
            "ack": ack_payload,
        }

    def close_demo_position(
        self,
        *,
        symbol: str,
        position_id: int,
        quantity_lots: float,
    ) -> Dict[str, Any]:
        if not ctd.is_demo_account_confirmed():
            reason = ctd.get_account_verification_error() or "Connected cTrader account is not confirmed as demo."
            raise RuntimeError(f"Demo close blocked: {reason}")

        sym = (symbol or "").strip().upper()
        symbol_id = (ctd.symbol_name_to_id or {}).get(sym)
        if symbol_id is None:
            raise RuntimeError(f"Demo close blocked: broker symbol {sym!r} is unavailable.")

        deferred = ctd.close_position(
            client=ctd.client,
            account_id=ctd.ACCOUNT_ID,
            position_id=int(position_id),
            symbol_id=int(symbol_id),
            volume_lots=float(quantity_lots),
        )
        ack = ctd.wait_for_deferred(deferred, timeout=25)

        ack_payload: Dict[str, Any] = {}
        if isinstance(ack, dict):
            ack_payload = dict(ack)
            if ack.get("status") in {"failed", "order_rejected"}:
                reason = ack.get("error") or ack.get("reject_reason") or ack["status"]
                raise RuntimeError(f"cTrader demo close failed: {reason}; ack={ack_payload}")
        else:
            try:
                event = ctd.Protobuf.extract(ack)
                ack_payload = ctd.MessageToDict(event, preserving_proto_field_name=True)
                error_code = getattr(event, "errorCode", None)
                description = getattr(event, "description", None)
                reject_reason = getattr(event, "rejectReason", None)
                if error_code:
                    raise RuntimeError(
                        f"cTrader demo close rejected: errorCode={error_code} "
                        f"description={description or ''}; ack={ack_payload}"
                    )
                if reject_reason:
                    raise RuntimeError(
                        f"cTrader demo close rejected: rejectReason={reject_reason}; ack={ack_payload}"
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                ack_payload = {"parse_error": str(exc), "raw_type": type(ack).__name__}

        for attempt in range(8):
            if attempt:
                time.sleep(0.5)
            still_open = any(
                int(row.get("position_id") or 0) == int(position_id)
                for row in (ctd.get_open_positions() or [])
            )
            if not still_open:
                return {
                    "status": "closed",
                    "symbol": sym,
                    "position_id": int(position_id),
                    "quantity_lots": float(quantity_lots),
                    "verified": True,
                    "ack": ack_payload,
                }

        raise RuntimeError(
            f"cTrader demo close could not verify position {position_id} as closed; ack={ack_payload}"
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

        lot_size_units = ctd.symbol_lot_size_map.get(symbol_id)
        try:
            protocol_per_lot = float(lot_size_units) * 100.0
        except (TypeError, ValueError):
            protocol_per_lot = 0.0
        if protocol_per_lot <= 0:
            return self._default_symbol_limits(sym)

        min_api = int(ctd.symbol_min_volume_map.get(symbol_id) or 1)
        step_api = int(ctd.symbol_step_volume_map.get(symbol_id) or min_api or 1)
        max_api = int(ctd.symbol_max_volume_map.get(symbol_id) or max(min_api, step_api))
        max_api = max(max_api, min_api)
        step_api = max(step_api, 1)

        return SymbolLimits(
            symbol=sym,
            source="broker",
            min_lots=min_api / protocol_per_lot,
            step_lots=step_api / protocol_per_lot,
            max_lots=max_api / protocol_per_lot,
            min_api_units=min_api,
            step_api_units=step_api,
            max_api_units=max_api,
            hard_min=bool(ctd.symbol_min_verified.get(symbol_id)),
            hard_step=bool(ctd.symbol_step_verified.get(symbol_id)),
        )

    @classmethod
    def _quote_currency(cls, symbol: str) -> str | None:
        sym = symbol.upper().replace("/", "")
        if any(alias in sym for alias in cls._USD_INDEX_ALIASES):
            return "USD"
        for currency in ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"):
            if sym.endswith(currency):
                return currency
        return None

    def get_instrument_spec(self, symbol: str, account_currency: str = "USD") -> InstrumentSpec:
        sym = (symbol or "").strip().upper()
        account_ccy = (account_currency or "USD").strip().upper()
        symbol_id = (ctd.symbol_name_to_id or {}).get(sym)
        quote_ccy = self._quote_currency(sym)
        if symbol_id is None:
            return InstrumentSpec(
                symbol=sym,
                account_currency=account_ccy,
                quote_currency=quote_ccy,
                notes=["Symbol contract is not loaded from the broker."],
            )

        lot_size = ctd.symbol_lot_size_map.get(symbol_id)
        digits = ctd.symbol_digits_map.get(symbol_id)
        tick_size = (10.0 ** -int(digits)) if digits is not None else None
        conversion = 1.0 if quote_ccy == account_ccy else None
        cash_per_unit = float(lot_size) * conversion if lot_size and conversion else None
        notes: List[str] = []
        if not lot_size:
            notes.append("Broker lotSize metadata is unavailable.")
        if quote_ccy is None:
            notes.append("Quote currency could not be inferred from the broker symbol name.")
        elif conversion is None:
            notes.append(f"{quote_ccy}/{account_ccy} P&L conversion is not available yet.")
        ready = cash_per_unit is not None and cash_per_unit > 0
        return InstrumentSpec(
            symbol=sym,
            source="ctrader_contract" if lot_size else "fallback",
            account_currency=account_ccy,
            quote_currency=quote_ccy,
            lot_size_units=float(lot_size) if lot_size else None,
            tick_size=tick_size,
            tick_value_per_lot=(tick_size * cash_per_unit) if tick_size and cash_per_unit else None,
            cash_per_price_unit_per_lot=cash_per_unit,
            conversion_rate_to_account=conversion,
            valuation_ready=ready,
            verified=bool(ready and lot_size),
            notes=notes,
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
