from __future__ import annotations

from datetime import UTC, datetime, timedelta
import threading
import time
from typing import Any, Dict, List

import backend.ctrader_client as ctd
import pandas as pd

from backend.domain.models import BrokerAccountSnapshot, BrokerStatus, CTraderAccount, InstrumentSpec, SymbolLimits
from backend.services.runtime_state import external_dependency_state, market_data_dependency_state
from backend.services.market_bar_validation import assess_market_frame


class DemoOrderAcknowledgementTimeout(RuntimeError):
    """Broker submission was sent, but the acknowledgement outcome is unknown."""

    def __init__(self, message: str, *, client_msg_id: str | None = None) -> None:
        super().__init__(message)
        self.client_msg_id = client_msg_id
        self.submitted = True


class DemoProtectionSyncFailure(RuntimeError):
    """Broker protection amend/verification failed after the target-sync path began."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        broker_position_id: int,
        ack: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.broker_position_id = int(broker_position_id)
        self.ack = dict(ack or {})


class DemoCloseRejected(RuntimeError):
    """Broker explicitly rejected a submitted demo close request."""

    def __init__(
        self,
        message: str,
        *,
        broker_position_id: int,
        ack: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.broker_position_id = int(broker_position_id)
        self.ack = dict(ack or {})
        self.submitted = True
        self.ambiguous = False


class DemoCloseOutcomeAmbiguous(RuntimeError):
    """A demo close was submitted, but broker truth did not resolve its outcome."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        broker_position_id: int,
        ack: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.broker_position_id = int(broker_position_id)
        self.ack = dict(ack or {})
        self.submitted = True
        self.ambiguous = True


def _utc_now() -> datetime:
    return datetime.now(UTC)


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
        self._account_snapshot_cache: BrokerAccountSnapshot | None = None
        self._account_snapshot_cached_at: float = 0.0
        self._conversion_rate_cache: Dict[tuple[str, str], tuple[float, float]] = {}

    def start_transport(self) -> bool:
        with self._thread_lock:
            if self._thread_started:
                return False
            threading.Thread(target=ctd.init_client, daemon=True).start()
            self._thread_started = True
            return True

    def transport_started(self) -> bool:
        return self._thread_started

    def list_accounts(self) -> List[CTraderAccount]:
        return [CTraderAccount(**row) for row in ctd.get_available_accounts()]

    def demo_symbol_execution_readiness(self, symbol: str) -> tuple[bool, str]:
        """Return whether broker state and metadata are ready to safely submit a demo order."""
        if not self.connected():
            return False, "cTrader transport is not connected."
        if not ctd.is_authorized():
            reason = ctd.get_auth_error() or "cTrader account is not authorized."
            return False, reason
        if not ctd.is_demo_account_confirmed():
            reason = ctd.get_account_verification_error() or "Connected cTrader account is not confirmed as demo."
            return False, reason
        if not ctd.is_symbol_metadata_ready():
            return False, "Broker symbol contract metadata is not loaded for the current cTrader session yet."

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

    def get_account_snapshot(self, *, force: bool = False) -> BrokerAccountSnapshot:
        if not ctd.is_demo_account_confirmed():
            reason = ctd.get_account_verification_error() or "Connected cTrader account is not confirmed as demo."
            return BrokerAccountSnapshot(
                account_id=self._account_id_value(),
                source="ctrader",
                verified=False,
                notes=[reason],
            )

        now = time.monotonic()
        if (
            not force
            and self._account_snapshot_cache is not None
            and (now - self._account_snapshot_cached_at) < 5.0
        ):
            return self._account_snapshot_cache

        try:
            raw = ctd.get_account_snapshot()
            as_of_raw = raw.get("as_of")
            as_of = datetime.fromisoformat(str(as_of_raw)) if as_of_raw else _utc_now()
            snapshot = BrokerAccountSnapshot(
                account_id=int(raw.get("account_id") or 0) or self._account_id_value(),
                currency=str(raw.get("currency") or "").upper() or None,
                balance=float(raw["balance"]) if raw.get("balance") is not None else None,
                unrealized_pnl=float(raw["unrealized_pnl"]) if raw.get("unrealized_pnl") is not None else None,
                equity=float(raw["equity"]) if raw.get("equity") is not None else None,
                money_digits=int(raw["money_digits"]) if raw.get("money_digits") is not None else None,
                deposit_asset_id=int(raw["deposit_asset_id"]) if raw.get("deposit_asset_id") is not None else None,
                source="ctrader",
                verified=bool(raw.get("verified")),
                as_of=as_of,
                notes=[],
            )
        except Exception as exc:
            snapshot = BrokerAccountSnapshot(
                account_id=self._account_id_value(),
                source="ctrader",
                verified=False,
                as_of=_utc_now(),
                notes=[str(exc)],
            )

        self._account_snapshot_cache = snapshot
        self._account_snapshot_cached_at = now
        return snapshot

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
        metadata_ready = bool(ctd.is_symbol_metadata_ready())
        symbols_loaded = len(ctd.symbol_name_to_id or {}) if metadata_ready else 0

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
        if not metadata_ready:
            notes.append("Broker symbol contract metadata is not loaded for the current cTrader session.")
        elif symbols_loaded == 0:
            notes.append("No broker symbols are loaded.")
        demo_confirmed = bool(ctd.is_demo_account_confirmed())
        account_snapshot = self.get_account_snapshot() if demo_confirmed else None
        account_type = (
            "demo"
            if ctd.ACCOUNT_IS_DEMO is True
            else ("live" if ctd.ACCOUNT_IS_DEMO is False else "unknown")
        )
        verification_error = ctd.get_account_verification_error()
        if verification_error and verification_error not in notes:
            notes.append(verification_error)
        if account_snapshot is not None and not account_snapshot.verified:
            snapshot_reason = "; ".join(account_snapshot.notes) or "cTrader monetary account snapshot is unavailable."
            notes.append(f"account_snapshot_unavailable: {snapshot_reason}")
        if connected and not demo_confirmed:
            notes.append("Order execution is blocked until the connected account is confirmed as demo.")
        
        notes.extend(external_dependency_state.snapshot_notes())
        auth_note = next((note for note in notes if "not authorized" in note.lower()), "")
        if authorized and auth_note:
            authorized = False
            auth_error = auth_note

        ready = connected and authorized and metadata_ready and symbols_loaded > 0
        market_data_ready = bool(market_data_dependency_state.market_data_ready and ready)
        if (
            ready
            and market_data_dependency_state.last_checked_at is not None
            and not market_data_dependency_state.market_data_ready
        ):
            reason = str(market_data_dependency_state.last_reason or "Market data is not ready.")
            notes.append(f"market_data_unavailable: {reason}")
        monetary_snapshot_ready = bool(
            account_snapshot is not None
            and account_snapshot.verified
            and isinstance(account_snapshot.currency, str)
            and len(account_snapshot.currency.strip()) == 3
            and account_snapshot.equity is not None
            and float(account_snapshot.equity) > 0
        )
        execution_ready = bool(ready and demo_confirmed and monetary_snapshot_ready)
        if demo_confirmed and not monetary_snapshot_ready:
            notes.append(
                "Execution is blocked until cTrader provides a verified account currency and positive equity."
            )

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
            account_snapshot=account_snapshot,
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
        if not self.connected():
            raise RuntimeError("Demo order blocked: cTrader transport is not connected.")
        if not ctd.is_authorized():
            reason = ctd.get_auth_error() or "cTrader account is not authorized."
            raise RuntimeError(f"Demo order blocked: {reason}")
        if not ctd.is_demo_account_confirmed():
            reason = ctd.get_account_verification_error() or "Connected cTrader account is not confirmed as demo."
            raise RuntimeError(f"Demo order blocked: {reason}")

        sym = (symbol or "").strip().upper()
        metadata_ready, metadata_reason = self.demo_symbol_execution_readiness(sym)
        if not metadata_ready:
            raise RuntimeError(f"Demo order blocked: {metadata_reason}")

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
            if result.get("status") == "failed" and "timeout" in str(reason).lower():
                raise DemoOrderAcknowledgementTimeout(
                    "cTrader demo order acknowledgement timed out after submission; broker outcome is ambiguous.",
                    client_msg_id=client_msg_id,
                )
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
                failure_kind = (
                    "ack_timeout"
                    if "timeout" in str(reason).lower()
                    else ("amend_rejected" if ack.get("status") == "order_rejected" else "amend_failed")
                )
                raise DemoProtectionSyncFailure(
                    f"cTrader demo target sync failed: {reason}; ack={ack_payload}",
                    failure_kind=failure_kind,
                    broker_position_id=broker_position_id,
                    ack=ack_payload,
                )
        else:
            try:
                event = ctd.Protobuf.extract(ack)
                ack_payload = ctd.MessageToDict(event, preserving_proto_field_name=True)
                error_code = getattr(event, "errorCode", None)
                description = getattr(event, "description", None)
                reject_reason = getattr(event, "rejectReason", None)
                execution_type = getattr(event, "executionType", None)
                if error_code:
                    raise DemoProtectionSyncFailure(
                        f"cTrader demo target sync rejected: errorCode={error_code} "
                        f"description={description or ''}; ack={ack_payload}",
                        failure_kind="amend_rejected",
                        broker_position_id=broker_position_id,
                        ack=ack_payload,
                    )
                if reject_reason:
                    raise DemoProtectionSyncFailure(
                        f"cTrader demo target sync rejected: rejectReason={reject_reason}; ack={ack_payload}",
                        failure_kind="amend_rejected",
                        broker_position_id=broker_position_id,
                        ack=ack_payload,
                    )
                # ProtoOAExecutionType.ORDER_REJECTED == 7.
                if execution_type is not None and int(execution_type) == 7:
                    raise DemoProtectionSyncFailure(
                        f"cTrader demo target sync rejected; ack={ack_payload}",
                        failure_kind="amend_rejected",
                        broker_position_id=broker_position_id,
                        ack=ack_payload,
                    )
            except DemoProtectionSyncFailure:
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
            raise DemoProtectionSyncFailure(
                "cTrader demo target sync could not verify SL/TP on "
                f"position {broker_position_id}; requested_sl={stop_loss} requested_tp={take_profit} "
                f"observed_sl={observed_sl} observed_tp={observed_tp} ack={ack_payload}",
                failure_kind="verification_failed",
                broker_position_id=broker_position_id,
                ack=ack_payload,
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

    def _normalize_closing_deals(
        self,
        deals: list[Any],
        *,
        symbol_hint: str | None = None,
    ) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        hinted_symbol_id = (
            (ctd.symbol_name_to_id or {}).get(str(symbol_hint or "").strip().upper())
            if symbol_hint
            else None
        )

        def _money_digits(deal: Any, detail: Any) -> int:
            for obj in (detail, deal):
                try:
                    if obj.HasField("moneyDigits"):
                        return int(getattr(obj, "moneyDigits"))
                except Exception:
                    value = getattr(obj, "moneyDigits", None)
                    if value not in (None, ""):
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            pass
            return 2

        for deal in deals:
            detail = getattr(deal, "closePositionDetail", None)
            has_detail = False
            try:
                has_detail = bool(deal.HasField("closePositionDetail"))
            except Exception:
                try:
                    has_detail = detail is not None and bool(detail.ListFields())
                except Exception:
                    has_detail = detail is not None
            if detail is None or not has_detail:
                continue

            try:
                deal_id = int(getattr(deal, "dealId", 0) or 0)
            except (TypeError, ValueError):
                deal_id = 0
            if deal_id <= 0:
                continue

            digits = _money_digits(deal, detail)
            scale = float(10 ** max(0, digits))
            gross_profit = float(getattr(detail, "grossProfit", 0) or 0) / scale
            swap = float(getattr(detail, "swap", 0) or 0) / scale
            commission = float(getattr(detail, "commission", 0) or 0) / scale
            pnl_conversion_fee = float(getattr(detail, "pnlConversionFee", 0) or 0) / scale
            closed_volume_api = float(
                getattr(detail, "closedVolume", 0)
                or getattr(deal, "filledVolume", 0)
                or getattr(deal, "volume", 0)
                or 0
            )

            trade_data = getattr(deal, "tradeData", None)
            symbol_id = getattr(trade_data, "symbolId", None) if trade_data is not None else None
            if symbol_id in (None, 0):
                symbol_id = hinted_symbol_id

            closed_volume_lots = None
            if symbol_id not in (None, 0) and closed_volume_api > 0:
                try:
                    closed_volume_lots = ctd.protocol_volume_to_lots(
                        int(symbol_id),
                        closed_volume_api,
                    )
                except Exception:
                    closed_volume_lots = None

            execution_ts_ms = int(getattr(deal, "executionTimestamp", 0) or 0)
            execution_at = (
                datetime.fromtimestamp(execution_ts_ms / 1000.0, tz=UTC).replace(tzinfo=None)
                if execution_ts_ms > 0
                else _utc_now().replace(tzinfo=None)
            )
            execution_price = float(getattr(deal, "executionPrice", 0.0) or 0.0)
            rows.append(
                {
                    "deal_id": deal_id,
                    "execution_price": execution_price,
                    "execution_at": execution_at,
                    "closed_volume_api": closed_volume_api,
                    "closed_volume_lots": (
                        float(closed_volume_lots)
                        if closed_volume_lots is not None
                        else None
                    ),
                    "gross_profit": gross_profit,
                    "swap": swap,
                    "commission": commission,
                    "pnl_conversion_fee": pnl_conversion_fee,
                    "net_profit": gross_profit + swap + commission + pnl_conversion_fee,
                }
            )
        return rows

    def get_position_close_deals(
        self,
        position_id: int,
        *,
        symbol: str | None = None,
        opened_at_hint: datetime | None = None,
    ) -> list[Dict[str, Any]]:
        """Return normalized cTrader closing deals for an open or closed position."""
        now_utc = _utc_now()
        window_end = now_utc
        window_start = now_utc - timedelta(days=2)
        if opened_at_hint is not None:
            hint = opened_at_hint
            if hint.tzinfo is None:
                hint = hint.replace(tzinfo=UTC)
            else:
                hint = hint.astimezone(UTC)
            # Keep the request bounded because the cTrader history endpoint can
            # reject very wide ranges. Real-time reconciliation runs frequently,
            # so a two-day floor covers newly observed partial closes while an
            # earlier opening hint still widens the range for recent positions.
            window_start = max(hint - timedelta(days=1), now_utc - timedelta(days=7))

        deals = ctd.get_deals_by_position_id(
            int(position_id),
            from_timestamp=int(window_start.timestamp() * 1000),
            to_timestamp=int(window_end.timestamp() * 1000),
        )
        return self._normalize_closing_deals(deals, symbol_hint=symbol)

    def get_closed_position_summary(
        self,
        position_id: int,
        *,
        closed_at_hint: datetime | None = None,
        symbol: str | None = None,
    ) -> Dict[str, Any] | None:
        """Return authoritative broker close price/P&L for a closed position."""
        now_utc = _utc_now()
        if closed_at_hint is not None:
            hint = closed_at_hint
            if hint.tzinfo is None:
                hint = hint.replace(tzinfo=UTC)
            else:
                hint = hint.astimezone(UTC)
            window_start = hint - timedelta(days=1)
            window_end = min(hint + timedelta(days=1), now_utc)
            if window_start >= window_end:
                window_start = window_end - timedelta(days=2)
        else:
            window_end = now_utc
            window_start = window_end - timedelta(days=2)

        deals = ctd.get_deals_by_position_id(
            int(position_id),
            from_timestamp=int(window_start.timestamp() * 1000),
            to_timestamp=int(window_end.timestamp() * 1000),
        )
        closing = self._normalize_closing_deals(deals, symbol_hint=symbol)
        if not closing:
            return None

        total_weight = sum(float(row.get("closed_volume_api") or 0.0) for row in closing)
        weighted_exit = sum(
            float(row.get("execution_price") or 0.0) * float(row.get("closed_volume_api") or 0.0)
            for row in closing
            if float(row.get("execution_price") or 0.0) > 0
        )
        exit_price = weighted_exit / total_weight if total_weight > 0 else None
        gross_profit = sum(float(row.get("gross_profit") or 0.0) for row in closing)
        swap = sum(float(row.get("swap") or 0.0) for row in closing)
        commission = sum(float(row.get("commission") or 0.0) for row in closing)
        pnl_conversion_fee = sum(float(row.get("pnl_conversion_fee") or 0.0) for row in closing)
        net_profit = sum(float(row.get("net_profit") or 0.0) for row in closing)
        latest = max(
            (row.get("execution_at") for row in closing if row.get("execution_at") is not None),
            default=None,
        )

        return {
            "status": "found",
            "position_id": int(position_id),
            "exit_price": exit_price,
            "closed_at": latest,
            "gross_profit": gross_profit,
            "swap": swap,
            "commission": commission,
            "pnl_conversion_fee": pnl_conversion_fee,
            "net_profit": net_profit,
            "closed_volume_api": total_weight,
            "closed_volume_lots": sum(
                float(row.get("closed_volume_lots") or 0.0)
                for row in closing
                if row.get("closed_volume_lots") is not None
            ),
            "deal_ids": [int(row["deal_id"]) for row in closing],
            "deals": closing,
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

        broker_position_id = int(position_id)
        deferred = ctd.close_position(
            client=ctd.client,
            account_id=ctd.ACCOUNT_ID,
            position_id=broker_position_id,
            symbol_id=int(symbol_id),
            volume_lots=float(quantity_lots),
        )
        ack = ctd.wait_for_deferred(deferred, timeout=25)

        ack_payload: Dict[str, Any] = {}
        ambiguous_kind: str | None = None
        if isinstance(ack, dict):
            ack_payload = dict(ack)
            status = str(ack.get("status") or "").lower()
            if status == "order_rejected":
                reason = ack.get("reject_reason") or ack.get("error") or status
                raise DemoCloseRejected(
                    f"cTrader demo close rejected: {reason}; ack={ack_payload}",
                    broker_position_id=broker_position_id,
                    ack=ack_payload,
                )
            if status == "failed":
                reason = ack.get("error") or status
                ambiguous_kind = "ack_timeout" if "timeout" in str(reason).lower() else "ack_failed"
        else:
            try:
                event = ctd.Protobuf.extract(ack)
                ack_payload = ctd.MessageToDict(event, preserving_proto_field_name=True)
                error_code = getattr(event, "errorCode", None)
                description = getattr(event, "description", None)
                reject_reason = getattr(event, "rejectReason", None)
                if error_code:
                    raise DemoCloseRejected(
                        f"cTrader demo close rejected: errorCode={error_code} "
                        f"description={description or ''}; ack={ack_payload}",
                        broker_position_id=broker_position_id,
                        ack=ack_payload,
                    )
                if reject_reason:
                    raise DemoCloseRejected(
                        f"cTrader demo close rejected: rejectReason={reject_reason}; ack={ack_payload}",
                        broker_position_id=broker_position_id,
                        ack=ack_payload,
                    )
            except DemoCloseRejected:
                raise
            except Exception as exc:
                ack_payload = {"parse_error": str(exc), "raw_type": type(ack).__name__}
                ambiguous_kind = "ack_unrecognized"

        last_open_rows: List[Dict[str, Any]] = []
        for attempt in range(8):
            if attempt:
                time.sleep(0.5)
            last_open_rows = ctd.get_open_positions() or []
            still_open = any(
                int(row.get("position_id") or 0) == broker_position_id
                for row in last_open_rows
            )
            if not still_open:
                close_summary = None
                for history_attempt in range(5):
                    try:
                        close_summary = self.get_closed_position_summary(broker_position_id)
                    except Exception:
                        close_summary = None
                    if close_summary is not None:
                        break
                    if history_attempt < 4:
                        time.sleep(0.4)
                return {
                    "status": "closed",
                    "symbol": sym,
                    "position_id": broker_position_id,
                    "quantity_lots": float(quantity_lots),
                    "verified": True,
                    "ack": ack_payload,
                    "acknowledgement_ambiguous": ambiguous_kind is not None,
                    "reconciled_from_broker": ambiguous_kind is not None,
                    "close_summary": close_summary,
                }

        raise DemoCloseOutcomeAmbiguous(
            (
                f"cTrader demo close outcome remains ambiguous for position {broker_position_id}; "
                f"broker position is still visible after verification polling; ack={ack_payload}"
            ),
            failure_kind=ambiguous_kind or "verification_failed",
            broker_position_id=broker_position_id,
            ack=ack_payload,
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
        if not ctd.is_symbol_metadata_ready():
            self._symbol_cache = []
            self._symbol_cache_count = 0
            return sorted(ctd.FALLBACK_SYMBOLS)

        # Quick check if symbols changed via length within the current verified session.
        current_id_map = ctd.symbol_name_to_id or {}
        count = len(current_id_map)

        if self._symbol_cache and self._symbol_cache_count == count and count > 0:
            return self._symbol_cache

        symbols = sorted(current_id_map.keys())
        if not symbols:
            return sorted(ctd.FALLBACK_SYMBOLS)

        self._symbol_cache = symbols
        self._symbol_cache_count = count
        return symbols

    def get_symbol_limits(self, symbol: str) -> SymbolLimits:
        sym = (symbol or "").strip().upper()
        if not sym or not ctd.is_symbol_metadata_ready():
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

    def _conversion_rate_to_account(self, quote_currency: str | None, account_currency: str) -> float | None:
        quote = str(quote_currency or "").strip().upper()
        account = str(account_currency or "").strip().upper()
        if not quote or not account:
            return None
        if quote == account:
            return 1.0

        key = (quote, account)
        cached = self._conversion_rate_cache.get(key)
        now = time.monotonic()
        if cached is not None and (now - cached[1]) < 60.0:
            return cached[0]

        candidates = (
            (f"{quote}{account}", False),
            (f"{account}{quote}", True),
        )
        available = ctd.symbol_name_to_id or {}
        for pair, invert in candidates:
            if pair not in available:
                continue
            try:
                rows = ctd.get_ohlc_data(pair, "M1", 1)
                if not rows:
                    continue
                close = float(rows[-1]["close"])
                if close <= 0:
                    continue
                rate = (1.0 / close) if invert else close
                self._conversion_rate_cache[key] = (rate, now)
                return rate
            except Exception:
                continue
        return None

    def get_instrument_spec(self, symbol: str, account_currency: str = "USD") -> InstrumentSpec:
        sym = (symbol or "").strip().upper()
        account_ccy = (account_currency or "USD").strip().upper()
        quote_ccy = self._quote_currency(sym)
        if not ctd.is_symbol_metadata_ready():
            return InstrumentSpec(
                symbol=sym,
                account_currency=account_ccy,
                quote_currency=quote_ccy,
                notes=["Broker symbol contract metadata is not loaded for the current cTrader session."],
            )

        symbol_id = (ctd.symbol_name_to_id or {}).get(sym)
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
        conversion = self._conversion_rate_to_account(quote_ccy, account_ccy)
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

        required_columns = {"time", "open", "high", "low", "close"}
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            raise RuntimeError(
                f"Malformed market data for {sym}:{tf}: missing columns {', '.join(missing)}."
            )

        df["time"] = self._normalize_time_column(df["time"])
        df = df.dropna(subset=["time"])
        if df.empty:
            raise RuntimeError(f"Malformed market data for {sym}:{tf}: no valid bar timestamps.")
        df = df.set_index("time").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if num_bars and len(df) > num_bars:
            df = df.iloc[-num_bars:]

        valid, _details, malformed_reason = assess_market_frame(tf, df)
        if not valid:
            raise RuntimeError(
                f"{malformed_reason or 'Malformed market data.'} {sym}:{tf}"
            )

        live_price = float(df["close"].iloc[-1])
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
            "latest_bar_at": None,
        }

        try:
            df, _ = self._get_real_bars(sym, tf, 5)
            if df is not None and not df.empty:
                latest = df.index[-1]
                if hasattr(latest, "to_pydatetime"):
                    latest = latest.to_pydatetime()
                status["ok"] = True
                status["reason"] = f"Fetched {len(df)} bars"
                status["latest_bar_at"] = (
                    latest.isoformat()
                    if isinstance(latest, datetime)
                    else str(latest)
                )
                return status
        except Exception as exc:
            status["reason"] = str(exc)
            return status

        status["reason"] = f"No market data available for {sym}:{tf}"
        return status


adapter = CTraderBrokerAdapter()
