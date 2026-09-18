from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.domain.models import (
    StrategyLifecycle,
    StrategyLifecycleEvidence,
    StrategyLifecycleTransition,
)
from backend.storage.db import get_db


STAGES = ("draft", "backtested", "validated", "paper", "eligible")
DEFAULT_GATES: dict[str, Any] = {
    "development_min_trades": 30,
    "validation_min_trades": 20,
    "regime_min_trades": 10,
    "max_backtest_drawdown_pct": 15.0,
    "min_backtest_return_pct": 0.0,
    "require_cost_model": True,
    "paper_min_trades": 20,
    "paper_max_drawdown_pct": 10.0,
    "paper_min_return_pct": 0.0,
}


class StrategyLifecycleError(ValueError):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def source_hash(source: str) -> str:
    normalized = str(source or "").replace("\r\n", "\n").strip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_strategy_name(strategy: str) -> str:
    value = str(strategy or "").strip().lower()
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in value):
        raise StrategyLifecycleError("Strategy name may contain only letters, numbers, underscores, and hyphens.")
    return value


def _strategy_source(strategy: str) -> str | None:
    path = Path("backend/strategies_generated") / f"{_safe_strategy_name(strategy)}.py"
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except OSError:
        return None


def _json(value: str, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return default


def _load_evidence(lifecycle_id: int) -> list[StrategyLifecycleEvidence]:
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM strategy_lifecycle_evidence WHERE lifecycle_id = ? ORDER BY id DESC",
            (lifecycle_id,),
        ).fetchall()
    return [
        StrategyLifecycleEvidence(
            id=int(row["id"]),
            lifecycle_id=int(row["lifecycle_id"]),
            evidence_type=str(row["evidence_type"]),
            passed=bool(row["passed"]),
            summary=str(row["summary"]),
            metrics=_json(row["metrics_json"], {}),
            context=_json(row["context_json"], {}),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
        for row in rows
    ]


def _load_transitions(lifecycle_id: int) -> list[StrategyLifecycleTransition]:
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM strategy_lifecycle_transitions WHERE lifecycle_id = ? ORDER BY id DESC",
            (lifecycle_id,),
        ).fetchall()
    return [
        StrategyLifecycleTransition(
            id=int(row["id"]),
            lifecycle_id=int(row["lifecycle_id"]),
            from_stage=str(row["from_stage"]),
            to_stage=str(row["to_stage"]),
            operator=str(row["operator"]),
            reason=str(row["reason"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
        for row in rows
    ]


def _hydrate(row) -> StrategyLifecycle:
    evidence = _load_evidence(int(row["id"]))
    transitions = _load_transitions(int(row["id"]))
    current_source = _strategy_source(str(row["strategy"]))
    record = StrategyLifecycle(
        id=int(row["id"]),
        strategy=str(row["strategy"]),
        version=int(row["version"]),
        version_hash=str(row["version_hash"]),
        stage=str(row["stage"]),
        hypothesis=str(row["hypothesis"]),
        gates=_json(row["gates_json"], dict(DEFAULT_GATES)),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        evidence=evidence,
        transitions=transitions,
        current_source=current_source is not None and source_hash(current_source) == str(row["version_hash"]),
    )
    return _with_readiness(record)


def _with_readiness(record: StrategyLifecycle) -> StrategyLifecycle:
    if record.stage in ("eligible", "retired"):
        record.next_stage = None
        record.promotion_ready = False
        record.blockers = []
        return record

    next_stage = STAGES[STAGES.index(record.stage) + 1]
    blockers: list[str] = []
    passing = {item.evidence_type for item in record.evidence if item.passed}
    if not record.current_source:
        blockers.append("Saved source no longer matches this version hash.")
    if not record.hypothesis.strip():
        blockers.append("Add a measurable hypothesis before promotion.")
    if record.stage == "draft" and "development_backtest" not in passing:
        blockers.append("A passing development backtest is required.")
    elif record.stage == "backtested":
        if "out_of_sample" not in passing:
            blockers.append("A passing 30% holdout backtest is required.")
        if "regime" not in passing:
            blockers.append("A passing regime test on a different market or period is required.")
    elif record.stage == "paper" and "paper" not in passing:
        blockers.append("Passing paper evidence from closed trades is required.")

    record.next_stage = next_stage
    record.blockers = blockers
    record.promotion_ready = not blockers
    return record


def ensure_lifecycle(strategy: str, source: str, hypothesis: str = "") -> StrategyLifecycle:
    name = _safe_strategy_name(strategy)
    digest = source_hash(source)
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM strategy_lifecycles WHERE strategy = ? AND version_hash = ?",
            (name, digest),
        ).fetchone()
        if row is None:
            previous = db.execute(
                "SELECT version, hypothesis FROM strategy_lifecycles WHERE strategy = ? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
            version = int(previous["version"]) + 1 if previous else 1
            inherited_hypothesis = str(previous["hypothesis"] or "") if previous else ""
            now = _utcnow().isoformat()
            cur = db.execute(
                """
                INSERT INTO strategy_lifecycles(
                    strategy, version, version_hash, stage, hypothesis, gates_json, created_at, updated_at
                ) VALUES(?, ?, ?, 'draft', ?, ?, ?, ?)
                """,
                (
                    name,
                    version,
                    digest,
                    hypothesis.strip() or inherited_hypothesis,
                    json.dumps(DEFAULT_GATES),
                    now,
                    now,
                ),
            )
            db.commit()
            row = db.execute("SELECT * FROM strategy_lifecycles WHERE id = ?", (int(cur.lastrowid),)).fetchone()
    return _hydrate(row)


def get_lifecycle(strategy: str, version_hash: str | None = None) -> StrategyLifecycle:
    name = _safe_strategy_name(strategy)
    with get_db() as db:
        if version_hash:
            row = db.execute(
                "SELECT * FROM strategy_lifecycles WHERE strategy = ? AND version_hash = ?",
                (name, version_hash),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT * FROM strategy_lifecycles WHERE strategy = ? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
    if row is None:
        raise StrategyLifecycleError(f"Strategy '{name}' is not registered. Save or backtest it first.")
    return _hydrate(row)


def list_lifecycles() -> list[StrategyLifecycle]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT lifecycle.*
            FROM strategy_lifecycles lifecycle
            JOIN (
                SELECT strategy, MAX(version) AS version
                FROM strategy_lifecycles
                GROUP BY strategy
            ) latest ON latest.strategy = lifecycle.strategy AND latest.version = lifecycle.version
            ORDER BY lifecycle.updated_at DESC
            """
        ).fetchall()
    return [_hydrate(row) for row in rows]


def update_hypothesis(strategy: str, hypothesis: str) -> StrategyLifecycle:
    record = get_lifecycle(strategy)
    now = _utcnow().isoformat()
    with get_db() as db:
        db.execute(
            "UPDATE strategy_lifecycles SET hypothesis = ?, updated_at = ? WHERE id = ?",
            (str(hypothesis or "").strip(), now, record.id),
        )
        db.commit()
    return get_lifecycle(record.strategy, record.version_hash)


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _evaluate_evidence(evidence_type: str, metrics: dict[str, Any], gates: dict[str, Any]) -> tuple[bool, str]:
    trades = int(_metric(metrics, "Number of Trades"))
    total_return = _metric(metrics, "Total Return [%]")
    drawdown = abs(_metric(metrics, "Max Drawdown [%]"))
    if evidence_type == "paper":
        required = int(gates.get("paper_min_trades", 20))
        passed = (
            trades >= required
            and total_return > float(gates.get("paper_min_return_pct", 0.0))
            and drawdown <= float(gates.get("paper_max_drawdown_pct", 10.0))
        )
        summary = f"Paper: {trades}/{required} trades, return {total_return:.2f}%, drawdown {drawdown:.2f}%."
        return passed, summary

    required_key = {
        "development_backtest": "development_min_trades",
        "out_of_sample": "validation_min_trades",
        "regime": "regime_min_trades",
    }.get(evidence_type)
    if required_key is None:
        raise StrategyLifecycleError(f"Unsupported lifecycle evidence type '{evidence_type}'.")
    required = int(gates.get(required_key, 20))
    costs = _metric(metrics, "Fees [bps]") + _metric(metrics, "Slippage [bps]")
    passed = (
        trades >= required
        and total_return > float(gates.get("min_backtest_return_pct", 0.0))
        and drawdown <= float(gates.get("max_backtest_drawdown_pct", 15.0))
        and (costs > 0.0 or not bool(gates.get("require_cost_model", True)))
    )
    summary = (
        f"{evidence_type.replace('_', ' ').title()}: {trades}/{required} trades, "
        f"return {total_return:.2f}%, drawdown {drawdown:.2f}%, modeled costs {costs:.2f} bps."
    )
    return passed, summary


def record_evidence(
    strategy: str,
    evidence_type: str,
    metrics: dict[str, Any],
    context: dict[str, Any] | None = None,
    summary: str = "",
    version_hash: str | None = None,
) -> StrategyLifecycle:
    record = get_lifecycle(strategy, version_hash)
    passed, generated_summary = _evaluate_evidence(evidence_type, metrics, record.gates)
    now = _utcnow().isoformat()
    with get_db() as db:
        db.execute(
            """
            INSERT INTO strategy_lifecycle_evidence(
                lifecycle_id, evidence_type, passed, summary, metrics_json, context_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                evidence_type,
                int(passed),
                summary.strip() or generated_summary,
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(context or {}, ensure_ascii=False),
                now,
            ),
        )
        db.execute("UPDATE strategy_lifecycles SET updated_at = ? WHERE id = ?", (now, record.id))
        db.commit()
    return get_lifecycle(record.strategy, record.version_hash)


def record_backtest(
    strategy: str,
    source: str,
    metrics: dict[str, Any],
    validation_kind: str,
    context: dict[str, Any] | None = None,
) -> StrategyLifecycle:
    record = ensure_lifecycle(strategy, source)
    return record_evidence(
        record.strategy,
        validation_kind,
        metrics,
        context=context,
        version_hash=record.version_hash,
    )


def record_paper_evidence(strategy: str) -> StrategyLifecycle:
    record = get_lifecycle(strategy)
    with get_db() as db:
        rows = db.execute(
            """
            SELECT realized_pnl FROM paper_positions
            WHERE status = 'closed' AND lower(strategy) = ? ORDER BY closed_at, id
            """,
            (record.strategy,),
        ).fetchall()
        config_row = db.execute("SELECT value FROM config WHERE key = 'engine_config'").fetchone()
    starting_equity = 100_000.0
    if config_row:
        config = _json(config_row["value"], {})
        try:
            starting_equity = float(config.get("paper_starting_equity_amount", starting_equity))
        except (TypeError, ValueError):
            pass
    pnls = [float(row["realized_pnl"] or 0.0) for row in rows]
    equity = starting_equity
    peak = equity
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, (equity / peak) - 1.0)
    metrics = {
        "Number of Trades": len(pnls),
        "Win Rate [%]": round((sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100.0) if pnls else 0.0, 2),
        "Total Return [%]": round(((equity / starting_equity) - 1.0) * 100.0, 4),
        "Max Drawdown [%]": round(max_drawdown * 100.0, 4),
        "Realized P&L": round(sum(pnls), 2),
    }
    return record_evidence(
        record.strategy,
        "paper",
        metrics,
        context={"source": "paper_positions", "starting_equity": starting_equity},
        version_hash=record.version_hash,
    )


def promote(strategy: str, operator: str, reason: str = "") -> StrategyLifecycle:
    record = get_lifecycle(strategy)
    if not record.next_stage:
        raise StrategyLifecycleError(f"Strategy is already {record.stage} and has no next promotion stage.")
    if not record.promotion_ready:
        raise StrategyLifecycleError("Promotion blocked: " + " ".join(record.blockers))
    now = _utcnow().isoformat()
    with get_db() as db:
        db.execute(
            """
            INSERT INTO strategy_lifecycle_transitions(
                lifecycle_id, from_stage, to_stage, operator, reason, created_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (record.id, record.stage, record.next_stage, operator.strip(), reason.strip(), now),
        )
        db.execute(
            "UPDATE strategy_lifecycles SET stage = ?, updated_at = ? WHERE id = ?",
            (record.next_stage, now, record.id),
        )
        db.commit()
    return get_lifecycle(record.strategy, record.version_hash)


def retire(strategy: str, operator: str, reason: str = "") -> StrategyLifecycle:
    record = get_lifecycle(strategy)
    if record.stage == "retired":
        return record
    now = _utcnow().isoformat()
    with get_db() as db:
        db.execute(
            """
            INSERT INTO strategy_lifecycle_transitions(
                lifecycle_id, from_stage, to_stage, operator, reason, created_at
            ) VALUES(?, ?, 'retired', ?, ?, ?)
            """,
            (record.id, record.stage, operator.strip(), reason.strip(), now),
        )
        db.execute(
            "UPDATE strategy_lifecycles SET stage = 'retired', updated_at = ? WHERE id = ?",
            (now, record.id),
        )
        db.commit()
    return get_lifecycle(record.strategy, record.version_hash)


def paper_execution_gate(strategy: str) -> tuple[bool, dict[str, Any], str | None]:
    """Allow legacy strategies, but enforce lifecycle stage for registered versions."""
    try:
        name = _safe_strategy_name(strategy)
    except StrategyLifecycleError:
        return True, {"governed": False}, None
    with get_db() as db:
        row = db.execute(
            "SELECT id FROM strategy_lifecycles WHERE strategy = ? ORDER BY version DESC LIMIT 1",
            (name,),
        ).fetchone()
    if row is None:
        return True, {"governed": False}, None
    record = get_lifecycle(name)
    details = {
        "governed": True,
        "strategy": record.strategy,
        "version": record.version,
        "version_hash": record.version_hash,
        "stage": record.stage,
        "current_source": record.current_source,
    }
    if not record.current_source:
        return False, details, "Registered strategy source does not match its approved lifecycle version."
    if record.stage not in ("paper", "eligible"):
        return False, details, f"Strategy lifecycle stage '{record.stage}' is not approved for paper execution."
    return True, details, None
