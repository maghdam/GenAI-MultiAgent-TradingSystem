from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.services.strategy_policy import validate_strategy_source


_RUNNER = r'''
import ast, json, sys
import numpy as np
import pandas as pd

payload = json.loads(sys.stdin.read())
tree = ast.parse(payload["source"])
tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
ast.fix_missing_locations(tree)
safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}
namespace = {"__builtins__": safe_builtins, "pd": pd, "np": np}
exec(compile(tree, "<isolated-strategy>", "exec"), namespace, namespace)
frame = pd.DataFrame(payload["records"])
frame.index = pd.to_datetime(payload["index"], utc=True)
result = namespace["signals"](frame, **payload.get("params", {}))
if not isinstance(result, pd.Series):
    raise TypeError("signals(df, ...) must return a pandas Series")
result = result.reindex(frame.index).fillna(0.0)
values = np.asarray(result.tolist(), dtype=float)
if not np.isfinite(values).all():
    raise ValueError("signals(df, ...) returned non-finite values")
sys.stdout.write(json.dumps({"ok": True, "values": values.tolist()}, separators=(",", ":")))
'''


class StrategySandboxError(RuntimeError):
    pass


def _configured_max_bars() -> int:
    try:
        return max(100, int(os.getenv("STRATEGY_SANDBOX_MAX_BARS", "20000")))
    except Exception:
        return 20_000


def _configured_max_timeout_seconds() -> float:
    try:
        return max(0.5, float(os.getenv("STRATEGY_SANDBOX_MAX_TIMEOUT_SECONDS", "30")))
    except Exception:
        return 30.0


def run_strategy_source(
    source: str,
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    timeout_seconds: float = 30.0,
) -> pd.Series:
    normalized = validate_strategy_source(source)
    if frame is None or frame.empty:
        raise StrategySandboxError("Strategy sandbox requires non-empty market data.")

    max_bars = _configured_max_bars()
    if len(frame.index) > max_bars:
        raise StrategySandboxError(
            f"Strategy sandbox input exceeds the {max_bars} bar resource limit."
        )
    columns = [column for column in ("open", "high", "low", "close", "volume") if column in frame.columns]
    payload = {
        "source": normalized,
        "records": frame[columns].reset_index(drop=True).to_dict(orient="records"),
        "index": [pd.Timestamp(value).isoformat() for value in frame.index],
        "params": params or {},
    }
    safe_env = {
        key: value for key, value in os.environ.items()
        if key.upper() in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH"}
    }
    requested_timeout = max(0.5, float(timeout_seconds))
    effective_timeout = min(requested_timeout, _configured_max_timeout_seconds())

    with tempfile.TemporaryDirectory(prefix="tradeagent-strategy-") as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                input=json.dumps(payload, separators=(",", ":")),
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                cwd=Path(workdir),
                env=safe_env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StrategySandboxError(f"Strategy exceeded the {effective_timeout:.1f}s execution limit.") from exc
    if result.returncode != 0:
        error = (result.stderr or "isolated strategy process failed").strip().splitlines()[-1]
        raise StrategySandboxError(error[:500])
    try:
        response = json.loads(result.stdout)
        if response.get("ok") is not True:
            raise ValueError("sandbox response did not report success")
        values = response["values"]
        if not isinstance(values, list):
            raise TypeError("sandbox response values must be a list")
        numeric = np.asarray(values, dtype=float)
    except Exception as exc:
        raise StrategySandboxError("Strategy sandbox returned an invalid response.") from exc
    if len(numeric) != len(frame.index):
        raise StrategySandboxError("Strategy output length does not match the input bars.")
    if not np.isfinite(numeric).all():
        raise StrategySandboxError("Strategy output contains non-finite values.")
    return pd.Series(numeric, index=frame.index, dtype=float)
