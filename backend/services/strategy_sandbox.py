from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

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
values = [float(value) for value in result.tolist()]
sys.stdout.write(json.dumps({"ok": True, "values": values}, separators=(",", ":")))
'''


class StrategySandboxError(RuntimeError):
    pass


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
    with tempfile.TemporaryDirectory(prefix="tradeagent-strategy-") as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                input=json.dumps(payload, separators=(",", ":")),
                text=True,
                capture_output=True,
                timeout=max(0.5, float(timeout_seconds)),
                cwd=Path(workdir),
                env=safe_env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StrategySandboxError(f"Strategy exceeded the {timeout_seconds:.1f}s execution limit.") from exc
    if result.returncode != 0:
        error = (result.stderr or "isolated strategy process failed").strip().splitlines()[-1]
        raise StrategySandboxError(error[:500])
    try:
        response = json.loads(result.stdout)
        values = response["values"]
    except Exception as exc:
        raise StrategySandboxError("Strategy sandbox returned an invalid response.") from exc
    if len(values) != len(frame.index):
        raise StrategySandboxError("Strategy output length does not match the input bars.")
    return pd.Series(values, index=frame.index, dtype=float)
