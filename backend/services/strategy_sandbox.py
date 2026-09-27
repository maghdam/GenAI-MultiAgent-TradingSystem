from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

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


def _configured_memory_limit_bytes() -> int:
    try:
        memory_mb = max(64, int(os.getenv("STRATEGY_SANDBOX_MEMORY_MB", "512")))
    except Exception:
        memory_mb = 512
    return memory_mb * 1024 * 1024


def _process_tree_rss(process: subprocess.Popen[str]) -> int:
    try:
        parent = psutil.Process(process.pid)
        processes = [parent, *parent.children(recursive=True)]
    except (psutil.Error, OSError):
        return 0

    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.Error, OSError):
                pass
        try:
            parent.kill()
        except (psutil.Error, OSError):
            pass
    except (psutil.Error, OSError):
        try:
            process.kill()
        except OSError:
            pass

    try:
        process.wait(timeout=2)
    except Exception:
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

    memory_limit_bytes = _configured_memory_limit_bytes()
    payload_text = json.dumps(payload, separators=(",", ":"))

    with tempfile.TemporaryDirectory(prefix="tradeagent-strategy-") as workdir:
        stdout_path = Path(workdir) / "stdout.txt"
        stderr_path = Path(workdir) / "stderr.txt"
        with stdout_path.open("w+", encoding="utf-8") as stdout_file, stderr_path.open(
            "w+", encoding="utf-8"
        ) as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", _RUNNER],
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                cwd=Path(workdir),
                env=safe_env,
            )
            if process.stdin is None:
                _kill_process_tree(process)
                raise StrategySandboxError("Strategy sandbox could not open child-process input.")

            try:
                process.stdin.write(payload_text)
                process.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                _kill_process_tree(process)
                raise StrategySandboxError("Strategy sandbox child process exited before receiving input.") from exc

            deadline = time.monotonic() + effective_timeout
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    _kill_process_tree(process)
                    raise StrategySandboxError(
                        f"Strategy exceeded the {effective_timeout:.1f}s execution limit."
                    )

                rss = _process_tree_rss(process)
                if rss > memory_limit_bytes:
                    _kill_process_tree(process)
                    limit_mb = memory_limit_bytes // (1024 * 1024)
                    raise StrategySandboxError(
                        f"Strategy exceeded the {limit_mb} MB memory resource limit."
                    )
                time.sleep(0.02)

            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
            returncode = int(process.returncode or 0)

    if returncode != 0:
        error = (stderr or "isolated strategy process failed").strip().splitlines()[-1]
        raise StrategySandboxError(error[:500])
    try:
        response = json.loads(stdout)
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
