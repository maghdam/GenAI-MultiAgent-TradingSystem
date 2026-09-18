from __future__ import annotations

import ast
import textwrap


_ALLOWED_IMPORTS = {"pandas": "pd", "numpy": "np"}
_DENIED_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "exit", "globals",
    "help", "input", "locals", "open", "print", "quit", "setattr", "delattr",
    "getattr", "vars", "memoryview",
}
_DENIED_ATTRIBUTES = {
    "system", "popen", "spawn", "fork", "socket", "connect", "request", "urlopen",
    "read_csv", "read_excel", "read_feather", "read_fwf", "read_gbq", "read_hdf",
    "read_html", "read_json", "read_orc", "read_parquet", "read_pickle", "read_sas",
    "read_spss", "read_sql", "read_stata", "read_table", "read_xml",
    "to_clipboard", "to_csv", "to_excel", "to_feather", "to_gbq", "to_hdf",
    "to_html", "to_json", "to_orc", "to_parquet", "to_pickle", "to_sql", "to_stata",
    "to_xml", "save", "dump", "dumps", "load", "loads",
}


class StrategyPolicyError(ValueError):
    pass


def validate_strategy_source(source: str) -> str:
    normalized = textwrap.dedent(str(source or "")).lstrip("\n").replace("\r\n", "\n")
    if not normalized.strip():
        raise StrategyPolicyError("Strategy source is empty.")
    if len(normalized.encode("utf-8")) > 100_000:
        raise StrategyPolicyError("Strategy source exceeds the 100 KB limit.")
    try:
        tree = ast.parse(normalized)
    except SyntaxError as exc:
        raise StrategyPolicyError(f"Generated code is not valid Python: {exc}") from exc

    signals_functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "signals"]
    if len(signals_functions) != 1 or isinstance(signals_functions[0], ast.AsyncFunctionDef):
        raise StrategyPolicyError("Strategy code must define exactly one synchronous signals(df, ...) function.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                expected_alias = _ALLOWED_IMPORTS.get(alias.name)
                if expected_alias is None or (alias.asname or alias.name) != expected_alias:
                    raise StrategyPolicyError("Only 'import pandas as pd' and 'import numpy as np' are allowed.")
        elif isinstance(node, ast.ImportFrom):
            raise StrategyPolicyError("from-import statements are not allowed in generated strategies.")
        elif isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Await, ast.Yield, ast.YieldFrom)):
            raise StrategyPolicyError(f"{type(node).__name__} is not allowed in generated strategies.")
        elif isinstance(node, ast.Name) and node.id in _DENIED_NAMES:
            raise StrategyPolicyError(f"Use of '{node.id}' is not allowed in generated strategies.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _DENIED_ATTRIBUTES:
                raise StrategyPolicyError(f"Attribute '{node.attr}' is not allowed in generated strategies.")

    return normalized


def source_without_imports(source: str) -> ast.Module:
    tree = ast.parse(validate_strategy_source(source))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return ast.fix_missing_locations(tree)
