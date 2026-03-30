from __future__ import annotations

import ast
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Optional

from backend.backtesting_agent import BacktestParams, run_backtest
from backend.services import studio_llm
from backend.optimizer_utils import extract_parameters
from backend.programmer_agent import ProgrammerAgent
from backend.services.studio_backtests import run_strategy_code_backtest
from backend.strategy import load_generated_strategies
from backend.domain.models import StudioTaskRequest, StudioTaskResponse


DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "XAUUSD")


def strategy_studio_help_text() -> str:
    return (
        "Strategy Studio works best as a normal strategy chat.\n"
        "- Ask for a new strategy draft.\n"
        '- Ask to improve or rewrite the current draft, for example: "make entries stricter" or "remove FVG".\n'
        '- Run Backtest to test the current draft on the selected symbol/timeframe.\n'
        '- Save Strategy when the draft looks good, or say: "save it as my_strategy_name".\n'
        "\n"
        'Tip: mention symbol, timeframe, and style when useful, for example: "create an intraday XAUUSD M5 strategy using FVG and market structure".'
    )


def _history_block(ctx: Optional[dict]) -> str:
    history_lines: list[str] = []
    if isinstance(ctx, dict):
        history = ctx.get("history")
        if isinstance(history, list):
            for item in history[-10:]:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip().lower()
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                if role not in ("user", "assistant"):
                    role = "user"
                history_lines.append(f"{role.title()}: {content[:400]}")
    block = "\n".join(history_lines).strip()
    return f"\nConversation so far:\n{block}\n" if block else ""


def _context_line(ctx: Optional[dict]) -> str:
    context_bits: list[str] = []
    if isinstance(ctx, dict):
        if ctx.get("symbol"):
            context_bits.append(f"symbol={ctx['symbol']}")
        if ctx.get("timeframe"):
            context_bits.append(f"timeframe={ctx['timeframe']}")
        if ctx.get("num_bars") or ctx.get("numBars"):
            context_bits.append(f"bars={ctx.get('num_bars') or ctx.get('numBars')}")
        if ctx.get("strategy_name") or ctx.get("strategy"):
            context_bits.append(f"strategy={ctx.get('strategy_name') or ctx.get('strategy')}")
    return ", ".join(context_bits) if context_bits else "n/a"


def _extract_save_name(message: str) -> str | None:
    match = re.match(r"^\s*save(?:\s+(?:this|it|strategy|draft))?(?:\s+as)?\s+([A-Za-z0-9_-]+)\s*$", message, flags=re.I)
    return match.group(1).strip() if match else None


def _clean_generated_code(text: str) -> str:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    if fence:
        cleaned = fence.group(1).strip()
    return cleaned


def _validate_strategy_code(code: str) -> str:
    cleaned = _clean_generated_code(code)
    if not cleaned:
        raise ValueError("The model returned empty strategy code.")

    try:
        ast.parse(cleaned)
    except SyntaxError as exc:
        raise ValueError(f"Generated code is not valid Python: {exc}") from exc

    module_globals: dict[str, object] = {}
    try:
        exec(cleaned, module_globals)
    except Exception as exc:
        raise ValueError(f"Generated code failed to load: {exc}") from exc
    if not callable(module_globals.get("signals")):
        raise ValueError("Generated code must define signals(df, ...) for backtesting.")
    return textwrap.dedent(cleaned).lstrip("\n").replace("\r\n", "\n")


def _save_strategy_code(name: str, code: str) -> StudioTaskResponse:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    dest_dir = Path("backend/strategies_generated")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{safe}.py"
    code_norm = textwrap.dedent(str(code)).lstrip("\n").replace("\r\n", "\n")
    dest_path.write_text(code_norm, encoding="utf-8")
    try:
        load_generated_strategies()
    except Exception:
        pass
    return StudioTaskResponse(status="success", message=f"Strategy saved as {dest_path}")


def _is_code_request(message: str, current_code: str | None) -> bool:
    lower = (message or "").lower()
    if not lower.strip():
        return False
    create_tokens = ("create", "build", "generate", "write", "draft")
    edit_tokens = ("improve", "change", "modify", "update", "rewrite", "refactor", "remove", "add", "replace", "tighten", "loosen")
    strategy_tokens = ("strategy", "signals", "entry", "exit", "stop loss", "take profit", "fvg", "smc", "market structure", "bos", "choch")
    if current_code and any(token in lower for token in edit_tokens):
        return True
    if any(token in lower for token in create_tokens) and any(token in lower for token in strategy_tokens):
        return True
    return bool(current_code and any(token in lower for token in ("remove", "add", "change", "modify", "update", "rewrite")))


async def _generate_strategy_code(message: str, ctx: Optional[dict]) -> dict[str, str]:
    provider = studio_llm.normalize_provider((ctx or {}).get("llm_provider") or (ctx or {}).get("provider"))
    model = (ctx or {}).get("llm_model") or (ctx or {}).get("model")
    current_code = str((ctx or {}).get("current_code") or "").strip()
    prompt = (
        "You are a senior quant developer working inside Strategy Studio.\n"
        "Return only runnable Python code. Do not include markdown fences or explanations.\n"
        "Rules:\n"
        "- The code must define signals(df: pd.DataFrame, ...) -> pd.Series.\n"
        "- Use only pandas and numpy unless absolutely unnecessary.\n"
        "- df includes open, high, low, close, volume.\n"
        "- Return +1 for long, -1 for short, 0 for flat.\n"
        "- Keep the code concise, readable, and backtest-friendly.\n"
        "- Avoid network access, file access, and external services.\n"
        f"\nCurrent UI context: {_context_line(ctx)}\n"
        f"{_history_block(ctx)}"
    )
    if current_code:
        prompt += f"\nCurrent draft strategy:\n{current_code}\n"
        prompt += "\nUpdate the draft according to the user's latest request while keeping the signals contract intact.\n"
    else:
        prompt += "\nCreate a new strategy from the user's request.\n"
    prompt += f"\nUser request: {message}\nPython code:"

    timeout_s = float(os.getenv("STUDIO_CODE_TIMEOUT", os.getenv("STUDIO_LLM_TIMEOUT", "30")) or 30)
    max_tokens = int(os.getenv("STUDIO_CODE_MAX_TOKENS", "900") or 900)

    try:
        response = await studio_llm.generate_text(
            prompt=prompt,
            provider=provider,
            model=model if isinstance(model, str) else None,
            timeout=timeout_s,
            num_predict=max_tokens,
        )
        code = _validate_strategy_code(response["text"])
        response["text"] = code
        return response
    except Exception:
        if current_code:
            raise

        programmer = ProgrammerAgent()
        code = await programmer.generate_code(message, "strategy")
        validated = _validate_strategy_code(code)
        return {
            "provider": "template",
            "model": "programmer_agent",
            "text": validated,
        }


async def strategy_studio_chat_reply(message: str, ctx: Optional[dict]) -> str:
    msg = (message or "").strip()
    if not msg:
        return strategy_studio_help_text()

    ml = msg.lower()
    if re.match(r"^(hi|hello|hey|yo|gm|good morning|good evening)\b", ml):
        return "Tell me what strategy you want to create or improve, then use Run Backtest when the draft is ready."
    if re.search(r"\b(thanks|thank you|thx|appreciate|great|awesome|nice|cool)\b", ml):
        return "Continue with the next change, or run a backtest on the current draft."
    if re.search(r"\b(what can (?:you )?do|what else can (?:you )?do|help|commands?|capabilit(?:y|ies))\b", ml):
        return strategy_studio_help_text()

    prompt = (
        "You are a practical assistant inside a trading Strategy Studio web app.\n"
        "Help the user think through strategy ideas, improvements, and backtest interpretation.\n"
        "Do not emit code unless the user explicitly asks to create, update, or rewrite strategy code.\n"
        "Be concise and concrete.\n"
        f"\nCurrent UI context: {_context_line(ctx)}\n"
        f"{_history_block(ctx)}"
    )
    current_code = str((ctx or {}).get("current_code") or "").strip()
    if current_code:
        prompt += f"\nCurrent draft strategy:\n{current_code}\n"
    prompt += f"\nUser: {msg}\nAssistant:"

    try:
        response = await studio_llm.generate_text(
            prompt=prompt,
            provider=(ctx or {}).get("llm_provider") or (ctx or {}).get("provider"),
            model=(ctx or {}).get("llm_model") or (ctx or {}).get("model"),
            timeout=float(os.getenv("STUDIO_LLM_TIMEOUT", "45") or 45),
            num_predict=int(os.getenv("STUDIO_CHAT_MAX_TOKENS", "320") or 320),
        )
        reply = response["text"].strip()
        return reply or strategy_studio_help_text()
    except Exception as exc:
        logging.error("Strategy Studio chat LLM error: %s", exc)
        return "The selected model could not answer right now. Change provider/model or check the API configuration."


def _seed_optimization_grid(task_type: str, strategy_effective: str, extra: dict) -> None:
    def _seed_int(key: str) -> Optional[int]:
        value = extra.get(key)
        if isinstance(value, list):
            if not value:
                return None
            value = value[0]
        try:
            return int(value)
        except Exception:
            return None

    def _seed_float(key: str) -> Optional[float]:
        value = extra.get(key)
        if isinstance(value, list):
            if not value:
                return None
            value = value[0]
        try:
            return float(value)
        except Exception:
            return None

    if task_type != "optimize":
        return

    try:
        wants_grid = any(isinstance(value, list) and len(value) > 1 for value in extra.values())
    except Exception:
        wants_grid = False
    if wants_grid:
        return

    if strategy_effective in ("sma",):
        fast_grid = list(range(10, 55, 5))
        slow_grid = list(range(60, 210, 10))
        seed_fast = _seed_int("fast")
        seed_slow = _seed_int("slow")
        if seed_fast is not None:
            fast_grid.append(max(2, seed_fast))
        if seed_slow is not None:
            slow_grid.append(max(2, seed_slow))
        extra["fast"] = sorted({int(value) for value in fast_grid})
        extra["slow"] = sorted({int(value) for value in slow_grid})
    elif strategy_effective in ("rsi",):
        period_grid = [10, 14, 20, 28]
        lower_grid = [20, 30, 40]
        upper_grid = [60, 70, 80]
        seed_period = _seed_int("period") or _seed_int("rsi")
        seed_lower = _seed_float("lower")
        seed_upper = _seed_float("upper")
        if seed_period is not None:
            period_grid.append(max(2, seed_period))
        if seed_lower is not None:
            lower_grid.append(float(seed_lower))
        if seed_upper is not None:
            upper_grid.append(float(seed_upper))
        extra["period"] = sorted({int(value) for value in period_grid})
        extra["lower"] = sorted({float(value) for value in lower_grid})
        extra["upper"] = sorted({float(value) for value in upper_grid})
    elif strategy_effective in ("smc",):
        zone_grid = [30, 50, 70]
        struct_grid = [3, 5, 8]
        ob_grid = [10, 20, 30]
        thr_grid = [1, 2, 3]
        seed_zone = _seed_int("zone_window")
        seed_struct = _seed_int("structure_lookback")
        seed_ob = _seed_int("ob_window")
        seed_thr = _seed_int("threshold")
        if seed_zone is not None:
            zone_grid.append(max(5, seed_zone))
        if seed_struct is not None:
            struct_grid.append(max(2, seed_struct))
        if seed_ob is not None:
            ob_grid.append(max(5, seed_ob))
        if seed_thr is not None:
            thr_grid.append(max(1, seed_thr))
        extra["zone_window"] = sorted({int(value) for value in zone_grid})
        extra["structure_lookback"] = sorted({int(value) for value in struct_grid})
        extra["ob_window"] = sorted({int(value) for value in ob_grid})
        extra["threshold"] = sorted({int(value) for value in thr_grid})


def _normalize_task_type(task_type: str, goal_text: str) -> str:
    goal_lower = goal_text.lower()
    if task_type == "save_strategy" or not goal_text:
        return task_type
    try:
        import re

        looks_like_opt_cmd = bool(
            re.match(r"\s*optimi[sz](e|ation|ing)?\b", goal_lower)
            or re.search(r"\b(can you|could you|please|plz|run|do|perform|execute|help me)\b.*\boptimi[sz](e|ation|ing)?\b", goal_lower)
        )
        looks_like_bt_cmd = bool(
            re.match(r"\s*backtest(ing)?\b", goal_lower)
            or re.search(r"\b(can you|could you|please|plz|run|do|perform|execute|help me)\b.*\bbacktest(ing)?\b", goal_lower)
        )
        if looks_like_opt_cmd:
            return "optimize"
        if looks_like_bt_cmd and task_type not in ("backtest_strategy", "backtest", "optimize"):
            return "backtest"
        if task_type in ("calculate_indicator", "create_strategy", "research_strategy") and re.search(
            r"\b(what can you do|help|commands?|capabilit(?:y|ies))\b", goal_lower
        ):
            return "chat"
        if task_type in ("calculate_indicator",) and not re.search(
            r"\b(indicator|code|script|function|generate|write)\b", goal_lower
        ):
            return "chat"
    except Exception:
        pass
    return task_type


async def execute_studio_task(request: StudioTaskRequest) -> StudioTaskResponse:
    task_type = (request.task_type or "").strip().lower()
    goal_text = (request.goal or "").strip()
    params = request.params or {}
    task_type = _normalize_task_type(task_type, goal_text)

    if task_type == "chat":
        ctx = params if isinstance(params, dict) else None
        current_code = str((ctx or {}).get("current_code") or "").strip()
        save_name = _extract_save_name(goal_text)
        if save_name:
            if not current_code:
                return StudioTaskResponse(status="error", message="There is no draft strategy to save yet.")
            return _save_strategy_code(save_name, current_code)
        if _is_code_request(goal_text, current_code):
            try:
                generated = await _generate_strategy_code(goal_text, ctx)
            except Exception as exc:
                return StudioTaskResponse(status="error", message=str(exc))
            action = "updated" if current_code else "created"
            if generated["provider"] == "template":
                message = f"Strategy draft {action} with built-in fallback after the selected model did not return code in time."
            else:
                message = f"Strategy draft {action} with {generated['provider']}:{generated['model']}."
            return StudioTaskResponse(
                status="success",
                message=message,
                result={"stdout": generated["text"], "provider": generated["provider"], "model": generated["model"]},
            )
        reply = await strategy_studio_chat_reply(goal_text, ctx)
        return StudioTaskResponse(status="success", message=reply)

    mapped = "indicator" if task_type in ("calculate_indicator",) else "backtest" if task_type in (
        "backtest_strategy",
        "backtest",
        "optimize",
    ) else "strategy"

    try:
        if task_type == "save_strategy":
            name = ((params or {}).get("strategy_name") or "").strip() if isinstance(params, dict) else ""
            code = ((params or {}).get("code") or "") if isinstance(params, dict) else ""
            if not name:
                return StudioTaskResponse(status="error", message="strategy_name is required")
            if not code:
                return StudioTaskResponse(status="error", message="code is required")
            return _save_strategy_code(name, code)

        if mapped == "backtest":
            symbol = (params.get("symbol") if isinstance(params, dict) else None) or DEFAULT_SYMBOL
            timeframe = (params.get("timeframe") if isinstance(params, dict) else None) or "M5"
            num_bars = int((params.get("num_bars") if isinstance(params, dict) else 1500) or 1500)
            extra = {}
            if isinstance(params, dict):
                extra.update(params)

            draft_code = str(extra.get("code") or "").strip() if isinstance(extra, dict) else ""
            if draft_code:
                if task_type == "optimize":
                    return StudioTaskResponse(status="error", message="Optimization for unsaved draft code is not supported yet.")
                try:
                    result = run_strategy_code_backtest(
                        code=draft_code,
                        symbol=symbol,
                        timeframe=timeframe,
                        num_bars=num_bars,
                        fee_bps=float(extra.get("fee_bps") or 0.0),
                        slippage_bps=float(extra.get("slippage_bps") or 0.0),
                        strategy_name=str(extra.get("strategy_name") or "draft"),
                    )
                except Exception as exc:
                    return StudioTaskResponse(status="error", message=str(exc))
                return StudioTaskResponse(status="success", message="Backtest complete.", result=result)

            if goal_text:
                try:
                    if (not isinstance(params, dict)) or not params.get("timeframe"):
                        match_tf = re.search(r"\b(M1|M5|M15|M30|H1|H4|D1|W1)\b", goal_text, flags=re.I)
                        if match_tf:
                            timeframe = match_tf.group(1).upper()
                    if (not isinstance(params, dict)) or not params.get("symbol"):
                        match_on = re.search(r"\bon\s+([A-Za-z0-9_]{3,20})\b", goal_text, flags=re.I)
                        match_pair = re.search(r"\b([A-Za-z]{3,10}USD)\b", goal_text, flags=re.I)
                        if match_on:
                            symbol = match_on.group(1).upper()
                        elif match_pair:
                            symbol = match_pair.group(1).upper()
                    if (not isinstance(params, dict)) or not params.get("num_bars"):
                        match_bars = re.search(r"(\d{2,9})\s*bars?\b", goal_text, flags=re.I)
                        if match_bars:
                            num_bars = int(match_bars.group(1))
                except Exception:
                    pass

            strategy_name = None
            if isinstance(params, dict):
                supplied = params.get("strategy_name") or params.get("strategy")
                if isinstance(supplied, str) and supplied.strip():
                    strategy_name = supplied.strip()
            if not strategy_name and goal_text:
                try:
                    import re

                    match = re.match(r"(?i)\s*(?:backtest|optimize)\s+([a-z0-9_-]+)\b", goal_text)
                    if match:
                        token = match.group(1).lower()
                        stop = {"the", "a", "an", "my", "this", "that", "strategy", "for", "on", "in", "to", "with"}
                        candidate = None
                        if token not in stop:
                            if token in ("sma", "smc"):
                                candidate = token
                            else:
                                path = Path("backend/strategies_generated") / f"{token}.py"
                                if path.exists():
                                    candidate = token
                        if candidate:
                            strategy_name = candidate
                except Exception:
                    pass

            if not strategy_name and goal_text:
                goal_lower = goal_text.lower()
                if "smc" in goal_lower:
                    strategy_name = "smc"
                elif "rsi" in goal_lower:
                    strategy_name = "rsi"
                elif "sma" in goal_lower:
                    strategy_name = "sma"

            if goal_text:
                extracted = extract_parameters(goal_text)
                if extracted:
                    extra.update(extracted)

            strategy_effective = strategy_name or extra.get("strategy_name") or extra.get("strategy") or "sma"
            try:
                strategy_effective = str(strategy_effective).strip().lower()
            except Exception:
                strategy_effective = "sma"
            _seed_optimization_grid(task_type, strategy_effective, extra)

            try:
                if "fees" not in extra and "fee_bps" in extra:
                    extra["fees"] = float(extra.get("fee_bps") or 0.0) / 10_000.0
                if "slippage" not in extra and "slippage_bps" in extra:
                    extra["slippage"] = float(extra.get("slippage_bps") or 0.0) / 10_000.0
            except Exception:
                pass

            raw_strategy = strategy_name or (extra.get("strategy_name") if isinstance(extra, dict) else None) or (
                extra.get("strategy") if isinstance(extra, dict) else None
            )
            if not isinstance(raw_strategy, str):
                raw_strategy = None
            backtest_params = BacktestParams(
                symbol=symbol,
                timeframe=timeframe,
                num_bars=num_bars,
                strategy_name=(raw_strategy.strip() if raw_strategy else "sma"),
            )
            result = run_backtest(backtest_params, extra_params=extra)
            if isinstance(result, dict) and result.get("error"):
                return StudioTaskResponse(
                    status="error",
                    message=str(result.get("error") or "Backtest failed."),
                    result=result,
                )
            return StudioTaskResponse(
                status="success",
                message=(result.get("message", "Backtest complete.") if isinstance(result, dict) else "Backtest complete."),
                result=result,
            )

        if mapped == "strategy":
            generated = await _generate_strategy_code(goal_text, params if isinstance(params, dict) else None)
            message = (
                "Code generated with built-in fallback after the selected model did not return code in time."
                if generated["provider"] == "template"
                else f"Code generated with {generated['provider']}:{generated['model']}."
            )
            return StudioTaskResponse(
                status="success",
                message=message,
                result={"stdout": generated["text"], "provider": generated["provider"], "model": generated["model"]},
            )

        programmer = ProgrammerAgent()
        code = await programmer.generate_code(goal_text, mapped)
        return StudioTaskResponse(status="success", message="Code generated.", result={"stdout": code})
    except Exception as exc:
        return StudioTaskResponse(status="error", message=str(exc))
