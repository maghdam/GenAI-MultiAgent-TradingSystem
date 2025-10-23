import textwrap
from typing import Literal

TaskKind = Literal["indicator", "backtest", "strategy"]


class ProgrammerAgent:
    """Very lightweight code generator stub.

    In a full setup this would call an LLM to synthesize code. For now we
    return concise, readable snippets that the UI can show in the results view.
    """

    async def generate_code(self, goal: str, task_type: TaskKind) -> str:
        goal = (goal or "").strip()
        if task_type == "indicator":
            return textwrap.dedent(
                f"""
                # Example: Simple RSI (14) using pandas
                import pandas as pd

                def rsi(series: pd.Series, period: int = 14) -> pd.Series:
                    delta = series.diff()
                    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
                    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
                    rs = gain / loss.replace(0, 1e-9)
                    return 100 - (100 / (1 + rs))

                # Usage: df['rsi14'] = rsi(df['close'], 14)
                """.strip()
            )

        if task_type == "strategy":
            return textwrap.dedent(
                f"""
                # Example: SMA crossover strategy skeleton
                import pandas as pd

                def sma(series: pd.Series, n: int) -> pd.Series:
                    return series.rolling(n, min_periods=n).mean()

                def signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
                    f = sma(df['close'], fast)
                    s = sma(df['close'], slow)
                    sig = (f > s).astype(int).diff().fillna(0)
                    # 1 = long entry, -1 = exit/short entry depending on rules
                    return sig
                """.strip()
            )

        # For backtest requests, return a helper snippet to show intent
        return textwrap.dedent(
            f"""
            # Backtest outline (pseudocode):
            # - Fetch OHLCV data
            # - Generate signals
            # - Iterate bars and track PnL with simple costs
            # - Compute metrics (total return, win rate, max drawdown)
            """.strip()
        )

