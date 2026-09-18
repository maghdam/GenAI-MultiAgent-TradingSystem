# Confluence Decision Replay

The decision replay compares the stored original technical policy with the event-adjusted shadow policy. It is research-only and cannot submit an order or change engine configuration.

## What is replayed

The input is the immutable `confluence_shadow_records` dataset produced during manual and automatic analysis. This preserves the information that was actually available when each decision occurred; the replay does not reconstruct historical event context with future data.

For each directional decision:

1. Entry is the open of the first bar strictly after the analysis timestamp.
2. Exit is the close of the first bar at or after its stored horizon (`5m`, `30m`, `4h`, or `1d`).
3. Configured transaction cost is deducted on entry and exit.
4. Repeated scans cannot open overlapping positions for the same symbol, timeframe, and strategy.
5. Original and shadow threshold policies are simulated independently.

The report compares trades, win rate, average trade expectancy, compounded cohort return, maximum drawdown, and trades per day. This is a decision-cohort study—not a broker-fill simulation or a portfolio backtest.

## Promotion gate

The system always returns `insufficient_data` until it has at least 100 priced directional decisions and at least 30 simulated trades under each policy. After that, the shadow policy remains observation-only unless it improves expectancy without worsening maximum drawdown. Passing these rules produces only `candidate_for_review`; it never activates the policy automatically.

Use a genuinely later, out-of-sample period for the final review. Paper execution should remain on the original strategy until that review is complete.
