
import sys
import os
import pandas as pd
import numpy as np
import itertools

# Fix path to allow backend imports
sys.path.append(os.path.join(os.getcwd()))

try:
    import vectorbt as vbt
    from backend import data_fetcher
    from backend.strategies_generated import meme_coin_short
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    # Cannot proceed without vectorbt or strategy
    sys.exit(1)

def generate_synthetic_data(n=1000):
    """Generate synthetic crypto-like data for fallback."""
    np.random.seed(42)
    # Random walk with some volatility
    returns = np.random.normal(0, 0.005, n)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create some trending/pumping periods
    # Force a pump at index 200-250
    price[200:250] = np.linspace(price[200], price[200]*1.4, 50) + np.random.normal(0, 1, 50)
    # Force a dump at 250-300
    price[250:300] = np.linspace(price[250], price[250]*0.7, 50) + np.random.normal(0, 1, 50)
    
    df = pd.DataFrame({
        "open": price, 
        "high": price * 1.01, 
        "low": price * 0.99, 
        "close": price,
        "volume": np.random.randint(100, 1000, n)
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    return df

def run_optimization():
    print("Starting Meme Coin Strategy Optimization...")
    
    # 1. Get Data
    symbol = "XAUUSD"
    print(f"Fetching data for {symbol}...")
    try:
        # Try fetching real data (assuming 5000 bars 1H)
        # Note: data_fetcher.fetch_data returns (df, live_price)
        result = data_fetcher.fetch_data(symbol, "M15", 5000)
        df = result[0]
        if df is None or df.empty:
            print(f"Real data fetch returned empty. Using synthetic data.")
            df = generate_synthetic_data()
    except Exception as e:
        print(f"Data fetch error: {e}. Using synthetic data.")
        df = generate_synthetic_data()

    print(f"Data Loaded: {len(df)} candles.")

    # 2. Define Parameter Ranges
    rsi_thresholds = [50.0, 55.0, 60.0]
    adx_thresholds = [15.0, 20.0, 25.0]
    ema_fasts = [9, 12]
    # Limit slow to reduce combinations
    ema_slows = [26]
    
    combinations = list(itertools.product(rsi_thresholds, adx_thresholds, ema_fasts, ema_slows))
    print(f"Testing {len(combinations)} combinations...")
    
    results = []
    
    for rsi_t, adx_t, fast, slow in combinations:
        if fast >= slow: continue
        
        try:
            signals = meme_coin_short.signals(
                df,
                ema_fast_len=fast,
                ema_slow_len=slow,
                rsi_threshold=rsi_t,
                adx_threshold=adx_t
            )
            
            # VectorBT Backtest
            # -1 = Short, 0 = Cash.
            # Using Portfolio.from_orders with 'targetpercent' handles the rebalancing.
            pf = vbt.Portfolio.from_orders(
                close=df["close"],
                size=signals,
                size_type='targetpercent',
                freq='1h',
                init_cash=10000,
                fees=0.001,
                slippage=0.001
            )
            
            # Check if any trades happened
            if len(pf.trades.records) == 0:
                 # Strategy didn't trade
                 tres = 0.0
                 sharpe = 0.0
                 mdd = 0.0
                 wr = 0.0
            else:
                tres = pf.total_return() * 100
                sharpe = pf.sharpe_ratio()
                mdd = pf.max_drawdown() * 100
                wr = pf.trades.win_rate() * 100
            
            res = {
                "RSI_Thresh": rsi_t,
                "ADX_Thresh": adx_t,
                "Fast_EMA": fast,
                "Slow_EMA": slow,
                "Total Return [%]": tres,
                "Sharpe": sharpe,
                "Max DD [%]": mdd,
                "Win Rate [%]": wr
            }
            results.append(res)
            
        except Exception as e:
            # print(f"Error with params {rsi_t}, {adx_t}: {e}")
            pass
            
    # 3. Sort and Report
    if not results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(results)
    results_df.sort_values(by="Total Return [%]", ascending=False, inplace=True)
    
    print("\nTop 5 Parameter Configurations:")
    print(results_df.head(5).to_string(index=False))
    
    # Save results to user?
    # Just print for now.
    
    best = results_df.iloc[0]
    print("\nBest Configuration:")
    print(best.to_string())

if __name__ == "__main__":
    run_optimization()
