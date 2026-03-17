
import asyncio
from playwright.async_api import async_playwright
import time
import re
import sys
import os

# Target file to modify
STRATEGY_FILE = "backend/strategies_generated/meme_coin_short.py"

# Optimization Range
# We will toggle between "Mean Reversion" (RSI Cross) and "Trend Following" (Tight EMAs)
PARAMS = [
    # 1. Conservative Trend (Slow EMA26, Filter RSI<45)
    {"ema_fast_len": 9, "ema_slow_len": 26, "rsi_long_threshold": 45.0, "rsi_short_threshold": 55.0},
    # 2. Aggressive Trend (Fast EMA12, Filter RSI<55)
    {"ema_fast_len": 5, "ema_slow_len": 15, "rsi_long_threshold": 55.0, "rsi_short_threshold": 45.0},
    # 3. Mean Reversion High Volatility (Deep RSI)
    {"ema_fast_len": 50, "ema_slow_len": 100, "rsi_long_threshold": 30.0, "rsi_short_threshold": 70.0},
    # 4. Balanced (Current Best)
    {"ema_fast_len": 9, "ema_slow_len": 21, "rsi_long_threshold": 40.0, "rsi_short_threshold": 60.0},
    # 5. Super Fast Scalp
    {"ema_fast_len": 3, "ema_slow_len": 8, "rsi_long_threshold": 50.0, "rsi_short_threshold": 50.0}
]

def update_strategy_file(params):
    """
    Update the strategy file with specific parameters.
    """
    try:
        with open(STRATEGY_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for param, value in params.items():
            pattern = f"({param}\\s*:\\s*(?:int|float)\\s*=\\s*)(\\d+(?:\\.\\d+)?)"
            new_content = re.sub(pattern, f"\\g<1>{value}", new_content)
            
        with open(STRATEGY_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"File Update Error: {e}")
        return False

async def run_visual_optimization():
    print("Launching Visual Optimizer...")
    print("This will open the browser and iterate through strategy settings.")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=50) # Visible speed
        page = await browser.new_page()
        
        try:
            await page.goto("http://localhost:8080/strategy-studio", timeout=15000)
        except:
            print("Failed to load dashboard. Is server running?")
            return

        # Initial Setup
        await page.wait_for_timeout(2000)
        
        # Select Strategy once
        try:
            await page.get_by_title("Saved Strategy").select_option(value="meme_coin_short")
            await page.get_by_title("Symbol").fill("XAUUSD")
            await page.get_by_title("Timeframe").select_option("H1") # Test on H1 for trend stability
        except:
            print("UI Setup Failed")
            
        best_ret = -999.0
        best_idx = -1
        
        for i, config in enumerate(PARAMS):
            print(f"\n--- Testing Config {i+1}/{len(PARAMS)} ---")
            print(f"Params: {config}")
            
            # 1. Update Code in Background
            update_strategy_file(config)
            
            # 2. Wait for reload (small delay)
            await page.wait_for_timeout(1000)
            
            # 3. Click Run (Visual)
            try:
                run_btn = page.get_by_text("Run Backtest")
                await run_btn.click()
            except Exception as e:
                print(f"Click failed: {e}")
                continue
                
            # 4. Wait for Result
            # Look for spinner to disappear or result text to update?
            # Simple wait is safest for visual demo
            await page.wait_for_timeout(3000)
            
            # 5. Scrape Result
            # We look for the "Total Return [%]" cell text
            # This is hard to robustly scrape generically, but we try:
            ret = -999.0
            try:
                # Find the text that follows "Total Return [%]"
                # Or grab the raw json view?
                # Let's switch to Raw JSON to be sure
                await page.get_by_text("Raw JSON").click()
                await page.wait_for_timeout(500)
                
                # Get code block text
                json_text = await page.locator("pre").inner_text()
                
                # Parse regex
                import json
                try:
                    data = json.loads(json_text)
                    ret = float(data.get("Total Return [%]", -999.0))
                    print(f"  -> RETURN: {ret}%")
                except:
                    # messy parse
                    m = re.search(r'"Total Return \[%\]":\s*([-\d\.]+)', json_text)
                    if m:
                        ret = float(m.group(1))
                        print(f"  -> RETURN: {ret}% (regex)")
                
                # Switch back to Formatted for visuals
                await page.get_by_text("Formatted").click()
                
            except Exception as e:
                print(f"Scrape failed: {e}")
                
            if ret > best_ret:
                best_ret = ret
                best_idx = i
                print("  => NEW BEST!")
                
        print("\nOptimization Complete.")
        print(f"Best Config: #{best_idx+1} with Return {best_ret}%")
        
        # Restore Best
        if best_idx >= 0:
            print("Restoring Best Configuration...")
            update_strategy_file(PARAMS[best_idx])
            # Run one last time to show user the best result
            await page.get_by_text("Run Backtest").click()
            
        print("Done. Leaving browser open for inspection.")
        await page.wait_for_timeout(30000) # 30s wait
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_visual_optimization())
