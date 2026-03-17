import vectorbt as vbt
import pandas as pd
import numpy as np
import json

try:
    print("Testing VectorBT Plotly generation...")
    price = pd.Series([10, 11, 10, 12, 13], name="Close")
    signals = pd.Series([1, 0, -1, 0, 1], name="Signal")
    pf = vbt.Portfolio.from_orders(price, signals, size_type='targetpercent')
    print("Portfolio created.")
    fig = pf.plot()
    print("Plot created.")
    js = fig.to_json()
    print(f"JSON generated. Length: {len(js)}")
    with open("test_plot.json", "w") as f:
        f.write(js)
    print("Success.")
except Exception as e:
    print(f"FAILED: {e}")
