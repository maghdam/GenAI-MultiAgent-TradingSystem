import re
from typing import List, Union, Dict, Any
import numpy as np

def parse_range(text: str) -> Union[List[int], List[float], None]:
    """
    Parses natural language ranges into a list of numbers.
    Examples:
    - "10 to 50 step 10" -> [10, 20, 30, 40, 50]
    """
    text = text.lower().strip()
    
    # Check for "X to Y step Z" pattern
    step_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*step\s*(\d+(?:\.\d+)?)', text)
    if step_match:
        start, end, step = map(float, step_match.groups())
        # Use arange but be careful with floating point
        vals = []
        curr = start
        while curr <= end + 0.000001:
            vals.append(curr)
            curr += step
            
        if all(x.is_integer() for x in [start, end, step]):
             return [int(round(v)) for v in vals]
        return [float(v) for v in vals]

    # Check for comma separated
    if ',' in text:
        try:
            return [float(x.strip()) if '.' in x else int(x.strip()) for x in text.split(',')]
        except:
            pass
            
    # Single number
    try:
        val = float(text)
        return [int(val)] if val.is_integer() else [val]
    except:
        return None

def extract_parameters(query: str) -> Dict[str, Any]:
    """
    Extracts variable parameters from a query string for optimization.
    """
    params = {}
    keywords = [
        # Common indicator/strategy params
        'fast', 'slow', 'length', 'period', 'lookback', 'signal', 'rsi', 'ma',
        'lower', 'upper', 'mult',
        # SMC strategy params (backend/strategies_generated/smc.py)
        'zone_window', 'structure_lookback', 'ob_window', 'trend_window', 'fvg_shift', 'ob_distance_pct', 'threshold',
        # Risk helpers / other strategy params
        'rr', 'atr_len', 'atr_mult', 'swing_lookback', 'tick_pct',
    ]
    
    # Simple regex to find "keyword <value>" where value can be "10 to 50 step 5"
    found_kws = []
    lower_q = query.lower()
    for kw in keywords:
        iter = re.finditer(rf'\b{kw}\b', lower_q)
        for m in iter:
            found_kws.append((m.start(), kw))
            
    found_kws.sort()
    
    for i, (start, kw) in enumerate(found_kws):
        end = found_kws[i+1][0] if i+1 < len(found_kws) else len(lower_q)
        segment = lower_q[start+len(kw):end].strip()
        parsed = parse_range(segment)
        if parsed is not None:
             params[kw] = parsed
             
    # SMA shorthand: "20/50" or "20 / 50" -> fast=20 slow=50 (only if SMA is mentioned).
    try:
        if "sma" in lower_q and ("fast" not in params or "slow" not in params):
            m = re.search(r"\b(\d{1,4})\s*/\s*(\d{1,4})\b", lower_q)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                if 2 <= a <= 2000 and 2 <= b <= 2000:
                    f, s = (a, b) if a <= b else (b, a)
                    params.setdefault("fast", [f])
                    params.setdefault("slow", [s])
    except Exception:
        pass

    return params
