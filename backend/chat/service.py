import logging
import json
from backend import data_fetcher
from backend.strategy import get_strategy
from backend.llm_analyzer import _ollama_generate, MODEL_DEFAULT, TradeDecision

async def _get_intent(message: str) -> dict:
    """Use an LLM to determine the user's intent and extract entities."""
    prompt = f"""You are a helpful trading assistant. Your job is to determine the user's intent and extract relevant information from their message. 

The user said: '{message}'

Possible intents are 'get_price', 'run_analysis', 'help', or 'unknown'.

- If the intent is 'get_price', extract the 'symbol'.
- If the intent is 'run_analysis', extract the 'symbol', 'timeframe', and 'strategy'.

Respond with ONLY a single JSON object in the following format and nothing else:
{{"intent": "<intent>", "symbol": "<symbol or null>", "timeframe": "<timeframe or null>", "strategy": "<strategy or null>"}}"""

    try:
        response_text = _ollama_generate(
            prompt=prompt,
            model=MODEL_DEFAULT, 
            timeout=15, # A shorter timeout for quick chat responses
            json_only=True,
            options_overrides={"num_predict": 48}
        )
        return json.loads(response_text)
    except Exception as e:
        logging.error(f"Error getting intent from LLM: {e}")
        return {"intent": "error", "detail": str(e)}

async def _summarize_analysis(analysis: TradeDecision) -> str:
    """Use an LLM to generate a human-readable summary of the analysis."""
    if not analysis.reasons:
        return "The analysis did not provide specific reasons for its decision."

    reasons_text = '\n'.join([f'- {r}' for r in analysis.reasons])
    prompt = f"""You are a trading analyst. Your colleague has produced the following trading signal and technical reasons.

Signal: {analysis.signal}
Confidence: {analysis.confidence}
SL: {analysis.sl}
TP: {analysis.tp}

Technical Reasons:
{reasons_text}

Your task is to write a concise, one-paragraph explanation of this trade idea for a report. Combine the technical reasons into a fluent, easy-to-understand narrative. Do not just list the reasons; synthesize them."""

    try:
        summary = _ollama_generate(
            prompt=prompt,
            model=MODEL_DEFAULT,
            timeout=45, # Longer timeout for a more detailed response
            json_only=False, # We want a text response
            options_overrides={"num_predict": 128}
        )
        return summary
    except Exception as e:
        logging.error(f"Error summarizing analysis: {e}")
        return "I was unable to generate a summary for the analysis."

async def process_message(message: str) -> str:
    """Processes an incoming chat message and returns a response."""
    logging.info(f"Processing chat message: '{message}'")
    
    intent_data = await _get_intent(message)
    intent = intent_data.get("intent")

    if intent == "get_price":
        symbol = intent_data.get("symbol")
        if not symbol:
            return "I see you want a price, but which symbol? Please be more specific."
        
        try:
            df, _ = data_fetcher.fetch_data(symbol.upper(), "M1", 2)
            if df.empty:
                return f"Sorry, I couldn't find any data for {symbol.upper()}."
            latest_price = df['close'].iloc[-1]
            return f"The latest price for {symbol.upper()} is {latest_price:.5f}."
        except Exception as e:
            logging.error(f"Chat: Error fetching price for {symbol}: {e}")
            return f"I ran into an error trying to get the price for {symbol.upper()}."

    elif intent == "run_analysis":
        symbol = intent_data.get("symbol")
        if not symbol:
            return "I can run an analysis, but I need to know for which symbol."
        
        # Provide defaults for timeframe and strategy if not extracted
        timeframe = intent_data.get("timeframe") or "H1"
        strategy_name = intent_data.get("strategy") or "smc"

        try:
            df, _ = data_fetcher.fetch_data(symbol.upper(), timeframe, 1000)
            if df.empty:
                return f"Sorry, I couldn't get any data to analyze for {symbol.upper()}."

            strategy = get_strategy(strategy_name)
            analysis_result = await strategy.analyze(df=df, symbol=symbol.upper(), timeframe=timeframe)

            # Format the result for the user
            sig = analysis_result.signal.replace('_', ' ').title()
            conf = f"{analysis_result.confidence * 100:.0f}%" if analysis_result.confidence is not None else "N/A"

            # NEW: Generate a human-readable summary
            summary = await _summarize_analysis(analysis_result)

            return f"""Analysis Complete for {symbol.upper()}:

Signal: {sig}
Confidence: {conf}
SL: {analysis_result.sl}
TP: {analysis_result.tp}

Summary:
{summary}"""

        except Exception as e:
            logging.error(f"Chat: Error running analysis: {e}")
            return f"I hit an error while trying to analyze {symbol.upper()}: {e}"

    elif intent == "help":
        return """I can help with the following:
- Get asset prices (e.g., 'price of gold?')
- Run analysis (e.g., 'analyze EURUSD on H1 with smc')
"""
    
    elif intent == "error":
        return f"Sorry, I had trouble understanding that. The AI model reported an error: {intent_data.get('detail')}"

    else: # 'unknown' or anything else
        return "I'm not sure how to help with that yet. You can ask me for the price of a symbol, like 'price of EURUSD'."