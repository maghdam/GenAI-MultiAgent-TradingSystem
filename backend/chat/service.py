import logging
import json
from dataclasses import asdict
from backend import data_fetcher, ctrader_client as ctd
from backend.strategy import get_strategy
from backend.agent_controller import controller, AgentConfig
from backend.llm_analyzer import _ollama_generate, MODEL_DEFAULT, TradeDecision
from backend.journal import db as journal_db
from backend import web_search

# In-memory store for pending trade confirmations
_pending_trades = {}

async def _get_intent(message: str) -> dict:
    """Use an LLM to determine the user's intent and extract entities."""
    prompt = f"""You are a helpful trading assistant. Your job is to determine the user's intent and extract relevant information from their message. 

The user said: '{message}'

Possible intents are 'get_price', 'run_analysis', 'start_agents', 'stop_agents', 'get_agent_status', 'place_order', 'confirm_action', 'cancel_action', 'get_news', 'help', or 'unknown'.

- If the intent is 'get_price', extract the 'symbol'.
- If the intent is 'run_analysis', extract the 'symbol', 'timeframe', and 'strategy'.
- If the intent is 'place_order', extract the 'symbol', 'direction' (buy/sell), and 'volume' (in lots).
- If the intent is 'get_news', extract the 'topic' or 'symbol'.

Respond with ONLY a single JSON object in the following format and nothing else:
{{"intent": "<intent>", "symbol": "<symbol or null>", "topic": "<topic or null>", "timeframe": "<timeframe or null>", "strategy": "<strategy or null>"}}"""

    try:
        response_text = _ollama_generate(
            prompt=prompt,
            model=MODEL_DEFAULT, 
            timeout=120, # A longer timeout for chat responses
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

async def process_message(message: str, websocket) -> str:
    """Processes an incoming chat message and returns a response."""
    logging.info(f"Processing chat message: '{message}'")

    # First, check if this is a confirmation for a pending trade
    client_id = websocket.client.host
    if client_id in _pending_trades and message.lower() in ["yes", "y"]:
        trade_details = _pending_trades.pop(client_id)
        try:
            symbol_id = ctd.symbol_name_to_id[trade_details['symbol'].upper()]
            volume_units = ctd.volume_lots_to_units(symbol_id, trade_details['volume'])

            deferred = ctd.place_order(
                client=ctd.client,
                account_id=ctd.ACCOUNT_ID,
                symbol_id=symbol_id,
                order_type="MARKET",
                side=trade_details['direction'].upper(),
                volume=volume_units
            )
            result = ctd.wait_for_deferred(deferred, timeout=15)

            # Journal the trade
            journal_db.add_trade_entry(
                symbol=trade_details['symbol'],
                direction=trade_details['direction'].upper(),
                volume=trade_details['volume'],
                entry_price=result.get('price'),
                rationale="Trade placed via chatbot"
            )

            return f"✅ Trade executed successfully! Details: {result}"
        except Exception as e:
            logging.error(f"Chat: Error executing trade: {e}")
            return f"❌ Failed to execute trade. Error: {e}"
    elif client_id in _pending_trades:
        _pending_trades.pop(client_id) # Clear pending trade on any other message
        return "Trade cancelled."
    
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

    elif intent == "start_agents":
        current_config = controller.config
        new_config = AgentConfig(**asdict(current_config))
        new_config.enabled = True
        await controller.apply_config(new_config)
        return "🤖 Agents have been enabled. They will start running based on the current watchlist."

    elif intent == "stop_agents":
        await controller.stop_all()
        current_config = controller.config
        new_config = AgentConfig(**asdict(current_config))
        new_config.enabled = False
        await controller.apply_config(new_config)
        return "🛑 Agents have been disabled and all running tasks have been stopped."

    elif intent == "place_order":
        symbol = intent_data.get("symbol")
        direction = intent_data.get("direction")
        volume = intent_data.get("volume")

        if not all([symbol, direction, volume]):
            return "I can place a trade, but I need a symbol, direction (buy/sell), and volume (in lots)."

        try:
            volume = float(volume)
        except (ValueError, TypeError):
            return f"The volume '{volume}' doesn't look like a valid number."

        # Store the pending trade and ask for confirmation
        _pending_trades[websocket.client.host] = {
            "symbol": symbol,
            "direction": direction,
            "volume": volume
        }
        return f"You want to {direction.upper()} {volume} lots of {symbol.upper()}. Is this correct? (yes/no)"

    elif intent == "get_news":
        topic = intent_data.get("topic") or intent_data.get("symbol")
        # If no specific topic, default to a general one
        if not topic:
            topic = "global financial markets"
        
        # This can be slow, so let the user know we're working on it
        await websocket.send_text(f"Searching for news on '{topic}'... This may take a moment.")
        
        summary = await web_search.get_news_summary(topic)
        return summary

    elif intent == "get_agent_status":
        snap = await controller.snapshot()
        cfg = snap.get("config") or {}
        running_pairs = snap.get("running_pairs") or []
        enabled = cfg.get("enabled", False)

        status_text = "✅ Enabled" if enabled else "❌ Disabled"
        pairs_text = ', '.join([f'{s}/{tf}' for s, tf in running_pairs]) if running_pairs else "none"
        watchlist_text = ', '.join([f'{s}/{tf}' for s, tf in cfg.get('watchlist', [])]) if cfg.get('watchlist') else "not set"


        return f"""Agent Status:
- Main switch: {status_text}
- Running pairs: {pairs_text}
- Watchlist: {watchlist_text}
- Strategy: {cfg.get('strategy')}
- Confidence threshold: {cfg.get('min_confidence')}"""

    elif intent == "help":
        return """I can help with the following:
- Get asset prices (e.g., 'price of gold?')
- Run analysis (e.g., 'analyze EURUSD on H1 with smc')
- Start, stop, or get the status of the trading agents.
- Execute trades with confirmation (e.g., 'buy 0.1 lots of EURUSD')
"""
    
    elif intent == "error":
        return f"Sorry, I had trouble understanding that. The AI model reported an error: {intent_data.get('detail')}"

    else: # 'unknown' or anything else
        return "I'm not sure how to help with that yet. You can ask me for the price of a symbol, like 'price of EURUSD'."