# market/condition_checker.py
from bot.config import MIN_24H_VOLUME_USDT, MAX_SPREAD_PERCENT, MIN_VOLATILITY_PERCENT
from .data_fetcher import get_ticker
from bot.logger_config import logger

def check_market_conditions(instId):
    """
    Checks market conditions based on ticker data.
    Returns True if conditions are met, False otherwise.
    """
    ticker_data = get_ticker(instId)
    if not ticker_data:
        logger.warning(f"Could not fetch ticker data for {instId} to check conditions.")
        return False # Cannot assess conditions without data

    try:
        volume_24h = float(ticker_data.get('volCcy24h', 0))
        last_price = float(ticker_data.get('last', 0))
        best_bid = float(ticker_data.get('bidPx', 0))
        best_ask = float(ticker_data.get('askPx', 0))
        high_24h = float(ticker_data.get('high24h', 0))
        low_24h = float(ticker_data.get('low24h', 0))

        # --- Check Volume ---
        if volume_24h < MIN_24H_VOLUME_USDT:
            logger.warning(f"24h volume too low for {instId}: ${volume_24h:,.2f} < ${MIN_24H_VOLUME_USDT:,.2f}")
            return False

        # --- Check Spread ---
        if best_bid <= 0 or best_ask <= 0 or last_price <= 0:
             logger.warning(f"Invalid price data for spread check for {instId}: bid={best_bid}, ask={best_ask}, last={last_price}")
             return False # Avoid division by zero or negative prices

        spread = (best_ask - best_bid) / best_bid
        max_spread_decimal = MAX_SPREAD_PERCENT / 100.0
        if spread > max_spread_decimal:
            logger.warning(f"Spread too high for {instId}: {spread*100:.3f}% > {MAX_SPREAD_PERCENT:.3f}%")
            return False

        # --- Check Volatility ---
        if low_24h <= 0 or high_24h <= 0: # Avoid division by zero
             logger.warning(f"Invalid price data for volatility check for {instId}: low={low_24h}, high={high_24h}")
             return False

        volatility = (high_24h - low_24h) / low_24h
        min_volatility_decimal = MIN_VOLATILITY_PERCENT / 100.0
        if volatility < min_volatility_decimal:
            logger.warning(f"24h volatility too low for {instId}: {volatility*100:.3f}% < {MIN_VOLATILITY_PERCENT:.3f}%")
            return False

        logger.info(f"Market conditions good for {instId}: "
                    f"Vol=${volume_24h:,.0f}, Spread={spread*100:.3f}%, Volatility={volatility*100:.3f}%")
        return True

    except (ValueError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Error processing market data for {instId} conditions check: {e}")
        logger.debug(f"Ticker data: {ticker_data}")
        return False
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error checking market conditions for {instId}: {e}")
        return False