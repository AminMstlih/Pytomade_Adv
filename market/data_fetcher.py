# market/data_fetcher.py
import pandas as pd
from exchange.okx_client import get_public_data
from bot.logger_config import logger

def get_ticker(instId):
    """Fetches ticker data for a given instrument."""
    request_path = f"/api/v5/market/ticker?instId={instId}"
    response = get_public_data(request_path)
    if response['code'] == '0' and response['data']:
        return response['data'][0] # Return the first (and usually only) ticker data dict
    else:
        logger.error(f"Failed to fetch ticker for {instId}: {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
        return None

def get_historical_candles(instId, bar="1m", limit=1000):
    """
    Fetches historical candle data.
    Returns a pandas DataFrame or None on failure.
    """
    request_path = f"/api/v5/market/candles?instId={instId}&bar={bar}&limit={limit}"
    response = get_public_data(request_path)
    if response['code'] == '0' and response['data']:
        candles = response['data']
        # OKX returns candles in descending order (newest first), reverse for chronological
        candles.reverse() 
        
        if len(candles) == 0:
             logger.warning(f"No candles returned for {instId}")
             return None

        try:
            df = pd.DataFrame({
                'timestamp': [float(c[0]) for c in candles],
                'open': [float(c[1]) for c in candles],
                'high': [float(c[2]) for c in candles],
                'low': [float(c[3]) for c in candles],
                'close': [float(c[4]) for c in candles],
                'volume': [float(c[5]) for c in candles],
                'volCcy': [float(c[6]) for c in candles] # Volume in quote currency (USDT)
            })
            
            # Basic data validation
            if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any() or (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
                logger.warning(f"Invalid candle data for {instId}: contains NaN or non-positive values")
                return None
                
            logger.debug(f"Fetched {len(candles)} candles for {instId}. Latest close: ${df['close'].iloc[-1]:.6f}")
            return df
        except (ValueError, IndexError, KeyError) as e:
            logger.error(f"Error processing candle data for {instId}: {e}")
            return None
    else:
        logger.error(f"Failed to fetch candles for {instId}: {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
        return None