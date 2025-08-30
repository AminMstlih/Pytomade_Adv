# market/strategy.py
import math
import pandas as pd
import numpy as np
from scipy.stats import linregress
from bot.config import (
    MA_FAST_PERIOD, MA_SLOW_PERIOD, RSI_STOCH_PERIOD, STOCH_K_PERIOD, STOCH_D_PERIOD,
    ATR_PERIOD, ADX_PERIOD, ADX_THRESHOLD, VOLUME_THRESHOLD_MULTIPLIER
)
from bot.logger_config import logger

# --- Indicator Calculation Functions ---
# (These helper functions remain the same as before)
def _ema(series, period):
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def _rsi(series, period=14):
    """
    Calculates RSI using Wilder's method to match TradingView/OKX.
    Uses EMA-like smoothing (ewm with com=period-1) after initial SMA seed.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use ewm with com=period-1 (alpha=1/period) and min_periods=period to seed with SMA
    avg_gain = gain.ewm(com=period-1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Fill NaNs with 50 (neutral) for early periods
    return pd.Series(rsi, index=series.index)

def _stochrsi(series, rsi_length=5, stoch_length=5, k=3, d=3):
    """Calculate StochRSI indicator."""
    rsi_val = _rsi(series, rsi_length)
    lowest_rsi = rsi_val.rolling(window=stoch_length, min_periods=stoch_length).min()
    highest_rsi = rsi_val.rolling(window=stoch_length, min_periods=stoch_length).max()
    
    # Handle cases where highest_rsi equals lowest_rsi
    stoch_k = np.where(
        highest_rsi == lowest_rsi, 
        50, 
        100 * (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi)
    )
    
    stoch_d = pd.Series(stoch_k, index=series.index).rolling(window=d, min_periods=d).mean()
    stoch_d = stoch_d.bfill().ffill()
    return pd.Series(stoch_k, index=series.index), stoch_d

def _true_range(high, low, close):
    """Calculate True Range for ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _atr(high, low, close, period=14):
    """Calculate Average True Range using simple moving average."""
    true_range = _true_range(high, low, close)
    return true_range.rolling(window=period, min_periods=period).mean()

def _rma(series, period):
    """Wilder's RMA used by many Pine indicators (including ATR)."""
    return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def _atr_rma(high, low, close, period=14):
    """ATR using Wilder's RMA to better match PineScript behavior."""
    tr = _true_range(high, low, close)
    return _rma(tr, period)

def _supertrend(high, low, close, factor=2, atr_period=15):
    """
    Supertrend calculation adapted from Pine-style logic.
    Returns (supertrend_series, direction_series) where direction is 1 (up) or -1 (down).
    """
    atr = _atr_rma(high, low, close, atr_period)
    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)

    first_valid = atr.first_valid_index()
    if first_valid is None:
        return st, direction

    upper_band = (high + factor * atr).ffill()
    lower_band = (low - factor * atr).ffill()

    # Vectorized implementation for better performance
    close_arr = close.values
    upper_band_arr = upper_band.values
    lower_band_arr = lower_band.values
    st_arr = np.full(len(close_arr), np.nan)
    direction_arr = np.ones(len(close_arr))
    
    # Find the starting index
    start_idx = close.index.get_loc(first_valid)
    
    # Initialize first value
    if start_idx < len(close_arr):
        direction_arr[start_idx] = 1 if close_arr[start_idx] > upper_band_arr[start_idx] else -1
        st_arr[start_idx] = lower_band_arr[start_idx] if direction_arr[start_idx] == 1 else upper_band_arr[start_idx]
    
    # Process the rest of the values
    for i in range(start_idx + 1, len(close_arr)):
        direction_arr[i] = direction_arr[i-1]
        prev_st = st_arr[i-1]
        
        if not np.isnan(prev_st):
            # Flip conditions
            if direction_arr[i-1] == 1:
                if close_arr[i] < prev_st:
                    direction_arr[i] = -1
            else:
                if close_arr[i] > prev_st:
                    direction_arr[i] = 1
            
            # Update supertrend
            if direction_arr[i] == 1:
                st_arr[i] = max(lower_band_arr[i], prev_st)
            else:
                st_arr[i] = min(upper_band_arr[i], prev_st)
        else:
            # Handle case where previous ST is NaN
            if direction_arr[i] == 1:
                st_arr[i] = lower_band_arr[i]
            else:
                st_arr[i] = upper_band_arr[i]
    
    return pd.Series(st_arr, index=close.index), pd.Series(direction_arr, index=close.index)

def _adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    high = high.bfill().ffill()
    low = low.bfill().ffill()
    close = close.bfill().ffill()

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = _atr(high, low, close, period)
    
    # Avoid division by zero
    tr_nonzero = tr.replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=period).mean() / tr_nonzero
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=period).mean() / tr_nonzero
    
    # Handle cases where both DIs are zero
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    return adx

def _hurst(close_series, length=64):
    """
    Calculates the Hurst Exponent exactly like QuantNomad's TradingView indicator.
    
    Parameters:
    close_series (pd.Series): Close prices
    length (int): Lookback period for calculation
    
    Returns:
    float: Hurst exponent value
    """
    try:
        if len(close_series) < length + 1:
            return 0.5  # Neutral value if not enough data
            
        # Initialize arrays to store Hurst values
        hurst_values = np.full(len(close_series), np.nan)
        
        # Calculate for each point in the series
        for i in range(length, len(close_series)):
            # Get the window of close prices
            window = close_series.iloc[i-length:i+1]
            
            # Calculate P&L (percentage change)
            pnl = window / window.shift(1) - 1
            pnl = pnl.dropna()  # Remove the first NaN
            
            # Calculate mean P&L
            mean_pnl = pnl.mean()
            
            # Initialize variables for cumulative calculation
            cum = 0.0
            cum_min = float('inf')
            cum_max = float('-inf')
            
            # Calculate cumulative deviation and find min/max
            for pnl_val in pnl:
                cum += pnl_val - mean_pnl
                cum_min = min(cum_min, cum)
                cum_max = max(cum_max, cum)
            
            # Calculate standard deviation
            dev_sum = 0.0
            for pnl_val in pnl:
                dev = pnl_val - mean_pnl
                dev_sum += dev * dev
                
            sd = math.sqrt(dev_sum / (length - 1)) if length > 1 else 0
            
            # Calculate R/S
            if sd == 0:
                hurst_values[i] = 0.5  # Avoid division by zero
                continue
                
            rs = (cum_max - cum_min) / sd
            
            # Calculate Hurst Exponent
            if rs <= 0:
                hurst_values[i] = 0.5  # Avoid log of non-positive number
                continue
                
            hurst_values[i] = math.log(rs) / math.log(length)
        
        return hurst_values
        
    except Exception as e:
        logger.debug(f"Hurst calculation error: {e}")
        # Return array of neutral values on error
        return np.full(len(close_series), 0.5)

# --- Main Strategy Functions ---

def calculate_indicators(df):
    """
    Calculates technical indicators on the DataFrame.
    Modifies the DataFrame in place by adding new columns.
    Returns the modified DataFrame or None if insufficient data.
    """
    # --- Corrected Calculation of Required Length ---
    # Calculate the maximum period needed for any single indicator
    max_single_period = max(
        MA_SLOW_PERIOD,      # Longest simple MA
        RSI_STOCH_PERIOD,    # RSI period (StochRSI uses this)
        max(STOCH_K_PERIOD, STOCH_D_PERIOD), # Max of Stoch K/D smoothing periods
        ATR_PERIOD,          # ATR period
        ADX_PERIOD,          # ADX period (also uses ATR internally)
        13,                  # SMA(13) used with Supertrend filter
        11                   # Supertrend ATR default
    )
    # StochRSI also needs data for its RSI calculation and then its own Stoch calculation
    # So, the total lookback for StochRSI is roughly RSI_STOCH_PERIOD + max(STOCH_K_PERIOD, STOCH_D_PERIOD) - 1
    # To be safe, we take the max of the longest single indicator period and the StochRSI total lookback
    stochrsi_lookback = RSI_STOCH_PERIOD + max(STOCH_K_PERIOD, STOCH_D_PERIOD) - 1
    required_len = max(max_single_period, stochrsi_lookback) + 5 # Add a small buffer

    if len(df) < required_len:
        logger.warning(f"Insufficient data for indicators in strategy.py: "
                       f"len(df)={len(df)}, required>={required_len} (calculated)")
        # Even if technically possible, let's demand a bit more data for robustness
        # Our original data fetch was 100 candles, so this should usually be fine.
        # If issues persist, consider increasing limit in data_fetcher or decreasing required_len logic.
        if len(df) < 21: # Absolute minimum if other checks fail
             logger.error(f"Critical lack of data, less than 21 candles.")
             return None
        # Potentially proceed with a warning if len(df) >= 21 but < required_len?
        # Depends on indicator robustness with less data. For now, be strict.
        return None

    try:
        df["ema_fast"] = _ema(df["close"], MA_FAST_PERIOD)
        df["ema_slow"] = _ema(df["close"], MA_SLOW_PERIOD)
        df["stochrsi_k"], df["stochrsi_d"] = _stochrsi(df["close"], rsi_length=RSI_STOCH_PERIOD, stoch_length=RSI_STOCH_PERIOD, k=STOCH_K_PERIOD, d=STOCH_D_PERIOD)
        df["atr"] = _atr(df["high"], df["low"], df["close"], ATR_PERIOD)
        df["adx"] = _adx(df["high"], df["low"], df["close"], ADX_PERIOD)
        df["hurst"] = _hurst(df["close"], length=51) # Adjust max_lag if needed
        df["volume_ema"] = _ema(df["volume"], 5) # Using fixed 5 for volume EMA as in originally intended

        # --- Supertrend suite (from market/supertrend.py logic) ---
        df["sma13"] = df["close"].rolling(window=13, min_periods=13).mean()
        st, st_dir = _supertrend(df["high"], df["low"], df["close"], factor=int(2), atr_period=15)
        df["supertrend"] = st
        df["st_direction"] = st_dir
        df["bull_signal"] = (df["close"] > df["supertrend"])
        df["bear_signal"] = (df["close"] < df["supertrend"])

        logger.info(f"Indicators calculated for latest row: "
            f"EMA_Fast={df['ema_fast'].iloc[-1]:.6f}, EMA_Slow={df['ema_slow'].iloc[-1]:.6f}, "
            f"StochRSI_K={df['stochrsi_k'].iloc[-1]:.2f}, StochRSI_D={df['stochrsi_d'].iloc[-1]:.2f}, "
            f"ATR={df['atr'].iloc[-1]:.6f}, ADX={df['adx'].iloc[-1]:.2f}, Hurst={df['hurst'].iloc[-1]:.4f}, "
            f"ST={df['supertrend'].iloc[-1]:.6f}, SMA13={df['sma13'].iloc[-1] if not pd.isna(df['sma13'].iloc[-1]) else float('nan'):.6f}")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def generate_signal(df):
    """
    Generates a trading signal based on the latest data in the DataFrame.
    Assumes indicators have been calculated.
    Returns 'long', 'short', or None.
    """
    # (The generate_signal function remains the same as provided previously)
    if df is None or len(df) < 2:
        logger.warning("Insufficient data or DataFrame is None for signal generation.")
        return None

    latest_required = ["supertrend", "sma13"]
    if any(col not in df.columns for col in latest_required):
        logger.warning("Required Supertrend columns missing; ensure calculate_indicators() was called.")
        return None
    latest_indicators = df.iloc[-1][["supertrend", "sma13"]]

    if latest_indicators.isna().any():
        logger.warning("NaN detected in latest indicator values; skipping signal generation.")
        logger.debug(f"Latest indicators with potential NaNs: {latest_indicators.to_dict()}")
        return None

    try:
        latest = df.iloc[-1]

        ma_long = latest["ema_fast"] > latest["ema_slow"]
        ma_short = latest["ema_fast"] < latest["ema_slow"]
        hurst_trend = latest["hurst"] > 0.515
        hurst_revert = latest["hurst"] < 0.485
        # Supertrend crossover with SMA(13) filter (as in market/supertrend.py)
        bull_cond = bool(latest.get('bull_signal', False)) and (latest["close"] >= latest["sma13"] if not pd.isna(latest["sma13"]) else False)
        bear_cond = bool(latest.get('bear_signal', False)) and (latest["close"] <= latest["sma13"] if not pd.isna(latest["sma13"]) else False)

        signal = None
        if ma_long :
            signal = "long"
        elif ma_short :
            signal = "short"

        logger.info(
            f"ST Signal check for {df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else 'N/A'}: "
            f"Bull={bull_cond}, Bear={bear_cond}, Close={latest['close']:.6f}, "
            f"ST={latest['supertrend'] if not pd.isna(latest['supertrend']) else float('nan'):.6f}, "
            f"H_Trend={hurst_trend}, H_Revert={hurst_revert}, "
            f"SMA13={latest['sma13'] if not pd.isna(latest['sma13']) else float('nan'):.6f} -> Signal={signal}"
        )
        return signal
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        logger.debug(f"DataFrame tail for debugging:\n{df.tail()}")
        return None