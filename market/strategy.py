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
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def _rsi(series, period=14):  # Updated default to 14 for standard alignment
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Fill NaNs with 50 (neutral) for early periods
    return pd.Series(rsi, index=series.index)

def _stochrsi(series, rsi_length=5, stoch_length=5, k=3, d=3):
    rsi_val = _rsi(series, rsi_length)
    lowest_rsi = rsi_val.rolling(window=stoch_length, min_periods=stoch_length).min()
    highest_rsi = rsi_val.rolling(window=stoch_length, min_periods=stoch_length).max()
    stoch_k = np.where(highest_rsi == lowest_rsi, 50, 100 * (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi))
    stoch_d = pd.Series(stoch_k, index=series.index).rolling(window=d, min_periods=d).mean()
    stoch_d = stoch_d.bfill().ffill()
    return pd.Series(stoch_k, index=series.index), stoch_d

def _atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()

def _rma(series, period):
    """Wilder's RMA used by many Pine indicators (including ATR)."""
    return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def _atr_rma(high, low, close, period=14):
    """ATR using Wilder's RMA to better match PineScript behavior."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _rma(tr, period)

def _supertrend(high, low, close, factor=5.5, atr_period=11):
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

    upper_band = high + factor * atr
    lower_band = low - factor * atr

    # Determine integer location of first valid atr
    try:
        start_i = int(np.atleast_1d(atr.index.searchsorted(first_valid))[0])
    except Exception:
        start_i = 0

    for i in range(start_i, len(close)):
        if i == 0:
            direction.iloc[i] = 1 if close.iloc[i] > upper_band.iloc[i] else -1
            st.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
            continue

        # carry previous
        direction.iloc[i] = direction.iloc[i-1]
        prev_st = st.iloc[i-1]

        # flip conditions
        if direction.iloc[i-1] == 1:
            if not pd.isna(prev_st) and close.iloc[i] < prev_st:
                direction.iloc[i] = -1
        else:
            if not pd.isna(prev_st) and close.iloc[i] > prev_st:
                direction.iloc[i] = 1

        # update supertrend
        if direction.iloc[i] == 1:
            st.iloc[i] = max(lower_band.iloc[i], prev_st) if not pd.isna(prev_st) else lower_band.iloc[i]
        else:
            st.iloc[i] = min(upper_band.iloc[i], prev_st) if not pd.isna(prev_st) else upper_band.iloc[i]

    return st, direction

def _adx(high, low, close, period=14):
    high = high.bfill().ffill()
    low = low.bfill().ffill()
    close = close.bfill().ffill()

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = _atr(high, low, close, period)
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=period).mean() / tr.where(tr != 0, np.nan)
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=period).mean() / tr.where(tr != 0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).where((plus_di + minus_di) != 0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx

def _hurst(series, max_lag=20):
    """
    Calculates the Hurst exponent using the Rescaled Range (R/S) method.
    ... (docstring) ...
    """
    try:
        ts = series.dropna()
        if len(ts) < max_lag * 2:
            return np.nan
            
        lags = range(2, min(max_lag, len(ts) // 2))
        # Calculate tau: standard deviation of price differences
        tau = [np.sqrt(np.std(ts.diff(lag).dropna())) for lag in lags]
        # --- Fix 1: Safer filtering of tau ---
        # Filter out non-positive values and non-numeric/NaN values *safely*
        filtered_tau_lags = [
            (tau[i], lags[i]) for i in range(len(tau))
            if isinstance(tau[i], (int, float)) and # Check if tau[i] is a number
               not math.isnan(tau[i]) and          # Check if it's not NaN
               tau[i] > 0                           # Check if it's positive
        ]
        if not filtered_tau_lags or len(filtered_tau_lags) < 3:
            return np.nan # Need at least 3 points for regression

        # Unpack the filtered lists
        tau_filtered, lags_filtered = zip(*filtered_tau_lags) # This might be the line Pylance dislikes
        # Convert back to lists if needed, though linregress can often handle tuples/arrays
        # tau_filtered = list(tau_filtered)
        # lags_filtered = list(lags_filtered)

        # --- Fix 2: Ensure inputs to linregress are clean ---
        # Log-log plot: log(tau) vs log(lag)
        # Add small epsilon to avoid log(0) if somehow values are zero (shouldn't be with > 0 check)
        epsilon = 1e-10
        log_lags = np.log(np.array(lags_filtered) + epsilon)
        log_tau = np.log(np.array(tau_filtered) + epsilon)

        # --- Fix 3: Check for valid data before regression ---
        if len(log_lags) < 3 or np.any(~np.isfinite(log_lags)) or np.any(~np.isfinite(log_tau)):
             return np.nan # Invalid data for regression

        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_tau)
        
        # --- Fix 4: Safer check for hurst_exp ---
        # Hurst exponent is the slope of the log-log plot
        hurst_exp = slope
        # Check if hurst_exp is a valid number before returning
        if isinstance(hurst_exp, (int, float)) and not math.isnan(hurst_exp):
            return hurst_exp
        else:
            return np.nan # Return NaN if slope is somehow invalid
    except Exception as e:
        # print(f"Hurst calculation error: {e}") # Optional debug print
        return np.nan # Return NaN on any error

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
        df["hurst"] = _hurst(df["close"], max_lag=20) # Adjust max_lag if needed
        df["volume_ema"] = _ema(df["volume"], 5) # Using fixed 5 for volume EMA as in original

        # --- Supertrend suite (from market/supertrend.py logic) ---
        df["sma13"] = df["close"].rolling(window=13, min_periods=13).mean()
        st, st_dir = _supertrend(df["high"], df["low"], df["close"], factor=5.5, atr_period=11)
        df["supertrend"] = st
        df["st_direction"] = st_dir
        df["bull_signal"] = (df["close"] > df["supertrend"]) & (df["close"].shift(1) <= df["supertrend"].shift(1))
        df["bear_signal"] = (df["close"] < df["supertrend"]) & (df["close"].shift(1) >= df["supertrend"].shift(1))

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
        # Supertrend crossover with SMA(13) filter (as in market/supertrend.py)
        bull_cond = bool(latest.get('bull_signal', False)) and (latest["close"] >= latest["sma13"] if not pd.isna(latest["sma13"]) else False)
        bear_cond = bool(latest.get('bear_signal', False)) and (latest["close"] <= latest["sma13"] if not pd.isna(latest["sma13"]) else False)

        signal = "long" if bull_cond else "short" if bear_cond else None

        logger.info(
            f"ST Signal check for {df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else 'N/A'}: "
            f"Bull={bull_cond}, Bear={bear_cond}, Close={latest['close']:.6f}, "
            f"ST={latest['supertrend'] if not pd.isna(latest['supertrend']) else float('nan'):.6f}, "
            f"SMA13={latest['sma13'] if not pd.isna(latest['sma13']) else float('nan'):.6f} -> Signal={signal}"
        )
        return signal
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        logger.debug(f"DataFrame tail for debugging:\n{df.tail()}")
        return None