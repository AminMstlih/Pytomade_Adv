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
    """Calculates the Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def _rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = np.where(loss == 0, 0, gain / loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)
    return pd.Series(rsi, index=series.index)

def _stochrsi(series, rsi_length=14, stoch_length=14, k=3, d=3):
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
        ADX_PERIOD           # ADX period (also uses ATR internally)
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
        logger.debug(f"Indicators calculated for latest row: "
            f"EMA_Fast={df['ema_fast'].iloc[-1]:.6f}, EMA_Slow={df['ema_slow'].iloc[-1]:.6f}, "
            f"StochRSI_K={df['stochrsi_k'].iloc[-1]:.2f}, StochRSI_D={df['stochrsi_d'].iloc[-1]:.2f}, "
            f"ATR={df['atr'].iloc[-1]:.6f}, ADX={df['adx'].iloc[-1]:.2f}, Hurst={df['hurst'].iloc[-1]:.4f}") # <-- Added Hurst

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

    latest_indicators = df.iloc[-1][["ema_fast", "ema_slow", "stochrsi_k", "stochrsi_d", "adx", "volume_ema", "hurst"]] # <-- Added hurst
    if latest_indicators.isna().any():
        logger.warning("NaN detected in latest indicator values; skipping signal generation.")
        logger.debug(f"Latest indicators with potential NaNs: {latest_indicators.to_dict()}")
        return None

    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None

        ma_long = latest["ema_fast"] > latest["ema_slow"]
        ma_short = latest["ema_fast"] < latest["ema_slow"]
        stochrsi_long = latest["stochrsi_k"] > latest["stochrsi_d"]
        stochrsi_short = latest["stochrsi_k"] < latest["stochrsi_d"]
        volume_trend = latest['volume'] > VOLUME_THRESHOLD_MULTIPLIER * latest['volume_ema']
        adx_strong = latest["adx"] > ADX_THRESHOLD
        stoch_not_overbought = latest["stochrsi_k"] < 85
        stoch_not_oversold = latest["stochrsi_k"] > 15
        hurst_trending = latest["hurst"] > 0.5
        hurst_mean_reverting = latest["hurst"] < 0.4

        signal = None
        if (ma_long and stochrsi_long and volume_trend and stoch_not_overbought and adx_strong and hurst_trending):
            signal = "long"
        elif (ma_short and stochrsi_short and volume_trend and stoch_not_oversold and adx_strong and hurst_mean_reverting):
            signal = "short"

        logger.info(f"Signal check for {df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else 'N/A'}: "
            f"EMA_L={ma_long}, EMA_S={ma_short}, SRSI_L={stochrsi_long}, SRSI_S={stochrsi_short}, "
            f"Vol_OK={volume_trend}, ADX_OK={adx_strong}, SRSI_NB={stoch_not_overbought}, SRSI_NS={stoch_not_oversold}, "
            f"H_Trend={hurst_trending}, H_Revert={hurst_mean_reverting} -> Signal={signal}") # <-- Added H_Trend
        return signal
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        logger.debug(f"DataFrame tail for debugging:\n{df.tail()}")
        return None