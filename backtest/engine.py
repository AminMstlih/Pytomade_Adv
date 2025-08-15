# backtest/engine.py
"""
Backtesting engine for the Pytomade Bot strategy.
Simulates trades on historical data.
"""

import pandas as pd
import numpy as np
from market import strategy
from bot import config
from bot.logger_config import logger # Use the same logger

def run_backtest(historical_df, initial_capital=100.0, fee_rate=0.001):
    """
    Runs a backtest on historical price data.

    Args:
        historical_df (pd.DataFrame): DataFrame with 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
        initial_capital (float): Starting virtual capital in USDT.
        fee_rate (float): Estimated fee rate per trade side (e.g., 0.001 for 0.1%).

    Returns:
        tuple: (modified_df_with_signals, list_of_closed_trades, final_metrics_dict)
    """
    logger.info("Starting backtest...")
    if historical_df is None or historical_df.empty:
        logger.error("Cannot run backtest: Empty or None historical data provided.")
        return None, [], {}

    # Ensure data is sorted by timestamp
    df = historical_df.sort_values('timestamp').reset_index(drop=True).copy()
    required_buffer = max(
        config.MA_SLOW_PERIOD,
        config.RSI_STOCH_PERIOD + max(config.STOCH_K_PERIOD, config.STOCH_D_PERIOD) - 1,
        config.ATR_PERIOD,
        config.ADX_PERIOD
    ) + 10 # Add a larger buffer for safety

    if len(df) <= required_buffer:
        logger.error(f"Insufficient data for backtest. Need more than {required_buffer} candles.")
        return df, [], {}

    # --- Tracking Variables ---
    capital = initial_capital
    # Simplified: Track one open position at a time for now, matching live bot's potential hedging logic later
    open_positions = {} # key: 'long'/'short', value: position dict
    closed_trades = [] # List of dictionaries for closed trades
    equity_curve = [initial_capital] # Track capital over time
    peak_capital = initial_capital
    max_drawdown = 0.0

    # Add a column to store signals in the main dataframe for analysis
    df['signal'] = None

    # --- Main Backtesting Loop ---
    # Iterate from the point where enough data exists for indicators
    for i in range(required_buffer, len(df)):
        current_index = df.index[i]
        current_timestamp = df.loc[current_index, 'timestamp']
        current_candle = df.loc[current_index] # Series for the current candle

        # --- Initialize variables for this loop iteration ---
        exit_happened = False # Initialize here to ensure it's always defined
        # --- End of initialization ---

        # 1. Analyze Data up to current point for signal generation
        # Use data including the current candle for decision (decision made at close)
        data_for_analysis = df.loc[:current_index].copy()

        # 2. Calculate Indicators
        data_with_indicators = strategy.calculate_indicators(data_for_analysis)
        if data_with_indicators is None or data_with_indicators.empty:
            logger.debug(f"Skipping index {i} due to indicator calculation failure or insufficient data.")
            equity_curve.append(capital) # Capital unchanged
            continue

        # 3. Generate Signal using the latest row (current candle)
        # Pass the df *with* indicators
        sig = strategy.generate_signal(data_with_indicators)
        df.loc[current_index, 'signal'] = sig # Store signal in main df

        ## --- Simulate Trade Execution (at the next candle's open) ---
        # Check for entry signal and if we don't already have a position on that side
        next_index = df.index[i + 1] if (i + 1) < len(df) else None
        if sig in ['long', 'short'] and next_index is not None and sig not in open_positions:
             next_candle = df.loc[next_index]
             entry_price = next_candle['open'] # Enter at next candle's open

             # --- Simplified Position Sizing (Matching risk manager logic conceptually) ---
             # Use config values to determine size. Backtest in USD value terms is common.
             # Size_USD = Margin_USD * Leverage (Conceptually, for PnL calculation)
             position_size_usd = config.TARGET_MARGIN_USD * config.LEVERAGE
             position_size_usd = min(position_size_usd, config.MAX_POSITION_SIZE_USDT) # Apply cap

             # Calculate TP/SL prices (same logic as risk manager)
             leverage_factor = config.LEVERAGE * 100.0
             if sig == "long":
                 tp1_distance_ratio = config.TP1_PNL_PERCENT / leverage_factor
                 tp2_distance_ratio = config.TP2_PNL_PERCENT / leverage_factor
                 sl_distance_ratio = config.SL_PNL_PERCENT / leverage_factor
                 tp1_price = entry_price * (1 + tp1_distance_ratio)
                 tp2_price = entry_price * (1 + tp2_distance_ratio)
                 sl_price = entry_price * (1 + sl_distance_ratio)
             else: # sig == "short"
                 tp1_distance_ratio = config.TP1_PNL_PERCENT / leverage_factor
                 tp2_distance_ratio = config.TP2_PNL_PERCENT / leverage_factor
                 sl_distance_ratio = config.SL_PNL_PERCENT / leverage_factor
                 tp1_price = entry_price * (1 - tp1_distance_ratio)
                 tp2_price = entry_price * (1 - tp2_distance_ratio)
                 sl_price = entry_price * (1 - sl_distance_ratio)

             tp1_price = round(tp1_price, 6) # Round appropriately
             tp2_price = round(tp2_price, 6)
             sl_price = round(sl_price, 6)

             # Calculate TP sizes in USD value (for PnL calculation)
             tp1_size_usd = round(config.TP1_SIZE_RATIO * position_size_usd, 2)
             tp2_size_usd = round((1 - config.TP1_SIZE_RATIO) * position_size_usd, 2)
             sl_size_usd = round(position_size_usd, 2) # SL covers full intended size conceptually

             # Deduct margin cost from capital (simulate margin usage)
             margin_cost = position_size_usd / config.LEVERAGE
             capital -= margin_cost
             if capital < 0:
                 logger.warning(f"Capital depleted at {current_timestamp}. Stopping backtest.")
                 break # Stop if capital runs out

             # Create and store the open position
             open_positions[sig] = {
                 'side': sig,
                 'entry_price': entry_price,
                 'size_usd': position_size_usd, # Total intended size (for reference)
                 'margin_usd': margin_cost,
                 'tp1_price': tp1_price,
                 'tp2_price': tp2_price,
                 'sl_price': sl_price,
                 'tp1_size_usd': tp1_size_usd,
                 'tp2_size_usd': tp2_size_usd,
                 'sl_size_usd': sl_size_usd, # For PnL calculation if needed
                 'entry_timestamp': next_candle['timestamp'], # Timestamp of entry candle
                 'entry_index': next_index, # Index in df for entry
                 'status': 'open'
             }
             logger.debug(f"[{next_candle['timestamp']}] Entered {sig} @ {entry_price:.6f}, "
                          f"Capital: ${capital:.4f}, Margin Used: ${margin_cost:.4f}")

        # --- Check Open Positions for Exits (based on current candle's H/L) ---
        # Iterate over a copy of keys to allow modification of open_positions dict
        # Important: Only check exits if there are open positions
        if open_positions: # Add this check
            for pos_side in list(open_positions.keys()):
                position = open_positions[pos_side]
                if position['status'] != 'open':
                    continue

            # Exit flags
            exit_happened = False
            exit_price = None
            exit_type = None
            pnl = 0
            fee = 0

            # Determine price range for exit checks
            # Use current candle (i) and potentially next candle (i+1) if it exists
            # This check happens after the entry decision for candle i
            check_low = current_candle['low']
            check_high = current_candle['high']
            # If we entered at the *next* candle's open (i+1), the earliest exit check
            # based on candle i's price is logically inconsistent unless we check
            # the *next* candle's price (i+1) against the entry (also at i+1 open).
            # Let's simplify: Check exits based on the *current* candle (i) against
            # an entry that hypothetically happened *before* or at the start of this evaluation.
            # A more precise way: Entry at i's close, exit check at i+1's high/low.
            # Or, Entry at i's close, exit check using i's high/low (if order fills instantly).
            # Let's assume entry happens quickly after signal (e.g., at close of signal candle i)
            # and exit is checked against the *next* candle (i+1).
            # But our loop is on i. To check exit for an entry at i, we need i+1.
            # To check exit for an entry at i-1, we use current candle i.
            # This is getting complex. Let's simplify:
            # Entry: At close of signal candle.
            # Exit Check: Against high/low of the *same* candle the signal was generated
            # (or the next, depending on interpretation). Let's check against current candle (i)
            # for an entry that happened previously (need to track entry time).

            # Better approach: Iterate through open positions and check against the current price bar.
            # Assume position was entered before or during the period represented by current_candle.

            if position['side'] == 'long':
                # Check for SL (price goes below SL)
                if check_low <= position['sl_price']:
                    exit_price = position['sl_price']
                    exit_type = 'SL'
                    # PnL for long: (Exit / Entry - 1) * Leverage * Position_Size_USD
                    # This calculates PnL for the entire intended position size that was margined
                    # In live trade, attached TPs/SLs handle partials. Here, we simplify.
                    # Let's assume the whole position is closed at SL for now.
                    # More accurate: Track partial closes like live bot.
                    # Simplification for now: Close full position at SL.
                    pnl = (exit_price / position['entry_price'] - 1) * config.LEVERAGE * position['size_usd']
                    # Fee: Charged on entry margin + exit value (approximated by entry size)
                    fee = (position['margin_usd'] + abs(pnl)) * fee_rate
                    exit_happened = True
                # Check for TP1 (price goes above TP1)
                elif check_high >= position['tp1_price']:
                     exit_price = position['tp1_price']
                     exit_type = 'TP1'
                     pnl = (exit_price / position['entry_price'] - 1) * config.LEVERAGE * position['tp1_size_usd']
                     fee = (position['margin_usd'] * config.TP1_SIZE_RATIO + abs(pnl)) * fee_rate
                     # For simplicity, close the TP1 part. In live, the algo order handles this.
                     # We can either close part or full. Let's close part and keep the rest.
                     # This requires tracking partial position. Simpler: Close full position on first TP/SL.
                     # Let's stick to close full position on first touch for simplicity in this version.
                     # TODO: Implement partial closes for more accurate backtest.
                     exit_happened = True # This will close the whole position in this simplified logic
                # Check for TP2 (price goes above TP2)
                elif check_high >= position['tp2_price']:
                     exit_price = position['tp2_price']
                     exit_type = 'TP2'
                     pnl = (exit_price / position['entry_price'] - 1) * config.LEVERAGE * position['tp2_size_usd']
                     fee = (position['margin_usd'] * (1 - config.TP1_SIZE_RATIO) + abs(pnl)) * fee_rate
                     exit_happened = True # This will close the whole position

            elif position['side'] == 'short':
                # Check for SL (price goes above SL)
                if check_high >= position['sl_price']:
                    exit_price = position['sl_price']
                    exit_type = 'SL'
                    pnl = (1 - exit_price / position['entry_price']) * config.LEVERAGE * position['size_usd']
                    fee = (position['margin_usd'] + abs(pnl)) * fee_rate
                    exit_happened = True
                # Check for TP1 (price goes below TP1)
                elif check_low <= position['tp1_price']:
                     exit_price = position['tp1_price']
                     exit_type = 'TP1'
                     pnl = (1 - exit_price / position['entry_price']) * config.LEVERAGE * position['tp1_size_usd']
                     fee = (position['margin_usd'] * config.TP1_SIZE_RATIO + abs(pnl)) * fee_rate
                     exit_happened = True
                # Check for TP2 (price goes below TP2)
                elif check_low <= position['tp2_price']:
                     exit_price = position['tp2_price']
                     exit_type = 'TP2'
                     pnl = (1 - exit_price / position['entry_price']) * config.LEVERAGE * position['tp2_size_usd']
                     fee = (position['margin_usd'] * (1 - config.TP1_SIZE_RATIO) + abs(pnl)) * fee_rate
                     exit_happened = True

            if exit_happened:
                # Add PnL and return margin
                capital = capital + position['margin_usd'] + pnl - fee # Return margin + PnL - Fee
                equity_curve.append(capital)
                # Update peak capital and drawdown
                if capital > peak_capital:
                    peak_capital = capital
                current_drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
                max_drawdown = max(max_drawdown, current_drawdown)

                # Log the closed trade
                closed_trades.append({
                    'entry_time': position['entry_timestamp'],
                    'exit_time': current_timestamp,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'type': exit_type,
                    'pnl': pnl,
                    'fee': fee,
                    'capital_after': capital
                })
                logger.debug(f"[{current_timestamp}] Exited {position['side']} @ {exit_price:.6f} ({exit_type}), "
                             f"PnL: ${pnl:.4f}, Fee: ${fee:.4f}, Capital: ${capital:.4f}")

                # Remove the closed position
                del open_positions[pos_side] # Close the position (simplified full close)

        # --- Update Equity Curve ---
        # If an exit happened, capital was already updated and appended to equity_curve inside the exit logic.
        # If no exit happened, or if there were no open positions to check, append current capital.
        # Because exit_happened is initialized at the start of the loop, this check is now safe.
        if not exit_happened:
             equity_curve.append(capital)

    # --- Finalize ---
    # Close any remaining open positions at the end? (Not typical, but depends on goal)
    # For now, leave them open in the record.

    # --- Calculate Final Metrics ---
    final_capital = capital
    total_pnl = final_capital - initial_capital
    total_return_percent = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0

    # Win Rate
    profitable_trades = [t for t in closed_trades if t['pnl'] > 0]
    win_rate = (len(profitable_trades) / len(closed_trades)) * 100 if closed_trades else 0

    # Total Fees
    total_fees = sum(t['fee'] for t in closed_trades)

    # Sharpe Ratio (Simplified, annualized, assuming 1H candles)
    # Calculate returns for each period (each step in the loop)
    returns = pd.Series(equity_curve).pct_change().dropna()
    if len(returns) > 1 and returns.std() != 0:
        # Assuming hourly data for Sharpe (adjust time period factor as needed)
        # Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Return * SQRT(N)
        # Assume risk-free rate = 0 for crypto
        periods_per_year = 365 * 24 # Hours in a year
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(min(len(returns), periods_per_year)) # Annualized
    else:
        sharpe_ratio = 0

    final_metrics = {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_pnl': total_pnl,
        'total_return_percent': total_return_percent,
        'total_fees': total_fees,
        'number_of_trades': len(closed_trades),
        'win_rate_percent': win_rate,
        'max_drawdown_percent': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': equity_curve
    }

    logger.info("--- Backtest Complete ---")
    logger.info(f"Initial Capital: ${initial_capital:.4f}")
    logger.info(f"Final Capital: ${final_capital:.4f}")
    logger.info(f"Total PnL: ${total_pnl:.4f}")
    logger.info(f"Total Return: {total_return_percent:.2f}%")
    logger.info(f"Total Fees Paid: ${total_fees:.4f}")
    logger.info(f"Number of Closed Trades: {len(closed_trades)}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    return df, closed_trades, final_metrics

# --- Helper function to load data (can be expanded) ---
def load_historical_data_from_csv(filepath):
    """Loads historical data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        # Ensure necessary columns are present and of correct type
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV file {filepath} is missing required columns: {set(required_columns) - set(df.columns)}")
            return None
        # Convert timestamp if needed (assuming milliseconds)
        if 'timestamp' in df.columns:
             if df['timestamp'].dtype == 'object':
                 # Try common formats
                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce') # Try ms first
                 if df['timestamp'].isna().any():
                      df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') # Try ISO format
             else:
                 # Assume it's already numeric (ms) or datetime
                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        # Convert OHLCV to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.isna().any().any():
             logger.warning("NaN values found in loaded data. Dropping rows with NaN.")
             df.dropna(inplace=True)

        logger.info(f"Loaded {len(df)} rows of historical data from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None