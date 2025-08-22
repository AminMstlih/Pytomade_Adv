# backtest/engine.py
"""
Backtesting engine for the Pytomade Bot strategy.
Simulates trades on historical data with improved realism:
- Timeline correctness: signal on bar i-1, enter at bar i open
- Deterministic intrabar execution order (OHLC or OLHC)
- Partial TP1/TP2 handling with optional SL-to-breakeven
- Notional-based PnL and fee model
- Mark-to-market equity (unrealized PnL)
"""

import pandas as pd
import numpy as np
from market import strategy
from bot import config
from bot.logger_config import logger # Use the same logger

# --- Backtest behavior flags (kept local to avoid changing config.py) ---
# Intrabar path assumption when both TP/SL are hit within the same candle.
# Options: "OHLC" (open->high->low->close) or "OLHC" (open->low->high->close)
BAR_ORDER = "OLHC"  # Conservative for longs (SL can hit before TP); switch to "OHLC" to test sensitivity

# Apply partial profit taking: TP1 closes TP1_SIZE_RATIO, TP2 closes remainder
ENABLE_PARTIALS = True

# After TP1 fill, move stop to breakeven (entry price) on remaining size
MOVE_SL_TO_BREAKEVEN_AFTER_TP1 = True

# Fee model: use taker fee on both entry and exit based on traded notional
# Use the fee_rate argument as taker fee rate per side

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
    # "capital" tracks cash after realized PnL and fees (margin not deducted)
    capital = initial_capital
    # Support at most one position per side (long/short) to match current strategy
    open_positions = {}  # key: 'long'|'short' -> position dict
    closed_trades = []   # list of dicts
    equity_curve = []    # mark-to-market equity each bar
    peak_equity = initial_capital
    max_drawdown = 0.0

    # Add columns for signals and bookkeeping
    df['signal'] = None

    # --- Helper functions ---
    def _pnl_for(side, entry_price, exit_price, notional):
        if side == 'long':
            return notional * (exit_price / entry_price - 1)
        else:
            return notional * (1 - exit_price / entry_price)

    def _apply_bar_order_for_long(high, low, tp_hit, sl_hit, tp_price, sl_price):
        # Returns list of fills [(type, price)] in order for the candle
        events = []
        if BAR_ORDER == "OHLC":
            if tp_hit and sl_hit:
                # High first then Low
                if high >= tp_price:
                    events.append(("TP", tp_price))
                if low <= sl_price:
                    events.append(("SL", sl_price))
            elif tp_hit:
                events.append(("TP", tp_price))
            elif sl_hit:
                events.append(("SL", sl_price))
        else:  # OLHC
            if tp_hit and sl_hit:
                # Low first then High (pessimistic for longs)
                if low <= sl_price:
                    events.append(("SL", sl_price))
                if high >= tp_price:
                    events.append(("TP", tp_price))
            elif sl_hit:
                events.append(("SL", sl_price))
            elif tp_hit:
                events.append(("TP", tp_price))
        return events

    def _apply_bar_order_for_short(high, low, tp_hit, sl_hit, tp_price, sl_price):
        events = []
        if BAR_ORDER == "OHLC":
            if tp_hit and sl_hit:
                # High then Low (pessimistic for shorts; SL above)
                if high >= sl_price:
                    events.append(("SL", sl_price))
                if low <= tp_price:
                    events.append(("TP", tp_price))
            elif sl_hit:
                events.append(("SL", sl_price))
            elif tp_hit:
                events.append(("TP", tp_price))
        else:  # OLHC
            if tp_hit and sl_hit:
                # Low then High (optimistic for shorts)
                if low <= tp_price:
                    events.append(("TP", tp_price))
                if high >= sl_price:
                    events.append(("SL", sl_price))
            elif tp_hit:
                events.append(("TP", tp_price))
            elif sl_hit:
                events.append(("SL", sl_price))
        return events

    # --- Main Backtesting Loop ---
    # Iterate from the point where enough data exists for indicators
    # We'll generate signal on bar i-1 close and enter at bar i open
    for i in range(required_buffer, len(df)):
        idx = df.index[i]
        ts = df.loc[idx, 'timestamp']
        candle = df.loc[idx]

        # Prepare indicators up to previous bar for signal
        exit_happened_this_bar = False
        if i - 1 >= 0:
            sig_df = df.loc[:df.index[i - 1]].copy()
        else:
            sig_df = None

        data_with_ind = strategy.calculate_indicators(sig_df) if sig_df is not None else None
        sig = strategy.generate_signal(data_with_ind) if (data_with_ind is not None and not data_with_ind.empty) else None
        df.loc[idx, 'signal'] = sig

        # 1) Check exits for existing positions using current candle only if the
        #    position was opened on a prior bar (position['entry_index'] < idx)
        if open_positions:
            for side_key in list(open_positions.keys()):
                pos = open_positions[side_key]
                if pos['status'] != 'open':
                    continue
                if pos['entry_index'] == idx:
                    # Prevent same-bar exit at the entry bar
                    continue

                # Determine if TP/SL levels are touched in this candle
                high = candle['high']
                low = candle['low']

                if pos['side'] == 'long':
                    tp_hit = (high >= pos['tp1_price']) or (high >= pos['tp2_price'])
                    sl_hit = low <= pos['sl_price']
                    events = _apply_bar_order_for_long(high, low, tp_hit, sl_hit,
                                                       min(pos['tp1_price'], pos['tp2_price']), pos['sl_price'])
                else:
                    tp_hit = (low <= pos['tp1_price']) or (low <= pos['tp2_price'])
                    sl_hit = high >= pos['sl_price']
                    events = _apply_bar_order_for_short(high, low, tp_hit, sl_hit,
                                                        max(pos['tp1_price'], pos['tp2_price']), pos['sl_price'])

                # Apply event sequence, supporting partials
                for ev_type, ev_price in events:
                    if pos['remaining_usd'] <= 0:
                        break
                    if ev_type == 'SL':
                        # Close all remaining size at SL
                        close_notional = pos['remaining_usd']
                        pnl = _pnl_for(pos['side'], pos['entry_price'], ev_price, close_notional)
                        exit_fee = close_notional * fee_rate
                        capital += pnl - exit_fee

                        closed_trades.append({
                            'entry_time': pos['entry_timestamp'],
                            'exit_time': ts,
                            'side': pos['side'],
                            'entry_price': pos['entry_price'],
                            'exit_price': ev_price,
                            'type': 'SL',
                            'pnl': pnl,
                            'fee': exit_fee,
                            'capital_after': capital
                        })
                        pos['remaining_usd'] = 0
                        pos['status'] = 'closed'
                        exit_happened_this_bar = True
                        break
                    elif ev_type == 'TP':
                        if ENABLE_PARTIALS:
                            # Decide which TP is hit first based on price distance
                            # Use TP1 first, then TP2 for remainder
                            if not pos.get('tp1_filled', False) and (
                                (pos['side'] == 'long' and ev_price >= pos['tp1_price']) or
                                (pos['side'] == 'short' and ev_price <= pos['tp1_price'])
                            ):
                                close_notional = min(pos['tp1_size_usd'], pos['remaining_usd'])
                                pnl = _pnl_for(pos['side'], pos['entry_price'], pos['tp1_price'], close_notional)
                                exit_fee = close_notional * fee_rate
                                capital += pnl - exit_fee
                                pos['remaining_usd'] -= close_notional
                                pos['tp1_filled'] = True
                                closed_trades.append({
                                    'entry_time': pos['entry_timestamp'],
                                    'exit_time': ts,
                                    'side': pos['side'],
                                    'entry_price': pos['entry_price'],
                                    'exit_price': pos['tp1_price'],
                                    'type': 'TP1',
                                    'pnl': pnl,
                                    'fee': exit_fee,
                                    'capital_after': capital
                                })
                                if MOVE_SL_TO_BREAKEVEN_AFTER_TP1 and pos['remaining_usd'] > 0:
                                    pos['sl_price'] = pos['entry_price']
                            elif pos.get('tp1_filled', False) and (
                                (pos['side'] == 'long' and ev_price >= pos['tp2_price']) or
                                (pos['side'] == 'short' and ev_price <= pos['tp2_price'])
                            ):
                                close_notional = pos['remaining_usd']
                                pnl = _pnl_for(pos['side'], pos['entry_price'], pos['tp2_price'], close_notional)
                                exit_fee = close_notional * fee_rate
                                capital += pnl - exit_fee
                                pos['remaining_usd'] = 0
                                pos['status'] = 'closed'
                                closed_trades.append({
                                    'entry_time': pos['entry_timestamp'],
                                    'exit_time': ts,
                                    'side': pos['side'],
                                    'entry_price': pos['entry_price'],
                                    'exit_price': pos['tp2_price'],
                                    'type': 'TP2',
                                    'pnl': pnl,
                                    'fee': exit_fee,
                                    'capital_after': capital
                                })
                                exit_happened_this_bar = True
                                break
                        else:
                            # Close full on first TP if partials disabled
                            close_notional = pos['remaining_usd']
                            pnl = _pnl_for(pos['side'], pos['entry_price'], ev_price, close_notional)
                            exit_fee = close_notional * fee_rate
                            capital += pnl - exit_fee
                            closed_trades.append({
                                'entry_time': pos['entry_timestamp'],
                                'exit_time': ts,
                                'side': pos['side'],
                                'entry_price': pos['entry_price'],
                                'exit_price': ev_price,
                                'type': 'TP',
                                'pnl': pnl,
                                'fee': exit_fee,
                                'capital_after': capital
                            })
                            pos['remaining_usd'] = 0
                            pos['status'] = 'closed'
                            exit_happened_this_bar = True
                            break

                # Clean up closed positions
                if pos['status'] == 'closed':
                    del open_positions[side_key]

        # 2) Process new entries at current bar open if signal was generated on prior bar
        if sig in ['long', 'short'] and sig not in open_positions and i < len(df):
            entry_price = candle['open']

            # Position sizing: notional = margin * leverage, capped
            notional = min(config.TARGET_MARGIN_USD * config.LEVERAGE, config.MAX_POSITION_SIZE_USDT)

            # Compute TP/SL prices
            lev_factor = config.LEVERAGE * 100.0
            if sig == 'long':
                tp1_price = round(entry_price * (1 + config.TP1_PNL_PERCENT / lev_factor), 6)
                tp2_price = round(entry_price * (1 + config.TP2_PNL_PERCENT / lev_factor), 6)
                sl_price = round(entry_price * (1 + config.SL_PNL_PERCENT / lev_factor), 6)
            else:
                tp1_price = round(entry_price * (1 - config.TP1_PNL_PERCENT / lev_factor), 6)
                tp2_price = round(entry_price * (1 - config.TP2_PNL_PERCENT / lev_factor), 6)
                sl_price = round(entry_price * (1 - config.SL_PNL_PERCENT / lev_factor), 6)

            tp1_notional = round(config.TP1_SIZE_RATIO * notional, 2)
            tp2_notional = round(notional - tp1_notional, 2)

            # Entry fee on traded notional
            entry_fee = notional * fee_rate
            capital -= entry_fee

            open_positions[sig] = {
                'side': sig,
                'entry_price': entry_price,
                'entry_timestamp': ts,
                'entry_index': idx,
                'tp1_price': tp1_price,
                'tp2_price': tp2_price,
                'sl_price': sl_price,
                'tp1_size_usd': tp1_notional,
                'tp2_size_usd': tp2_notional,
                'remaining_usd': notional,
                'tp1_filled': False,
                'status': 'open'
            }
            logger.debug(f"[{ts}] Entered {sig} @ {entry_price:.6f}, notional=${notional:.2f}, fee=${entry_fee:.4f}")

        # 3) Mark-to-market equity update using close price of current bar
        unrealized = 0.0
        for pos in open_positions.values():
            if pos['status'] != 'open' or pos['remaining_usd'] <= 0:
                continue
            close_price = candle['close']
            unrealized += _pnl_for(pos['side'], pos['entry_price'], close_price, pos['remaining_usd'])

        equity = capital + unrealized
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

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