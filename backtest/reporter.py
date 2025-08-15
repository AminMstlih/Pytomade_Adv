# backtest/reporter.py
"""
Handles the reporting and analysis of backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bot.logger_config import logger

def generate_summary_report(metrics, trade_list):
    """
    Generates a summary text report of the backtest results.
    """
    report_lines = []
    report_lines.append("=" * 40)
    report_lines.append("BACKTEST SUMMARY REPORT")
    report_lines.append("=" * 40)
    report_lines.append(f"Initial Capital:     ${metrics['initial_capital']:.4f}")
    report_lines.append(f"Final Capital:       ${metrics['final_capital']:.4f}")
    report_lines.append(f"Total PnL:           ${metrics['total_pnl']:.4f}")
    report_lines.append(f"Total Return:        {metrics['total_return_percent']:.2f}%")
    report_lines.append(f"Total Fees Paid:     ${metrics['total_fees']:.4f}")
    report_lines.append("-" * 30)
    report_lines.append(f"Number of Trades:    {metrics['number_of_trades']}")
    report_lines.append(f"Win Rate:            {metrics['win_rate_percent']:.2f}%")
    report_lines.append(f"Max Drawdown:        {metrics['max_drawdown_percent']:.2f}%")
    report_lines.append(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.4f}")
    report_lines.append("=" * 40)

    report_text = "\n".join(report_lines)
    logger.info(report_text)
    return report_text

def calculate_advanced_metrics(trade_list, initial_capital):
    """
    Calculates more advanced metrics from the list of closed trades.
    """
    if not trade_list:
        logger.warning("No closed trades to calculate advanced metrics.")
        return {}

    df_trades = pd.DataFrame(trade_list)

    # Profit/Loss per trade
    df_trades['net_pnl'] = df_trades['pnl'] - df_trades['fee']

    # Win/Loss
    df_trades['is_win'] = df_trades['net_pnl'] > 0

    # Advanced Metrics
    total_trades = len(df_trades)
    wins = df_trades['is_win'].sum()
    losses = total_trades - wins

    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (losses / total_trades) * 100 if total_trades > 0 else 0

    # Average Win / Loss (in USD)
    avg_win = df_trades[df_trades['is_win']]['net_pnl'].mean() if wins > 0 else 0
    avg_loss = df_trades[~df_trades['is_win']]['net_pnl'].mean() if losses > 0 else 0 # Note: Mean of negative numbers

    # Profit Factor
    gross_profits = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_losses = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum()) # Use absolute value
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    # Expectancy (Average profit per trade)
    expectancy = df_trades['net_pnl'].mean()

    # Largest Win / Loss
    largest_win = df_trades['net_pnl'].max()
    largest_loss = df_trades['net_pnl'].min()

    # Average Holding Time (Approximate, based on timestamps)
    # Note: Requires datetime conversion of timestamps
    try:
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'], unit='ms')
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'], unit='ms')
        df_trades['holding_time'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 60.0 # In minutes
        avg_holding_time = df_trades['holding_time'].mean()
    except Exception as e:
        logger.warning(f"Could not calculate holding time: {e}")
        avg_holding_time = np.nan

    advanced_metrics = {
        'win_rate_percent': win_rate,
        'loss_rate_percent': loss_rate,
        'average_win_usd': avg_win,
        'average_loss_usd': avg_loss, # This will be a negative number
        'profit_factor': profit_factor,
        'expectancy_usd': expectancy,
        'largest_win_usd': largest_win,
        'largest_loss_usd': largest_loss,
        'average_holding_time_minutes': avg_holding_time
    }
    return advanced_metrics

def print_advanced_metrics(advanced_metrics):
    """Prints the advanced metrics in a readable format."""
    if not advanced_metrics:
        return
    logger.info("--- Advanced Metrics ---")
    logger.info(f"Win Rate:              {advanced_metrics['win_rate_percent']:.2f}%")
    logger.info(f"Loss Rate:             {advanced_metrics['loss_rate_percent']:.2f}%")
    logger.info(f"Average Win:           ${advanced_metrics['average_win_usd']:.4f}")
    logger.info(f"Average Loss:          ${advanced_metrics['average_loss_usd']:.4f}") # Will show negative
    logger.info(f"Profit Factor:         {advanced_metrics['profit_factor']:.4f}")
    logger.info(f"Expectancy (per trade): ${advanced_metrics['expectancy_usd']:.4f}")
    logger.info(f"Largest Win:           ${advanced_metrics['largest_win_usd']:.4f}")
    logger.info(f"Largest Loss:          ${advanced_metrics['largest_loss_usd']:.4f}")
    if not np.isnan(advanced_metrics['average_holding_time_minutes']):
        logger.info(f"Avg Holding Time:      {advanced_metrics['average_holding_time_minutes']:.2f} minutes")


def plot_equity_curve(equity_curve, timestamps=None):
    """
    Plots the equity curve using matplotlib.
    """
    try:
        plt.figure(figsize=(12, 6))
        if timestamps is not None and len(timestamps) == len(equity_curve):
            # Assume timestamps are in milliseconds
            ts_converted = pd.to_datetime(timestamps, unit='ms')
            plt.plot(ts_converted, equity_curve, label='Equity')
            plt.xlabel('Time')
            # Improve date formatting on x-axis if needed
        else:
            plt.plot(equity_curve, label='Equity')
            plt.xlabel('Bar')
        plt.ylabel('Capital (USDT)')
        plt.title('Backtest Equity Curve')
        plt.legend()
        plt.grid(True)
        # plt.tight_layout() # Adjust layout if needed
        plt.show()
        logger.info("Equity curve plot displayed.")
    except ImportError:
        logger.warning("Matplotlib not installed. Skipping equity curve plot.")
    except Exception as e:
        logger.error(f"Error plotting equity curve: {e}")

def plot_trade_pnl(trade_list):
    """
    Plots the PnL of individual trades.
    """
    try:
        if not trade_list:
             logger.warning("No closed trades to plot PnL for.")
             return
        df_trades = pd.DataFrame(trade_list)
        df_trades['net_pnl'] = df_trades['pnl'] - df_trades['fee']
        df_trades['trade_number'] = range(1, len(df_trades) + 1)

        plt.figure(figsize=(12, 6))
        # Scatter plot or bar plot
        plt.bar(df_trades['trade_number'], df_trades['net_pnl'], color=['green' if x > 0 else 'red' for x in df_trades['net_pnl']])
        plt.xlabel('Trade Number')
        plt.ylabel('Net PnL (USDT)')
        plt.title('Individual Trade PnL')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
        logger.info("Trade PnL plot displayed.")
    except ImportError:
        logger.warning("Matplotlib not installed. Skipping trade PnL plot.")
    except Exception as e:
        logger.error(f"Error plotting trade PnL: {e}")

# --- Example of how to use reporter functions ---
# (This would typically be called from run_backtest.py after the engine finishes)
# if __name__ == "__main__" or within run_backtest.py:
#     # Assume metrics, trade_list are outputs from engine.run_backtest
#     generate_summary_report(metrics, trade_list)
#     advanced_metrics = calculate_advanced_metrics(trade_list, metrics['initial_capital'])
#     print_advanced_metrics(advanced_metrics)
#     # Plotting (requires data)
#     # plot_equity_curve(metrics.get('equity_curve', []), df['timestamp'].tolist() if df is not None else None)
#     # plot_trade_pnl(trade_list)