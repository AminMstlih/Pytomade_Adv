# run_backtest.py
"""
Main script to run the backtest for the Pytomade Bot.
"""

import sys
import os

# --- Add the project root directory to sys.path ---
# This allows importing modules from 'bot', 'market', 'backtest' etc.
project_root = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of this script
sys.path.insert(0, project_root)

from bot import config
from market.data_fetcher import get_historical_candles # For fetching data directly
from backtest.engine import run_backtest, load_historical_data_from_csv
from backtest.reporter import generate_summary_report, calculate_advanced_metrics, print_advanced_metrics # , plot_equity_curve, plot_trade_pnl

def main():
    """Main function to orchestrate the backtest."""
    print("Starting Pytomade Bot Backtest...")
    print(f"Instrument: {config.INSTRUMENT}")
    print(f"Target Margin: ${config.TARGET_MARGIN_USD}, Leverage: {config.LEVERAGE}x")
    print("-" * 30)

    # --- 1. Load Historical Data ---
    historical_df = None
    data_source = "api" # Options: "api", "csv"
    csv_file_path = f"historical_data_{config.INSTRUMENT.replace('-', '_')}_1H.csv" # Example filename

    if data_source == "csv":
        print(f"Loading data from CSV: {csv_file_path}")
        historical_df = load_historical_data_from_csv(csv_file_path)
    elif data_source == "api":
        print(f"Fetching data directly from OKX API for {config.INSTRUMENT}...")
        # Fetch a reasonable amount of historical 1-hour data for backtesting
        # Adjust 'bar' and 'limit' as needed. OKX has limits (e.g., 100 for 1H).
        # You might need to fetch multiple batches for longer history.
        historical_df = get_historical_candles(config.INSTRUMENT, bar="1H", limit=300)
        # Optional: Save fetched data to CSV for future use
        # if historical_df is not None and not historical_df.empty:
        #     historical_df.to_csv(csv_file_path, index=False)
        #     print(f"Saved fetched data to {csv_file_path}")

    if historical_df is None or historical_df.empty:
        print("Error: Failed to load historical data. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(historical_df)} candles for backtesting.")

    # --- 2. Run the Backtest Engine ---
    print("Running backtest...")
    df_with_signals, closed_trades, final_metrics = run_backtest(
        historical_df,
        initial_capital=100.0, # Use virtual capital
        fee_rate=0.001 # 0.1% fee per side (adjust if needed)
    )

    # --- 3. Generate and Display Reports ---
    if final_metrics:
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        generate_summary_report(final_metrics, closed_trades)

        # Calculate and print advanced metrics
        advanced_metrics = calculate_advanced_metrics(closed_trades, final_metrics['initial_capital'])
        print_advanced_metrics(advanced_metrics)

        # --- 4. Optional Plotting (Uncomment if you have matplotlib installed) ---
        # print("\nGenerating plots...")
        # try:
        #     from backtest.reporter import plot_equity_curve, plot_trade_pnl
        #     plot_equity_curve(final_metrics.get('equity_curve', []), df_with_signals['timestamp'].tolist() if df_with_signals is not None else None)
        #     plot_trade_pnl(closed_trades)
        # except ImportError:
        #     print("Matplotlib not found. Skipping plots. Install with 'pip install matplotlib' to enable plotting.")

        print("\nBacktest finished.")
    else:
        print("Backtest completed, but no metrics were returned. Check logs for details.")

if __name__ == "__main__":
    main()