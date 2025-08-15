# Pytomade Bot v3.1.7

An automated trading bot for OKX Futures/Swap markets, designed for small balances and high-frequency, low-risk trades.

## Features

*   **Modular Design:** Organized codebase for easy maintenance and modification.
*   **OKX Focused:** Built specifically for the OKX Futures/Swap API v5.
*   **Configurable:** Easily adjust trading parameters, instruments, risk settings via `bot/config.py`.
*   **Small Balance Optimized:** Trades based on a target margin cost (e.g., $2) rather than fixed contract sizes.
*   **Risk Management:** Implements take-profit (TP1, TP2) and stop-loss (SL) orders automatically.
*   **Market Filtering:** Checks 24h volume, spread, and volatility before trading.
*   **Technical Strategy:** Uses MA, StochRSI, ADX, and volume indicators.
*   **Hedging Support:** Can potentially hold both long and short positions (hedging logic can be expanded).
*   **Logging:** Comprehensive logging to console and file for debugging and monitoring.

## Prerequisites

*   Python 3.7 or higher
*   An OKX account with Futures trading enabled
*   OKX API Key, Secret Key, and Passphrase with relevant permissions (Trade, Read)

## Installation

1.  **Clone or Download:** Get the project files.
2.  **Install Dependencies:** Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API Keys:**
    *   Rename the `.env.example` file (if provided, or just create `.env`) in the project root.
    *   Edit `.env` and add your OKX API credentials:
        ```env
        OKX_API_KEY=your_actual_api_key_here
        OKX_SECRET_KEY=your_actual_secret_key_here
        OKX_PASSPHRASE=your_actual_passphrase_here
        ```
    *   **Keep `.env` secret!** Ensure it's listed in `.gitignore`.

## Configuration

Modify `bot/config.py` to customize the bot's behavior:

*   `INSTRUMENT`: The OKX Futures/Swap pair to trade (e.g., `DOGE-USDT-SWAP`, `WCT-USDT-SWAP`).
*   `LEVERAGE`: Leverage to use.
*   `TARGET_MARGIN_USD`: The desired margin cost per trade in USD (e.g., `2.0`).
*   `TP1_PNL_PERCENT`, `TP2_PNL_PERCENT`, `SL_PNL_PERCENT`: PNL targets for exits.
*   `MA_FAST_PERIOD`, `MA_SLOW_PERIOD`, etc.: Strategy indicator parameters.
*   `POLLING_INTERVAL_SECONDS`: How often the bot checks for signals.
*   `SIGNAL_COOLDOWN_SECONDS`: Minimum time between trade attempts for the *same* signal.
*   Market filter thresholds (`MIN_24H_VOLUME_USDT`, etc.).
*   Logging level (`LOG_LEVEL`).

## Running the Bot

Navigate to the project root directory in your terminal and run:

```bash
python -m bot.main