# bot/main.py
"""Main execution loop for the Pytomade Bot."""

import time
import sys
from bot import config, logger_config
from bot.state_manager import get_last_signal, set_last_signal, get_last_trade_time, set_last_trade_time, is_signal_cooldown_active
from exchange import okx_client  # Import to ensure session is initialized
from market import data_fetcher, condition_checker, strategy
from risk import manager as risk_manager
from orders import executor as order_executor

# Initialize logger
logger = logger_config.logger

def run():
    """Main bot execution function."""
    logger.info("=" * 50)
    logger.info("Starting Pytomade Bot v3.1.7")
    logger.info(f"Target Instrument: {config.INSTRUMENT}")
    logger.info(f"Target Margin (USD): ${config.TARGET_MARGIN_USD}")
    logger.info(f"Leverage: {config.LEVERAGE}x")
    logger.info(f"Trade Mode: {config.TRADE_MODE}")
    logger.info("=" * 50)

    # --- Initial Setup Checks ---
    if not risk_manager.check_unified_account():
        logger.error("Account is NOT in Unified mode. Bot cannot proceed safely.")
        return  # Exit if account mode is incorrect

    # Set leverage (OKX requires this to be set before trading)
    if not risk_manager.set_leverage(config.INSTRUMENT, config.LEVERAGE, "cross"):
        logger.error("Failed to set leverage. Bot cannot proceed.")
        return  # Exit if leverage cannot be set

    logger.info("Initial setup checks passed. Entering main loop...")
    logger.info("-" * 30)

    # --- Main Trading Loop ---
    while True:
        try:
            loop_start_time = time.time()
            logger.debug("Starting main loop iteration...")

            # 1. Cooldown Check
            if is_signal_cooldown_active(config.SIGNAL_COOLDOWN_SECONDS):
                remaining_cooldown = config.SIGNAL_COOLDOWN_SECONDS - (time.time() - get_last_trade_time())
                logger.debug(f"Signal cooldown active. Waiting {remaining_cooldown:.1f}s...")
                time.sleep(min(config.POLLING_INTERVAL_SECONDS, remaining_cooldown))
                continue  # Skip the rest of the loop

            # 2. Market Condition Check
            if not condition_checker.check_market_conditions(config.INSTRUMENT):
                logger.info(f"Market conditions for {config.INSTRUMENT} not favorable. Waiting...")
                time.sleep(config.POLLING_INTERVAL_SECONDS)
                continue

            # 3. Fetch Market Data (Candles)
            df = data_fetcher.get_historical_candles(config.INSTRUMENT, bar="1m", limit=100)
            if df is None or df.empty:
                logger.warning("Failed to fetch valid candle data. Waiting...")
                time.sleep(config.POLLING_INTERVAL_SECONDS)
                continue

            # 4. Calculate Indicators
            df = strategy.calculate_indicators(df)
            if df is None:
                logger.warning("Failed to calculate indicators. Waiting...")
                time.sleep(config.POLLING_INTERVAL_SECONDS)
                continue

            # 5. Generate Signal
            signal = strategy.generate_signal(df)
            if signal is None:
                logger.info("No valid trading signal generated. Waiting...")
                time.sleep(config.POLLING_INTERVAL_SECONDS)
                continue

            # 6. Check Open Positions
            current_positions = order_executor.check_open_positions(config.INSTRUMENT)

            # 7. Decision Making & Order Placement
            if signal not in current_positions:
                logger.info(f"Signal '{signal}' detected and no existing {signal} position. Evaluating entry...")

                # 7a. Handle opposite position based on trade mode
                opposite_signal = "short" if signal == "long" else "long"
                if opposite_signal in current_positions:
                    if config.TRADE_MODE == "close":
                        logger.info(f"Closing opposite {opposite_signal} position before opening new {signal} position")
                        close_success = order_executor.close_position(config.INSTRUMENT, opposite_signal)
                        if not close_success:
                            logger.error(f"Failed to close {opposite_signal} position. Skipping trade.")
                            time.sleep(config.POLLING_INTERVAL_SECONDS)
                            continue
                        # Wait a moment for position to close
                        time.sleep(1)
                        # Refresh positions after closing
                        current_positions = order_executor.check_open_positions(config.INSTRUMENT)
                    else:
                        logger.info(f"Hedge mode: Keeping {opposite_signal} position open while opening {signal} position")

                # 7b. Risk Check: Balance
                required_margin = config.TARGET_MARGIN_USD
                if not risk_manager.check_account_balance(required_margin):
                    logger.error("Insufficient USDT balance for intended position. Waiting...")
                    time.sleep(config.POLLING_INTERVAL_SECONDS)
                    continue  # Skip placing order

                # 7c. Get Entry Price
                entry_price = df["close"].iloc[-1]
                if entry_price <= 0:
                    logger.error("Invalid entry price retrieved. Skipping order.")
                    time.sleep(config.POLLING_INTERVAL_SECONDS)
                    continue

                # 7d. Calculate Order Details
                order_details = risk_manager.calculate_position_details(signal, entry_price)
                if order_details is None:
                    logger.error("Failed to calculate order details. Skipping order.")
                    time.sleep(config.POLLING_INTERVAL_SECONDS)
                    continue

                # 7e. Place Order
                order_result = order_executor.place_order(config.INSTRUMENT, order_details)
                if order_result:
                    logger.info(f"Successfully placed {signal} order: {order_result}")
                    # Update state
                    set_last_signal(signal)
                    set_last_trade_time(time.time())

                    # Trailing Stop Logic
                    if hasattr(config, 'TRAILING_STOP_ACTIVATION_PNL_PERCENT') and \
                       config.TRAILING_STOP_ACTIVATION_PNL_PERCENT > 0:
                        
                        entry_price_for_ts = order_result.get('entry_price')
                        size_for_ts = order_result.get('size_contracts', 0)
                        pos_side_for_ts = order_result.get('pos_side')
                        
                        if entry_price_for_ts and size_for_ts > 0 and pos_side_for_ts:
                            ts_activation_pnl = config.TRAILING_STOP_ACTIVATION_PNL_PERCENT
                            ts_callback_type = config.TRAILING_STOP_CALLBACK_TYPE
                            ts_callback_value = config.TRAILING_STOP_CALLBACK_RATIO_PERCENT
                            ts_inst_id = config.INSTRUMENT
                            ts_td_mode = "cross"

                            ts_response = order_executor.place_trailing_stop(
                                instId=ts_inst_id,
                                entry_price=entry_price_for_ts,
                                pos_size_contracts=size_for_ts,
                                activation_pnl_percent=ts_activation_pnl,
                                callback_type=ts_callback_type,
                                callback_value=ts_callback_value,
                                position_side=pos_side_for_ts,
                                td_mode=ts_td_mode
                            )
                            if ts_response:
                                logger.info(f"Trailing stop order initiated for new {pos_side_for_ts} position.")
                            else:
                                logger.error(f"Failed to initiate trailing stop for new {pos_side_for_ts} position.")
                        else:
                            logger.warning("Could not get order details for trailing stop placement.")
                else:
                    logger.error("Order placement failed.")
            else:
                logger.info(f"Signal '{signal}' detected, but {signal} position already exists. Skipping entry.")

            # --- End of Loop ---
            loop_end_time = time.time()
            loop_duration = loop_end_time - loop_start_time
            sleep_time = max(0, config.POLLING_INTERVAL_SECONDS - loop_duration)
            logger.debug(f"Loop iteration took {loop_duration:.2f}s. Sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Initiating graceful shutdown...")
            # Attempt to close any open positions
            try:
                final_positions = order_executor.check_open_positions(config.INSTRUMENT)
                if final_positions:
                    logger.info(f"Found open positions during shutdown: {list(final_positions.keys())}")
                    for pos_side in list(final_positions.keys()):
                        logger.info(f"Attempting to close {pos_side} position...")
                        order_executor.close_position(config.INSTRUMENT, pos_side)
                else:
                    logger.info("No open positions found during shutdown.")
            except Exception as e:
                logger.error(f"Error during position closing on shutdown: {e}")
            logger.info("Bot shutdown complete.")
            break  # Exit the while loop

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            logger.info("Continuing loop after error...")
            time.sleep(config.POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()