# risk/manager.py
import math
from exchange.okx_client import get_private_data, post_private_data
from bot.config import LEVERAGE, TARGET_MARGIN_USD, MAX_POSITION_SIZE_USDT, TP1_PNL_PERCENT, TP2_PNL_PERCENT, TP1_SIZE_RATIO, SL_PNL_PERCENT
from bot.logger_config import logger

def check_account_balance(required_margin_usd):
    """Checks if the account has sufficient USDT balance for the required margin."""
    # Note: For cross margin, available balance is used.
    response = get_private_data("/api/v5/account/balance?ccy=USDT")
    if response["code"] == "0" and response["data"]:
        # OKX structure: data[0] -> details[0] -> availBal
        details = response["data"][0].get("details", [])
        if details:
            avail_bal = float(details[0].get("availBal", 0))
            logger.info(f"USDT available balance: ${avail_bal:.4f}, Required margin: ${required_margin_usd:.4f}")
            return avail_bal >= required_margin_usd
        else:
            logger.warning("Account details not found in balance response.")
            return False
    else:
        logger.error(f"Failed to check balance: {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
        return False

def check_unified_account():
    """Checks if the account is in Unified Account mode (required for certain operations)."""
    response = get_private_data("/api/v5/account/config")
    if response["code"] == "0" and response["data"]:
        acct_mode = response["data"][0].get("acctLv", "0")
        # acctLv: 1=Simple, 2=Single-currency margin, 3=Multi-currency margin, 4=Portfolio margin
        is_unified = acct_mode in ["2", "3", "4"]
        logger.info(f"Account mode check: {'Unified' if is_unified else 'Non-Unified'} (acctLv: {acct_mode})")
        return is_unified
    else:
        logger.error(f"Failed to check account mode: {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
        return False

def set_leverage(instId, leverage, mgnMode="cross"):
    """Sets the leverage for an instrument."""
    payload = {'instId': instId, 'lever': str(leverage), 'mgnMode': mgnMode}
    response = post_private_data('/api/v5/account/set-leverage', payload)
    if response['code'] == '0':
        logger.info(f"Leverage successfully set to {leverage}x for {instId} with {mgnMode} margin mode.")
        return True
    else:
        logger.error(f"Failed to set leverage for {instId}: {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
        return False

def calculate_position_details(signal, entry_price, leverage=LEVERAGE):
    """
    Calculates order size (in USDT) and TP/SL prices based on target margin.
    This is the core of making it work with small balances.
    Returns a dictionary with order parameters or None on error.
    """
    try:
        if signal not in ["long", "short"]:
            logger.error(f"Invalid signal for position calculation: {signal}")
            return None

        if entry_price <= 0:
             logger.error(f"Invalid entry price for position calculation: {entry_price}")
             return None

        # 1. Calculate the desired position size in USDT based on target margin and leverage
        # Size_USDT = Margin_USD * Leverage
        size_usdt = TARGET_MARGIN_USD * leverage
        size_usdt = min(size_usdt, MAX_POSITION_SIZE_USDT) # Cap by max size

        # 2. Calculate TP/SL distances in terms of price change ratio
        # PNL% = (Exit_Price / Entry_Price - 1) * Leverage * 100 (for Long)
        # PNL% = (1 - Exit_Price / Entry_Price) * Leverage * 100 (for Short)
        # Solving for Exit_Price:
        # Long TP1: TP1_Price = Entry * (1 + TP1_PNL% / (100 * Leverage))
        # Long SL:  SL_Price  = Entry * (1 + SL_PNL% / (100 * Leverage)) (SL_PNL% is negative)
        # Short TP1: TP1_Price = Entry * (1 - TP1_PNL% / (100 * Leverage))
        # Short SL:  SL_Price  = Entry * (1 - SL_PNL% / (100 * Leverage)) (SL_PNL% is negative)

        leverage_factor = leverage * 100.0

        if signal == "long":
            tp1_distance_ratio = TP1_PNL_PERCENT / leverage_factor
            tp2_distance_ratio = TP2_PNL_PERCENT / leverage_factor
            sl_distance_ratio = SL_PNL_PERCENT / leverage_factor # SL_PNL_PERCENT is negative

            tp1_price = entry_price * (1 + tp1_distance_ratio)
            tp2_price = entry_price * (1 + tp2_distance_ratio)
            sl_price = entry_price * (1 + sl_distance_ratio) # Adding a negative number

        else: # signal == "short"
            tp1_distance_ratio = TP1_PNL_PERCENT / leverage_factor
            tp2_distance_ratio = TP2_PNL_PERCENT / leverage_factor
            sl_distance_ratio = SL_PNL_PERCENT / leverage_factor # SL_PNL_PERCENT is negative

            tp1_price = entry_price * (1 - tp1_distance_ratio)
            tp2_price = entry_price * (1 - tp2_distance_ratio)
            sl_price = entry_price * (1 - sl_distance_ratio) # Subtracting a negative number

        # 3. Round prices appropriately (OKX usually requires 5 decimal places for price, check instrument specifics if issues arise)
        tp1_price = round(tp1_price, 5)
        tp2_price = round(tp2_price, 5)
        sl_price = round(sl_price, 5)

        # 4. Calculate sizes for TP1 and TP2 (in USDT)
        tp1_size_usdt = round(TP1_SIZE_RATIO * size_usdt, 0) # Round to 2 decimal places for USDT
        tp2_size_usdt = round((1 - TP1_SIZE_RATIO) * size_usdt, 0)
        # SL size is typically the full position size for the main order
        sl_size_usdt = round(size_usdt, 0)

        # 5. Determine order sides
        order_side = "buy" if signal == "long" else "sell"
        close_side = "sell" if signal == "long" else "buy" # Side to close/exit
        pos_side = signal

        margin_used = size_usdt / leverage

        details = {
            "size_usdt": size_usdt,
            "margin_used_usd": margin_used,
            "entry_price": entry_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "sl_price": sl_price,
            "tp1_size_usdt": tp1_size_usdt,
            "tp2_size_usdt": tp2_size_usdt,
            "sl_size_usdt": sl_size_usdt,
            "order_side": order_side,
            "close_side": close_side,
            "pos_side": pos_side
        }
        logger.debug(f"Calculated position details for {signal}: {details}")
        return details

    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as e: # Catch math errors
        logger.error(f"Math error calculating position details for {signal} at price {entry_price}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calculating position details for {signal} at price {entry_price}: {e}")
        return None