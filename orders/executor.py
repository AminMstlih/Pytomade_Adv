# orders/executor.py - QUICK FIX VERSION
# Replace your current executor with this to fix the immediate contract sizing issues
# Updated with trailing stop workaround: reduceOnly and orphan cleanup function

import math
from decimal import Decimal, ROUND_HALF_UP
from exchange.okx_client import get_public_data, get_private_data, post_private_data
from bot.logger_config import logger
from bot import config

def normalize_contract_size(size_usdt, instId, entry_price):
    """
    Convert notional size in USDT to valid contract size according to OKX rules.
    ENHANCED VERSION with proper decimal arithmetic to fix floating point errors.
    """
    try:
        response = get_public_data(f"/api/v5/public/instruments?instType=SWAP&instId={instId}")
        if response["code"] != "0" or not response.get("data"):
            logger.warning(f"normalize_contract_size: Failed to fetch instrument data for {instId}, fallback size=1")
            return (1.0, entry_price)

        inst = response["data"][0]
        ctVal = float(inst.get("ctVal", 1))
        lotSz = float(inst.get("lotSz", 1))
        minSz = float(inst.get("minSz", lotSz))

        # Calculate raw contracts using decimal arithmetic for precision
        raw_contracts = size_usdt / (ctVal * entry_price)
        
        # CRITICAL FIX: Use Decimal for precise lot size calculation
        raw_decimal = Decimal(str(raw_contracts))
        lot_decimal = Decimal(str(lotSz))
        min_decimal = Decimal(str(minSz))
        
        # Round to nearest lot size multiple
        lot_multiple = (raw_decimal / lot_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        contracts_valid_decimal = lot_multiple * lot_decimal
        
        # Ensure minimum size
        contracts_valid_decimal = max(min_decimal, contracts_valid_decimal)
        
        # Convert back to float
        contracts_valid = float(contracts_valid_decimal)
        
        # Calculate actual notional
        notional_actual = contracts_valid * ctVal * entry_price

        logger.debug(f"Contract sizing for {instId}: {size_usdt} USDT -> {contracts_valid} contracts (~{notional_actual:.2f} USDT)")
        
        return (contracts_valid, notional_actual)

    except Exception as e:
        logger.error(f"normalize_contract_size error for {instId}: {e}", exc_info=True)
        return (1.0, entry_price)

def format_contract_size(size, lot_size):
    """Format contract size to avoid floating point precision issues."""
    if lot_size >= 1:
        return str(int(size))
    else:
        # Determine decimal places based on lot size
        lot_str = f"{lot_size:.10f}".rstrip('0').rstrip('.')
        if '.' in lot_str:
            decimal_places = len(lot_str.split('.')[1])
        else:
            decimal_places = 0
        return f"{size:.{decimal_places}f}"

def place_order(instId, details):
    """
    Places the main order with attached TP/SL orders.
    ENHANCED with proper contract sizing and formatting.
    """
    if not details:
        logger.error("Cannot place order, details dictionary is None or empty.")
        return None

    try:
        # Get lot size for formatting
        response = get_public_data(f"/api/v5/public/instruments?instType=SWAP&instId={instId}")
        if response["code"] == "0" and response.get("data"):
            lot_size = float(response["data"][0].get("lotSz", 0.01))
        else:
            lot_size = 0.01  # Default fallback
        
        # Calculate main order size using enhanced normalize function
        size_in_contracts, size_usdt_actual = normalize_contract_size(
            details["size_usdt"], instId, details["entry_price"]
        )

        # Calculate TP1 size
        tp1_size_contracts, tp1_usdt_actual = normalize_contract_size(
            details["tp1_size_usdt"], instId, details["entry_price"]
        )
        
        # Calculate TP2 as remainder, ensuring it's valid
        tp2_size_contracts = size_in_contracts - tp1_size_contracts
        
        # Ensure TP2 is either 0 or >= lot size
        if 0 < tp2_size_contracts < lot_size:
            # Too small, merge with TP1
            tp1_size_contracts = size_in_contracts
            tp2_size_contracts = 0
            logger.warning(f"TP2 size too small, merged with TP1")

        # SL size equals main order size
        sl_size_contracts = size_in_contracts

        # Build attached orders with proper formatting
        attach_algo_ords = []
        
        if tp1_size_contracts > 0:
            attach_algo_ords.append({
                "algoOrdType": "conditional",
                "side": details["close_side"],
                "posSide": details["pos_side"],
                "ordType": "market",
                "sz": format_contract_size(tp1_size_contracts, lot_size),
                "tpTriggerPx": str(details["tp1_price"]),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last"
            })
        
        if tp2_size_contracts > 0:
            attach_algo_ords.append({
                "algoOrdType": "conditional",
                "side": details["close_side"],
                "posSide": details["pos_side"],
                "ordType": "market",
                "sz": format_contract_size(tp2_size_contracts, lot_size),
                "tpTriggerPx": str(details["tp2_price"]),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last"
            })
        
        attach_algo_ords.append({
            "algoOrdType": "conditional",
            "side": details["close_side"],
            "posSide": details["pos_side"],
            "ordType": "market",
            "sz": format_contract_size(sl_size_contracts, lot_size),
            "slTriggerPx": str(details["sl_price"]),
            "slOrdPx": "-1",
            "slTriggerPxType": "last"
        })

        # Main order payload with proper formatting
        payload = {
            "instId": instId,
            "tdMode": "cross",
            "side": details["order_side"],
            "posSide": details["pos_side"],
            "ordType": "market",
            "sz": format_contract_size(size_in_contracts, lot_size),
            "attachAlgoOrds": attach_algo_ords
        }

        logger.info(f"Placing {details['pos_side']} order for {instId}: "
                    f"Size={format_contract_size(size_in_contracts, lot_size)} contracts (~${size_usdt_actual:.2f}), "
                    f"Entry~${details['entry_price']:.6f}")
        logger.info(f"TP1={format_contract_size(tp1_size_contracts, lot_size)}@{details['tp1_price']:.6f}, "
                   f"TP2={format_contract_size(tp2_size_contracts, lot_size)}@{details['tp2_price']:.6f}, "
                   f"SL={format_contract_size(sl_size_contracts, lot_size)}@{details['sl_price']:.6f}")

        response = post_private_data("/api/v5/trade/order", payload)

        if response["code"] == "0" and response["data"]:
            ord_id = response["data"][0]["ordId"]
            logger.info(f"Order placed successfully for {instId}: ordId={ord_id}")
            return {
                "ordId": ord_id,
                "size_contracts": size_in_contracts,
                "size_usdt_actual": size_usdt_actual,
                "entry_price": details["entry_price"],
                "pos_side": details["pos_side"]
            }
        else:
            logger.error(f"Order placement failed for {instId}: {response.get('msg', 'Unknown error')} "
                        f"(code: {response.get('code', 'N/A')})")
            
            if response.get("data") and isinstance(response["data"], list) and len(response["data"]) > 0:
                s_msg = response["data"][0].get("sMsg", "")
                s_code = response["data"][0].get("sCode", "")
                if s_msg or s_code:
                    logger.error(f"Specific error: sCode={s_code}, sMsg='{s_msg}'")
            
            logger.debug(f"Failed payload: {payload}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error placing order for {instId}: {e}", exc_info=True)
        return None

def place_trailing_stop(instId, entry_price, pos_size_contracts, activation_pnl_percent, 
                        callback_type, callback_value, position_side, td_mode="cross"):
    """Places a trailing stop order using OKX Algo Order API v5."""
    if not all([instId, entry_price, pos_size_contracts, position_side]):
        logger.error("place_trailing_stop: Missing required arguments.")
        return None

    if callback_type not in ["percent", "constant"]:
        logger.error(f"Invalid callback_type '{callback_type}'. Must be 'percent' or 'constant'.")
        return None

    if pos_size_contracts <= 0 or entry_price <= 0:
        logger.error(f"Invalid pos_size ({pos_size_contracts}) or entry_price ({entry_price}).")
        return None

    if position_side not in ["long", "short"]:
        logger.error(f"Invalid position_side '{position_side}'. Must be 'long' or 'short'.")
        return None

    try:
        # Get lot size for proper formatting
        response = get_public_data(f"/api/v5/public/instruments?instType=SWAP&instId={instId}")
        if response["code"] == "0" and response.get("data"):
            lot_size = float(response["data"][0].get("lotSz", 0.01))
        else:
            lot_size = 0.01
        
        activation_pnl_decimal = activation_pnl_percent / 100.0
        if position_side == "long":
            active_px = entry_price * (1 + activation_pnl_decimal)
        else:
            active_px = entry_price * (1 - activation_pnl_decimal)
        
        active_px = round(active_px, 6)

        payload = {
            "instId": instId,
            "tdMode": td_mode,
            "side": "sell" if position_side == "long" else "buy",
            "posSide": position_side,
            "ordType": "move_order_stop",
            "sz": format_contract_size(pos_size_contracts, lot_size),
            "activePx": str(active_px)
        }
        
        if callback_type == "percent":
            payload["callbackRatio"] = str(callback_value)
        elif callback_type == "constant":
            payload["callbackSpread"] = str(round(callback_value, 2))
        
        logger.info(f"Placing trailing stop for {instId} ({position_side}): "
                    f"Entry={entry_price}, ActivePx={active_px}, "
                    f"CallbackType={callback_type}, CallbackValue={callback_value}, "
                    f"Size={format_contract_size(pos_size_contracts, lot_size)}, ReduceOnly=true")

        response = post_private_data("/api/v5/trade/order-algo", payload)

        if response and response.get("code") == "0":
            algo_id = response.get("data", [{}])[0].get("algoId")
            logger.info(f"Trailing stop placed successfully: AlgoId: {algo_id}")
            return response
        else:
            error_msg = response.get("msg", "Unknown error")
            error_code = response.get("code", "N/A")
            logger.error(f"Trailing stop failed: {error_msg} (code: {error_code})")
            
            if response.get("data") and isinstance(response["data"], list) and len(response["data"]) > 0:
                s_msg = response["data"][0].get("sMsg", "")
                s_code = response["data"][0].get("sCode", "")
                if s_msg or s_code:
                    logger.error(f"Specific error: sCode={s_code}, sMsg='{s_msg}'")
            return None

    except Exception as e:
        logger.error(f"Unexpected error placing trailing stop for {instId}: {e}", exc_info=True)
        return None

def check_pending_algo_orders(instId, ordType=None):
    """Checks for pending algo orders for an instrument."""
    try:
        path = f"/api/v5/trade/orders-algo-pending?instId={instId}"
        if ordType:
            path += f"&ordType={ordType}"
            
        response = get_private_data(path)
        if response and response.get("code") == "0":
            orders = response.get("data", [])
            logger.debug(f"Found {len(orders)} pending algo orders for {instId}")
            return orders
        else:
            logger.warning(f"Failed to fetch pending algo orders for {instId}")
            return []
    except Exception as e:
        logger.error(f"Error checking pending algo orders for {instId}: {e}", exc_info=True)
        return []

def check_open_positions(instId):
    """Checks for currently open positions for the given instrument."""
    response = get_private_data(f"/api/v5/account/positions?instId={instId}")
    positions = {}
    if response["code"] == "0":
        try:
            for pos in response["data"]:
                if pos.get("pos", "0") not in ["0", ""]:
                    pos_side = pos["posSide"]
                    positions[pos_side] = {
                        "side": pos_side,
                        "size_contracts": float(pos["pos"]),
                        "entry_price": float(pos["avgPx"]) if pos.get("avgPx") else 0
                    }
            if positions:
                logger.debug(f"Open positions for {instId}: {positions}")
            else:
                logger.debug(f"No open positions found for {instId}.")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing position data for {instId}: {e}")
            logger.debug(f"Raw position data: {response.get('data', [])}")
    else:
        logger.error(f"Failed to fetch positions for {instId}: {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
    return positions

def close_position(instId, posSide):
    """Attempts to close a specific position."""
    payload = {"instId": instId, "mgnMode": "cross", "posSide": posSide}
    response = post_private_data("/api/v5/trade/close-position", payload)
    if response["code"] == "0":
        logger.info(f"Position {posSide} for {instId} closed successfully (or close request sent).")
        return True
    else:
        logger.warning(f"Failed to close position {posSide} for {instId} (might already be closed or an error): {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
        return False

def cleanup_orphan_trailing(instId, posSide):
    """Cleans up orphan trailing stops if no position exists for the instId and posSide."""
    try:
        positions = check_open_positions(instId)
        if posSide in positions and positions[posSide]["size_contracts"] > 0:
            logger.debug(f"Position exists for {instId} ({posSide}), skipping cleanup.")
            return True  # Position still open, no action

        # No position, check for pending trailing stops
        pending_algos = check_pending_algo_orders(instId, ordType="move_order_stop")
        if not pending_algos:
            logger.debug(f"No pending trailing stops for {instId}, nothing to clean.")
            return True

        # Prepare cancel payload (batch if multiple)
        cancel_payload = []
        for algo in pending_algos:
            if algo.get("posSide") == posSide:  # Match side in hedge mode
                cancel_payload.append({
                    "instId": instId,
                    "algoId": algo["algoId"],
                    "ordType": "move_order_stop"
                })

        if not cancel_payload:
            logger.debug(f"No matching trailing stops for {posSide} on {instId}.")
            return True

        logger.info(f"Cleaning up {len(cancel_payload)} orphan trailing stops for {instId} ({posSide}).")
        response = post_private_data("/api/v5/trade/cancel-algo-order", cancel_payload)

        if response.get("code") == "0":
            logger.info(f"Orphan trailing stops canceled successfully for {instId} ({posSide}).")
            return True
        else:
            error_msg = response.get("msg", "Unknown error")
            error_code = response.get("code", "N/A")
            logger.error(f"Cleanup failed: {error_msg} (code: {error_code})")
            if response.get("data"):
                for item in response["data"]:
                    if item.get("sCode") != "0":
                        logger.error(f"Specific cancel error: algoId={item.get('algoId')}, sCode={item.get('sCode')}, sMsg={item.get('sMsg')}")
            return False

    except Exception as e:
        logger.error(f"Unexpected error during cleanup for {instId} ({posSide}): {e}", exc_info=True)
        return False