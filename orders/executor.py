# orders/executor.py
import math # Add this import at the top if not already there
from exchange.okx_client import get_public_data, get_private_data, post_private_data
from bot.logger_config import logger

# (Other functions remain largely the same until place_order)

def get_contract_size(instId):
    """Fetches the contract size (e.g., how much base currency 1 contract represents)."""
    # This might be static for some contracts or fetched via API.
    # For most standard perpetual swaps on OKX, it's often 1, but check for KAITO.
    # Checking via instruments API is more robust.
    try:
        response = get_public_data(f"/api/v5/public/instruments?instType=SWAP&instId={instId}")
        if response["code"] == "0" and response["data"]:
             ct_val = float(response["data"][0].get("ctVal", 1)) # Contract Value in base currency
             # ct_mult = float(response["data"][0].get("ctMult", 1)) # Contract Multiplier (usually 1)
             logger.debug(f"Fetched contract details for {instId}: ctVal={ct_val}")
             return ct_val
        else:
             logger.warning(f"Could not fetch contract details for {instId}, assuming ctVal=1. Error: {response.get('msg', 'Unknown')}")
             return 1.0 # Default fallback
    except Exception as e:
         logger.error(f"Error fetching contract size for {instId}: {e}. Assuming ctVal=1.")
         return 1.0 # Default fallback

def place_order(instId, details):
    """
    Places the main order with attached TP/SL orders.
    `details` is the dictionary returned by risk.manager.calculate_position_details
    """
    if not details:
        logger.error("Cannot place order, details dictionary is None or empty.")
        return None

    try:
        # --- Determine size units ---
        # Fetch contract value (how much base currency 1 contract is worth)
        contract_value = get_contract_size(instId) # e.g., 1 KAITO per contract

        # Calculate size in contracts based on the USDT size and entry price
        # Size (contracts) = (Size_USDT) / (Entry_Price * Contract_Value)
        # We use entry price as an estimate. OKX might use mark price or last price internally.
        size_in_contracts = details["size_usdt"] / (details["entry_price"] * contract_value)
        # OKX usually requires integer contract sizes for market orders. Round down to be safe.
        size_in_contracts = math.floor(size_in_contracts)
        # Ensure size is at least 1 contract
        size_in_contracts = max(1, size_in_contracts)

        # Similarly, calculate TP/SL sizes in contracts
        tp1_size_contracts_raw = (details["tp1_size_usdt"] / (details["entry_price"] * contract_value))
        tp1_size_contracts = max(1, math.floor(tp1_size_contracts_raw)) # Floor TP1 size, min 1

        #    Calculate TP2 size as the REMAINDER to ensure total TP size matches main order size
        #    This is the key fix!
        #    Total TP size MUST equal main order size.
        #    So, TP2 size = Main Order Size - TP1 size
        #    We need to make sure TP2 size is at least 1.
        tp2_size_contracts = size_in_contracts - tp1_size_contracts
        tp2_size_contracts = max(1, tp2_size_contracts) # Ensure TP2 is at least 1 contract

        # 4. SL size is typically the full intended size, but for conditional orders,
        #    it might be tied to the main order size or handled differently by OKX.
        #    Let's keep it as the calculated/floored size for now, or also match main order size.
        #    Check OKX docs or test. For now, let's floor it like others.
        sl_size_contracts_raw = (details["sl_size_usdt"] / (details["entry_price"] * contract_value))
        sl_size_contracts = max(1, math.floor(sl_size_contracts_raw)) # Floor SL size, min 1

        # --- Optional: Add a sanity check log (can be removed later) ---
        if (tp1_size_contracts + tp2_size_contracts) != size_in_contracts:
            logger.warning(f"Sanity check failed after size calculation for {instId}: "
                   f"TP1 ({tp1_size_contracts}) + TP2 ({tp2_size_contracts}) = {tp1_size_contracts + tp2_size_contracts} "
                   f"!= Main Order Size ({size_in_contracts}). This should not happen with the new logic.")

        # --- Ensure the attach_algo_ords uses the corrected sizes ---
        attach_algo_ords = [
            {
                "algoOrdType": "conditional",
                "side": details["close_side"],
                "posSide": details["pos_side"],
                "ordType": "market",
                "sz": str(tp1_size_contracts), # Use corrected TP1 size
                # "tgtCcy": "quote_ccy", # REMOVED
                "tpTriggerPx": str(details["tp1_price"]),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last"
                },
                {
                    "algoOrdType": "conditional",
                    "side": details["close_side"],
                    "posSide": details["pos_side"],
                    "ordType": "market",
                    "sz": str(tp2_size_contracts), # Use corrected TP2 size
                    # "tgtCcy": "quote_ccy", # REMOVED
                    "tpTriggerPx": str(details["tp2_price"]),
                    "tpOrdPx": "-1",
                    "tpTriggerPxType": "last"
                    },
                    {
                        "algoOrdType": "conditional",
                        "side": details["close_side"],
                        "posSide": details["pos_side"],
                        "ordType": "market",
                        "sz": str(sl_size_contracts), # Use corrected SL size
                        # "tgtCcy": "quote_ccy", # REMOVED
                        "slTriggerPx": str(details["sl_price"]),
                        "slOrdPx": "-1",
                        "slTriggerPxType": "last"
                        }
                        ]

        # Main order payload - Specify size in contracts
        payload = {
            "instId": instId,
            "tdMode": "cross",
            "side": details["order_side"],
            "posSide": details["pos_side"],
            "ordType": "market",
            "sz": str(size_in_contracts), # Size in contracts
            # "tgtCcy": "quote_ccy", # REMOVE this line
            # "px": str(details["entry_price"]), # Not needed for market order
            "attachAlgoOrds": attach_algo_ords
        }

        logger.info(f"Placing {details['pos_side']} order for {instId}: "
                    f"Size={size_in_contracts} contracts (~${details['size_usdt']:.2f} margin=${details['margin_used_usd']:.4f}), "
                    f"Entry~${details['entry_price']:.6f}, "
                    f"TP1=${details['tp1_price']:.6f} ({tp1_size_contracts} contracts ~${details['tp1_size_usdt']:.2f}), "
                    f"TP2=${details['tp2_price']:.6f} ({tp2_size_contracts} contracts ~${details['tp2_size_usdt']:.2f}), "
                    f"SL=${details['sl_price']:.6f} ({sl_size_contracts} contracts ~${details['sl_size_usdt']:.2f})")

        response = post_private_data("/api/v5/trade/order", payload)

        # (Rest of the response handling remains the same)
        if response["code"] == "0" and response["data"]:
            ord_id = response["data"][0]["ordId"]
            logger.info(f"Main order placed successfully for {instId}: ordId={ord_id}")
            return {"ordId": ord_id, "size_contracts": size_in_contracts, "size_usdt_approx": details["size_usdt"], "entry_price": details["entry_price"], "pos_side": details["pos_side"]}
        else:
            logger.error(f"Order placement failed for {instId}: {response.get('msg', 'Unknown error')} (code: {response.get('code', 'N/A')})")
            logger.debug(f"Failed order payload: {payload}")
            # Log the specific error from the data array if available
            if response.get("data") and isinstance(response["data"], list) and len(response["data"]) > 0:
                s_msg = response["data"][0].get("sMsg", "")
                s_code = response["data"][0].get("sCode", "")
                if s_msg or s_code:
                    logger.error(f"Specific error details: sCode={s_code}, sMsg='{s_msg}'")
            return None

    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e: # Catch math/lookup errors
        logger.error(f"Error calculating contract sizes or preparing order for {instId}: {e}")
        logger.debug(f"Details provided: {details}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing order for {instId}: {e}")
        logger.debug(f"Details provided: {details}")
        return None
    
# (check_open_positions and close_position functions remain largely the same)
# Ensure check_open_positions correctly parses the size returned by OKX (it's in contracts)
def check_open_positions(instId):
    """Checks for currently open positions for the given instrument."""
    response = get_private_data(f"/api/v5/account/positions?instId={instId}")
    positions = {}
    if response["code"] == "0":
        try:
            for pos in response["data"]:
                if pos.get("pos", "0") not in ["0", ""]: # Check if position size is non-zero
                    pos_side = pos["posSide"]
                    # pos_size is in contracts, avgPx is entry price
                    positions[pos_side] = {
                        "side": pos_side,
                        "size_contracts": float(pos["pos"]), # Raw contract size
                        "entry_price": float(pos["avgPx"]) if pos.get("avgPx") else 0
                        # Add more fields if needed (e.g., unrealized PnL)
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
        # It's common for this to fail if the position is already closed by TP/SL
        return False # Indicate potential issue, but don't stop the bot