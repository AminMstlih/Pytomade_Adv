# orders/executor.py
import math # Add this import at the top if not already there
from exchange.okx_client import get_public_data, get_private_data, post_private_data
from bot.logger_config import logger
from bot import config

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
            
        # --- Decide whether to attach SL based on config ---    
        attach_sl = not getattr(config, 'USE_TRAILING_STOP_INSTEAD_OF_FIXED_SL', False)
    
        # --- Ensure the attach_algo_ords uses the corrected sizes ---
        attach_algo_ords = [
            {
                "algoOrdType": "conditional",
                "side": details["close_side"],
                "posSide": details["pos_side"],
                "ordType": "market",
                "sz": str(tp1_size_contracts),
                "tpTriggerPx": str(details["tp1_price"]),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last"
            },
            {
                "algoOrdType": "conditional",
                "side": details["close_side"],
                "posSide": details["pos_side"],
                "ordType": "market",
                "sz": str(tp2_size_contracts),
                "tpTriggerPx": str(details["tp2_price"]),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last"
            }
        ]

# Conditionally add the SL dictionary
        if attach_sl:
            attach_algo_ords.append({
                "algoOrdType": "conditional",
                "side": details["close_side"],
                "posSide": details["pos_side"],
                "ordType": "market",
                "sz": str(sl_size_contracts),
                "slTriggerPx": str(details["sl_price"]),
                "slOrdPx": "-1",
                "slTriggerPxType": "last"
            })

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

def place_trailing_stop(instId, entry_price, pos_size_contracts, activation_pnl_percent, 
                        callback_type, callback_value, position_side, td_mode="cross"):
    """
    Places a trailing stop order using OKX Algo Order API v5.

    Args:
        instId (str): Instrument ID (e.g., "KAITO-USDT-SWAP").
        entry_price (float): The entry price of the position.
        pos_size_contracts (float): The size of the position in contracts.
        activation_pnl_percent (float): PnL % at which the trailing stop activates.
        callback_type (str): Type of callback ('percent' or 'constant').
        callback_value (float): The callback value (e.g., 0.5 for 0.5% if percent).
        position_side (str): 'long' or 'short'.
        td_mode (str): Trade mode ('cross' or 'isolated'). Default 'cross'.

    Returns:
        dict or None: Response from OKX API or None on failure.
    """
    if not all([instId, entry_price, pos_size_contracts, position_side]):
        logger.error("place_trailing_stop: Missing required arguments.")
        return None

    if callback_type not in ["percent", "constant"]:
        logger.error(f"place_trailing_stop: Invalid callback_type '{callback_type}'. Must be 'percent' or 'constant'.")
        return None

    if pos_size_contracts <= 0 or entry_price <= 0:
         logger.error(f"place_trailing_stop: Invalid pos_size ({pos_size_contracts}) or entry_price ({entry_price}).")
         return None

    if position_side not in ["long", "short"]:
         logger.error(f"place_trailing_stop: Invalid position_side '{position_side}'. Must be 'long' or 'short'.")
         return None

    try:
        # 1. Hitung activePx berdasarkan activation_pnl_percent dan entry_price
        # Formula: activePx = entry_price * (1 + activation_pnl_ratio) untuk Long
        #          activePx = entry_price * (1 - activation_pnl_ratio) untuk Short
        # activation_pnl_ratio = activation_pnl_percent / 100 / leverage
        # Namun, karena OKX trailing stop menggunakan activePx absolut, dan leverage sudah
        # terkandung dalam posisi, kita hitung langsung berdasarkan entry_price.
        # PnL% = (ActivePx / Entry - 1) * Leverage * 100 (Long)
        # Solving for ActivePx: ActivePx = Entry * (1 + PnL% / (Leverage * 100))
        # Untuk trailing stop OKX, activePx adalah harga pasar yang harus dicapai UNTUK AKTIFKAN trailing.
        # Jadi, kita hitung harga yang sesuai dengan PnL% target tsb.
        # Kita asumsikan leverage dari konfigurasi untuk estimasi, tapi sebenarnya trailing stop
        # aktif berdasarkan harga pasar, bukan PnL internal kita. 
        # Lebih aman: Gunakan PnL% langsung untuk menghitung harga aktif.
        # Misalnya, untuk long, jika ingin aktif di 2% profit:
        # ActivePx = Entry * (1 + 0.02) = Entry * 1.02 (jika leverage=1 di perhitungan ini)
        # Tapi karena trailing stop OKX aktif saat harga pasar menyentuh ActivePx,
        # dan PnL% yang kita maksud adalah berdasarkan leverage bot kita,
        # maka kita harus menghitung ActivePx berdasarkan leverage bot kita.
        # Rumus yang benar (mengacu pada logika PnL% bot):
        # Untuk Long: ActivePx = Entry * (1 + (Activation_PnL% / (Leverage * 100)))
        # Untuk Short: ActivePx = Entry * (1 - (Activation_PnL% / (Leverage * 100)))
        # Tapi, OKX trailing stop seringkali didefinisikan lebih sederhana sebagai
        # harga pasar yang harus dicapai untuk mulai trailing.
        # Mari kita gunakan pendekatan sederhana dulu:
        # ActivePx = Entry * (1 + Activation_PnL%_as_decimal) for Long
        # ActivePx = Entry * (1 - Activation_PnL%_as_decimal) for Short
        # Ini berarti kita aktifkan trailing ketika harga pasar naik/turun Activation_PnL% dari entry.
        # Ini mungkin sedikit berbeda dari PnL% internal yang menggunakan leverage, 
        # tapi lebih sesuai dengan cara OKX trailing stop bekerja berdasarkan harga pasar.
        # Mari pakai ini dulu, bisa disesuaikan nanti jika perlu.
        
        activation_pnl_decimal = activation_pnl_percent / 100.0
        active_px = 0.0
        if position_side == "long":
            # Trailing aktif jika harga naik ke level ini
            active_px = entry_price * (1 + activation_pnl_decimal) 
        else: # position_side == "short"
            # Trailing aktif jika harga turun ke level ini
            active_px = entry_price * (1 - activation_pnl_decimal)
        
        # Bulatkan activePx sesuai presisi harga instrumen (biasanya 5 desimal)
        # Untuk lebih aman, bisa fetch dari API /public/instruments, tapi untuk sekarang hardcode 5.
        active_px = round(active_px, 5) 

        # 2. Siapkan payload untuk API OKX
        payload = {
            "instId": instId,
            "tdMode": td_mode,
            "side": "sell" if position_side == "long" else "buy", # Arah order untuk menutup posisi
            "posSide": position_side,
            "ordType": "move_order_stop", # Tipe order Algo
            "sz": str(pos_size_contracts), # Ukuran dalam kontrak
            "activePx": str(active_px), # Harga aktivasi
            # Tambahkan callback parameter berdasarkan tipe
        }
        
        # Tambahkan callback parameter
        if callback_type == "percent":
            # OKX menggunakan string desimal untuk persentase (e.g., "0.1" untuk 0.1%)
            payload["callbackRatio"] = str(callback_value) 
        elif callback_type == "constant":
            # OKX menggunakan string desimal untuk nilai konstan (dalam USD untuk swap)
             # Bulatkan ke presisi yang sesuai, misal 2 desimal untuk USD
            payload["callbackSpread"] = str(round(callback_value, 2)) 

        # 3. Kirim request ke OKX API
        logger.info(f"Placing trailing stop for {instId} ({position_side}): "
                    f"Entry={entry_price}, ActivePx={active_px}, "
                    f"CallbackType={callback_type}, CallbackValue={callback_value}, "
                    f"Size={pos_size_contracts} contracts")
        
        logger.debug(f"Attempting to place trailing stop with payload: {payload}")

        response = post_private_data("/api/v5/trade/order-algo", payload)

        # 4. Tangani response
        if response and response.get("code") == "0":
            algo_id = response.get("data", [{}])[0].get("algoId")
            logger.info(f"Trailing stop order placed successfully for {instId} ({position_side}). "
                        f"AlgoId: {algo_id}")
            return response
        else:
            error_msg = response.get("msg", "Unknown error")
            error_code = response.get("code", "N/A")
            logger.error(f"Failed to place trailing stop for {instId} ({position_side}): "
                         f"{error_msg} (code: {error_code})")
            # Log detail error jika ada di data array
            if response.get("data") and isinstance(response["data"], list) and len(response["data"]) > 0:
                s_msg = response["data"][0].get("sMsg", "")
                s_code = response["data"][0].get("sCode", "")
                if s_msg or s_code:
                    logger.error(f"Specific error details: sCode={s_code}, sMsg='{s_msg}'")
            return None

    except (ValueError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Math error in place_trailing_stop for {instId}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error placing trailing stop for {instId}: {e}", exc_info=True)
        return None


# Fungsi opsional untuk monitoring (bisa dikembangkan lebih lanjut)
def check_pending_algo_orders(instId, ordType=None):
    """
    Checks for pending algo orders for an instrument.
    This is a basic check and can be expanded for specific order types or IDs.
    """
    try:
        # Build request path with parameters
        path = f"/api/v5/trade/orders-algo-pending?instId={instId}"
        if ordType:
            path += f"&ordType={ordType}"
            
        response = get_private_data(path)
        if response and response.get("code") == "0":
            orders = response.get("data", [])
            logger.debug(f"Found {len(orders)} pending algo orders for {instId} (ordType filter: {ordType})")
            # Log detail singkat untuk setiap order
            for order in orders:
                logger.debug(f"Pending Algo Order: ID={order.get('algoId')}, Type={order.get('ordType')}, "
                             f"Inst={order.get('instId')}, Side={order.get('side')}, PosSide={order.get('posSide')}")
            return orders
        else:
            error_msg = response.get("msg", "Unknown error")
            error_code = response.get("code", "N/A")
            logger.warning(f"Failed to fetch pending algo orders for {instId}: {error_msg} (code: {error_code})")
            return []
    except Exception as e:
        logger.error(f"Error checking pending algo orders for {instId}: {e}", exc_info=True)
        return []    

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