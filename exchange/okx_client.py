# exchange/okx_client.py
import time
import hmac
import base64
import json
import requests
from bot.config import OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, OKX_BASE_URL, API_RATE_LIMIT_DELAY
from bot.logger_config import logger
import sys

# --- Add type checks and defaults for environment variables ---
# Ensure these are strings, even if os.getenv returned None
_API_KEY = OKX_API_KEY or "" 
_SECRET_KEY = OKX_SECRET_KEY or ""
_PASSPHRASE = OKX_PASSPHRASE or ""

# Basic check to warn if keys seem unset (though technically "" is a valid string)
if not _API_KEY or not _SECRET_KEY or not _PASSPHRASE:
    logger.warning("OKX API credentials (KEY, SECRET, PASSPHRASE) are not set or are empty strings. API requests will likely fail.")

# HTTP session for optimized API calls
_session = requests.Session()

def get_server_time():
    """Fetches the current server time from OKX."""
    endpoint = "/api/v5/public/time"
    try:
        response = _session.get(OKX_BASE_URL + endpoint)
        response.raise_for_status()
        server_time = str(float(response.json()["data"][0]["ts"]) / 1000.0)
        logger.debug(f"Fetched server time: {server_time}")
        return server_time
    except Exception as e:
        logger.error(f"Error fetching server time: {e}")
        # Fallback to local time if server time fails, not ideal but prevents complete stop
        return str(int(time.time()))

def _generate_signature(timestamp, method, request_path, body=''):
    """Generates the signature for OKX API requests."""
    # --- Use the checked variables ---
    message = timestamp + method.upper() + request_path + body
    # Ensure _SECRET_KEY is bytes
    mac = hmac.new(
        _SECRET_KEY.encode('utf-8'), # Use checked variable
        message.encode('utf-8'),
        digestmod='sha256'
    )
    d = mac.digest()
    return base64.b64encode(d).decode()

def make_request(method, request_path, data=None):
    # ... (setup code) ...

    # --- Crucially, define 'url' and 'body' at the start ---
    url = OKX_BASE_URL + request_path  # <-- Define 'url' here
    # Convert data dictionary to JSON string, or use empty string if None
    body = json.dumps(data) if data else '' # <-- Define 'body' here
    # --- End of crucial definitions ---

    is_private = method.upper() in ['GET', 'POST'] and any(
        prefix in request_path for prefix in ['/api/v5/account', '/api/v5/trade']
    )
    
    headers = {
        'Content-Type': 'application/json',
        'x-simulated-trading': '0'
    }
    
    if is_private:
        # --- Use the checked variables and ensure they are strings ---
        # Even if _API_KEY etc are "", str() ensures they are treated as strings for the dict
        timestamp = get_server_time()
        headers['OK-ACCESS-KEY'] = str(_API_KEY) # Use checked variable, ensure string
        headers['OK-ACCESS-SIGN'] = _generate_signature(timestamp, method, request_path, body)
        headers['OK-ACCESS-TIMESTAMP'] = str(timestamp) # Ensure string
        headers['OK-ACCESS-PASSPHRASE'] = str(_PASSPHRASE) # Use checked variable, ensure string
        logger.debug(f"Generated private headers for {method} {request_path}")

    for attempt in range(3):
        try:
            logger.debug(f"Making {method} request to {url} (Attempt {attempt+1})")
            response = _session.request(method, url, headers=headers, data=body, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Request {method} {request_path} response: code={data.get('code', 'N/A')}")
            
            # Basic rate limiting delay
            time.sleep(API_RATE_LIMIT_DELAY)
            
            # Check for common OKX error codes in the response body
            if data.get("code") != "0":
                 logger.warning(f"OKX API returned non-zero code for {method} {request_path}: {data}")
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Request {method} {request_path} failed (attempt {attempt+1}/3): {e}")

            # --- Enhanced Error Logging ---
            error_data = {"code": "1", "msg": f"Network/Request error: {str(e)}", "data": []}
            error_logged = False # Flag to track if we logged specific details

            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Raw error response text for {method} {request_path}: {e.response.text}") # Log raw text first
                try:
                    # Try to parse the error response body as JSON
                    error_body = e.response.json()
                    logger.error(f"Parsed error response body for {method} {request_path}: {error_body}") # Log parsed JSON
                    error_data.update(error_body) # Merge parsed data

                    # Extract and log specific error details (sMsg, sCode) if present
                    # This is the critical part often found in data[0] for order errors
                    if isinstance(error_body.get("data"), list) and len(error_body["data"]) > 0:
                        error_item = error_body["data"][0]
                        if isinstance(error_item, dict):
                            s_msg = error_item.get("sMsg", "")
                            s_code = error_item.get("sCode", "")
                            if s_msg or s_code:
                                logger.error(f"*** SPECIFIC ERROR for {method} {request_path}: sCode={s_code}, sMsg='{s_msg}' ***")
                                error_logged = True # Mark that we found specific details
                except json.JSONDecodeError as je:
                    logger.error(f"Could not decode JSON from error response for {method} {request_path}: {je}")
                except Exception as parse_error:
                    logger.error(f"Unexpected error parsing error response JSON for {method} {request_path}: {parse_error}")

            # If we couldn't get specific details from the response body, at least log the HTTP details
            if not error_logged:
                logger.error(f"Request failed with HTTP {e.response.status_code if e.response else 'N/A'} "
                             f"for {method} {request_path}. Reason: {e.response.reason if e.response else 'N/A'}")

            if attempt < 2:
                time.sleep(5) # Wait before retrying
            else:
                logger.error(f"Final failure for {method} {request_path} after 3 attempts.")
                return error_data # Return structured error data

    # This line should ideally not be reached due to the loop logic, but added for safety
    return {"code": "1", "msg": "Request failed after all retries (internal logic error)", "data": []}

# Convenience wrappers (optional, but can make calls cleaner)
def get_public_data(request_path):
    return make_request("GET", request_path)

def get_private_data(request_path):
    return make_request("GET", request_path)

def post_private_data(request_path, data):
    return make_request("POST", request_path, data)