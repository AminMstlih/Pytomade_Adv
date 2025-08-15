# exchange/okx_client.py
import time
import hmac
import base64
import json
import requests
from bot.config import OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, OKX_BASE_URL, API_RATE_LIMIT_DELAY
from bot.logger_config import logger
import sys

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
    message = timestamp + method.upper() + request_path + body
    mac = hmac.new(
        bytes(OKX_SECRET_KEY, encoding='utf8'),
        bytes(message, encoding='utf-8'),
        digestmod='sha256'
    )
    d = mac.digest()
    return base64.b64encode(d).decode()

def make_request(method, request_path, data=None):
    """
    Makes a request to the OKX API.
    Handles both public (GET) and private (GET/POST) requests.
    """
    url = OKX_BASE_URL + request_path
    body = json.dumps(data) if data else ''
    is_private = method.upper() in ['GET', 'POST'] and any(
        prefix in request_path for prefix in ['/api/v5/account', '/api/v5/trade']
    )
    
    headers = {
        'Content-Type': 'application/json',
        'x-simulated-trading': '0' # 0 for real market
    }
    
    if is_private:
        timestamp = get_server_time()
        headers['OK-ACCESS-KEY'] = OKX_API_KEY
        headers['OK-ACCESS-SIGN'] = _generate_signature(timestamp, method, request_path, body)
        headers['OK-ACCESS-TIMESTAMP'] = timestamp
        headers['OK-ACCESS-PASSPHRASE'] = OKX_PASSPHRASE
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
            if attempt < 2:
                time.sleep(5) # Wait before retrying
            else:
                logger.error(f"Final failure for {method} {request_path} after 3 attempts.")
                return {"code": "1", "msg": f"Network/Request error: {str(e)}", "data": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response for {method} {request_path} (attempt {attempt+1}/3): {e}")
            logger.error(f"Response text: {response.text}")
            if attempt < 2:
                time.sleep(5)
            else:
                return {"code": "1", "msg": f"JSON decode error: {str(e)}", "data": []}
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error during {method} {request_path} (attempt {attempt+1}/3): {e}")
             if attempt < 2:
                time.sleep(5)
             else:
                return {"code": "1", "msg": f"Unexpected error: {str(e)}", "data": []}

    # This line should ideally not be reached due to the loop logic, but added for safety
    return {"code": "1", "msg": "Request failed after all retries (internal logic error)", "data": []}

# Convenience wrappers (optional, but can make calls cleaner)
def get_public_data(request_path):
    return make_request("GET", request_path)

def get_private_data(request_path):
    return make_request("GET", request_path)

def post_private_data(request_path, data):
    return make_request("POST", request_path, data)