# bot/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- OKX API Credentials ---
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")
OKX_BASE_URL = "https://www.okx.com"

# --- Trading Parameters ---
INSTRUMENT = "KAITO-USDT-SWAP" # Change this for different pairs
LEVERAGE = 15
TARGET_MARGIN_USD = 1.0       # Key for small balance trading
MAX_POSITION_SIZE_USDT = 10000

# --- Risk Management ---
TP1_PNL_PERCENT = 15.0         # 15% PNL for TP1
TP2_PNL_PERCENT = 21.0         # 21% PNL for TP2
TP1_SIZE_RATIO = 0.71       # 71% of position for TP1
SL_PNL_PERCENT = -15.0         # -21% PNL for SL

# --- Market Filters ---
MIN_24H_VOLUME_USDT = 1000000
MAX_SPREAD_PERCENT = 0.1      # 0.1%
MIN_VOLATILITY_PERCENT = 0.2  # 0.2%

# --- Strategy Parameters ---
MA_FAST_PERIOD = 13
MA_SLOW_PERIOD = 21
RSI_STOCH_PERIOD = 5
STOCH_K_PERIOD = 3
STOCH_D_PERIOD = 3
ATR_PERIOD = 14
ADX_PERIOD = 14
ADX_THRESHOLD = 7
VOLUME_THRESHOLD_MULTIPLIER = 0.2 # Volume vs EMA(5)

# --- Execution & Timing ---
POLLING_INTERVAL_SECONDS = 15
SIGNAL_COOLDOWN_SECONDS = 60
API_RATE_LIMIT_DELAY = 0.1

# --- Logging ---
LOG_LEVEL = "INFO" # Can be DEBUG for more details
LOG_FILE = "trading_bot.log"