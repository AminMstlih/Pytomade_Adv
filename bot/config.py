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
INSTRUMENT = "ETH-USDT-SWAP" # Change this for different pairs
LEVERAGE = 51
TARGET_MARGIN_USD = 0.5       # Key for small balance trading
MAX_POSITION_SIZE_USDT = 10000

# Trade mode: "hedge" to keep opposite positions, "close" to close them before new trades
TRADE_MODE = "close"  # or "hedge"

# --- Risk Management ---
TP1_PNL_PERCENT = 150.0         # 15% PNL for TP1
TP2_PNL_PERCENT = 210.0         # 27% PNL for TP2
TP1_SIZE_RATIO = 0.75           # 50% of position size for TP1       
SL_PNL_PERCENT = -15.0         # -21% PNL for SL

# --- Trailing Stop Parameters ---
USE_TRAILING_STOP_WITH_FIXED_SL = False               
TRAILING_STOP_ACTIVATION_PNL_PERCENT = 0.15           # Aktifkan trailing stop setelah posisi mencapai profit ini (%)
TRAILING_STOP_CALLBACK_RATIO_PERCENT = 0.007          # 5% pnl Jarak trailing stop dari harga tertinggi/terendah (%)
TRAILING_STOP_CALLBACK_TYPE = "percent"               # Tipe callback: 'percent' atau 'constant' (kita gunakan 'percent')
TRAILING_STOP_CHECK_INTERVAL_SECONDS = 60             # Interval (detik) untuk memeriksa dan memasang trailing stop (jika diperlukan)

# --- Market Filters ---
MIN_24H_VOLUME_USDT = 1000000
MAX_SPREAD_PERCENT = 0.2      # 0.1%
MIN_VOLATILITY_PERCENT = 0.2  # 0.2%

# --- Strategy Parameters ---
MA_FAST_PERIOD = 21
MA_SLOW_PERIOD = 22
RSI_STOCH_PERIOD = 5
STOCH_K_PERIOD = 3
STOCH_D_PERIOD = 3
ATR_PERIOD = 14
ADX_PERIOD = 14
ADX_THRESHOLD = 21
VOLUME_THRESHOLD_MULTIPLIER = 1.0 # Volume vs EMA(5)

# --- Execution & Timing ---
POLLING_INTERVAL_SECONDS = 10
SIGNAL_COOLDOWN_SECONDS = 60
API_RATE_LIMIT_DELAY = 0.5

# --- Logging ---
LOG_LEVEL = "INFO" # Can be DEBUG for more details
LOG_FILE = "trading_bot.log"