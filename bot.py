"""
To run this bot, set the following environment variables in a `.env` file or your shell:
$env:APCA_API_KEY_ID="YOUR_KEY_ID"
$env:APCA_API_SECRET_KEY="YOUR_SECRET_KEY"
$env:APCA_BASE_URL="YOUR_ENDPOINT_URL" e.g. https://paper-api.alpaca.markets

python bot.py
"""

import os
import time
import math
import pytz
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from dotenv import load_dotenv
load_dotenv()  # automatically finds .env in current working directory

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")

# basic validation
if not API_KEY or not API_SECRET:
    raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in environment or .env")

# trim accidental whitespace/newlines
API_KEY = API_KEY.strip()
API_SECRET = API_SECRET.strip()
BASE_URL = BASE_URL.strip()

# normalize BASE_URL: remove trailing '/v2' or trailing slash so SDK builds correct endpoint
if BASE_URL.endswith("/v2"):
    BASE_URL = BASE_URL[:-3]
BASE_URL = BASE_URL.rstrip("/")

SYMBOL = os.getenv("SYMBOL", "AAPL").upper()

# Force 5-minute timeframe regardless of environment variable
TIMEFRAME_STR = "5Minute"

# Build TF_MAP robustly for different alpaca-py versions
try:
    # preferred: TimeFrame exposes constants / supports multiplication
    TF_MAP = {
        "Minute": TimeFrame.Minute,
        "5Minute": TimeFrame.Minute * 5,
        "15Minute": TimeFrame.Minute * 15,
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
except Exception:
    # try to use TimeFrameUnit enum if available
    try:
        from alpaca.data.timeframe import TimeFrameUnit
        TF_MAP = {
            "Minute": TimeFrame(1, TimeFrameUnit.Minute),
            "5Minute": TimeFrame(5, TimeFrameUnit.Minute),
            "15Minute": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
    except Exception:
        # fallback using unit strings (last resort)
        TF_MAP = {
            "Minute": TimeFrame(1, "minute"),
            "5Minute": TimeFrame(5, "minute"),
            "15Minute": TimeFrame(15, "minute"),
            "1Hour": TimeFrame(1, "hour"),
            "1Day": TimeFrame(1, "day"),
        }

if TIMEFRAME_STR not in TF_MAP:
    raise ValueError(f"Unsupported TIMEFRAME '{TIMEFRAME_STR}'. Use one of: {list(TF_MAP.keys())}")

TIMEFRAME = TF_MAP[TIMEFRAME_STR]

# helper: ensure the TimeFrame unit is an enum-like object (avoid passing plain str unit)
def _ensure_timeframe_enum(tf: TimeFrame) -> TimeFrame:
    unit = getattr(tf, "unit", None)
    # if unit already has 'value' it's likely an enum -> ok
    if hasattr(unit, "value"):
        return tf
    # attempt to map common unit strings to TimeFrame constants / TimeFrameUnit
    unit_str = str(unit).lower() if unit is not None else ""
    try:
        # prefer TimeFrame constants if available
        if unit_str.startswith("minute"):
            base = getattr(TimeFrame, "Minute", None)
        elif unit_str.startswith("hour"):
            base = getattr(TimeFrame, "Hour", None)
        elif unit_str.startswith("day"):
            base = getattr(TimeFrame, "Day", None)
        else:
            base = None
        if base is not None:
            return TimeFrame(tf.n, base)
        # try TimeFrameUnit enum if present
        from alpaca.data.timeframe import TimeFrameUnit
        if unit_str.startswith("minute"):
            return TimeFrame(tf.n, TimeFrameUnit.Minute)
        if unit_str.startswith("hour"):
            return TimeFrame(tf.n, TimeFrameUnit.Hour)
        if unit_str.startswith("day"):
            return TimeFrame(tf.n, TimeFrameUnit.Day)
    except Exception:
        pass
    # as a last resort return original tf (may still fail inside client)
    return tf


# --- strategy / runtime parameters (provide sensible defaults via env) ---
SHORT_SMA = int(os.getenv("SHORT_SMA", 50))
LONG_SMA = int(os.getenv("LONG_SMA", 200))
DOLLARS_PER_TRADE = float(os.getenv("DOLLARS_PER_TRADE", 200.0))
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", 300))  # default 5 minutes


# Alpaca clients — some alpaca-py versions use `paper=` not `base_url`
IS_PAPER = "paper" in BASE_URL.lower()

# Try constructing TradingClient / data client robustly across alpaca-py versions
trading = None
data_client = None
_last_exc = None

# 1) prefer constructors that accept base_url (newer versions)
try:
    trading = TradingClient(API_KEY, API_SECRET, base_url=BASE_URL)
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET, base_url=BASE_URL)
except TypeError as e:
    _last_exc = e

# 2) fallback to 'paper=' constructor
if trading is None:
    try:
        trading = TradingClient(API_KEY, API_SECRET, paper=IS_PAPER)
    except TypeError as e:
        _last_exc = e

# 3) final fallback: no extra kwargs
if trading is None:
    try:
        trading = TradingClient(API_KEY, API_SECRET)
    except Exception as e:
        _last_exc = e

# data client fallbacks
if data_client is None:
    try:
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    except Exception as e:
        _last_exc = e

if trading is None or data_client is None:
    raise RuntimeError(f"Failed to construct Alpaca clients. Last error: {_last_exc!r}")

# quick auth check: call get_account and surface helpful message on auth failure
try:
    acct = trading.get_account()
except Exception as e:
    # mask keys when printing
    def mask(k): return (k[:4] + "..." + k[-4:]) if k and len(k) > 8 else "<redacted>"
    raise RuntimeError(
        "Alpaca authentication failed. Check API key/secret and BASE_URL.\n"
        f"Key: {mask(API_KEY)} Secret: {mask(API_SECRET)} BASE_URL: {BASE_URL}\n"
        f"Underlying error: {e}"
    ) from e

TZ_NY = pytz.timezone("America/New_York")


def log(msg: str):
    now = datetime.now(TZ_NY).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{now}] {msg}", flush=True)


def get_bars(symbol: str, timeframe: TimeFrame, limit: int = 400) -> pd.DataFrame:
    """
    Fetch recent bars into a pandas DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    Pull enough history to compute the long SMA comfortably
    """
    # ensure timeframe is a TimeFrame instance (sometimes environment or callers pass a str)
    if isinstance(timeframe, str):
        if timeframe in TF_MAP:
            tf = TF_MAP[timeframe]
        else:
            raise ValueError(f"Unknown timeframe string '{timeframe}'")
    else:
        tf = timeframe

    # normalize to TimeFrame with enum-like unit to avoid "'str' object has no attribute 'value'" inside alpaca client
    tf = _ensure_timeframe_enum(tf)

    # debug: surface types to help identify 'str'. Remove or comment out if noisy.
    log(f"get_bars: symbol={symbol} timeframe_type={type(tf)} limit={limit}")

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        limit=limit,
        start=(datetime.now(TZ_NY) - timedelta(days=90)),
        end=datetime.now(TZ_NY),
        adjustment="raw",
        feed=DataFeed.IEX,
    )

    try:
        bars = data_client.get_stock_bars(req)
    except Exception as e:
        # Re-raise with contextual info to diagnose "'str' object has no attribute 'value'" errors
        raise RuntimeError(f"failed to fetch bars (symbol={symbol}, timeframe={tf!r}): {e}") from e

    if symbol not in bars.data:
        raise RuntimeError(f"No bar data returned for {symbol}")

    rows = []
    for bar in bars.data[symbol]:
        rows.append({
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def compute_signals(df: pd.DataFrame, short_n: int, long_n: int):
    """
    Compute SMA cross signals. Returns 'BUY', 'SELL', or None for hold.
    Signal triggers *only when a crossover happens* on the most recent bar.
    """
    if len(df) < max(short_n, long_n) + 2:
        return None

    df["sma_short"] = df["close"].rolling(short_n).mean()
    df["sma_long"] = df["close"].rolling(long_n).mean()

    # Use last two points to detect fresh cross
    s_prev = df["sma_short"].iloc[-2]
    l_prev = df["sma_long"].iloc[-2]
    s_now = df["sma_short"].iloc[-1]
    l_now = df["sma_long"].iloc[-1]

    # Require both SMAs to be valid
    if any(math.isnan(x) for x in [s_prev, l_prev, s_now, l_now]):
        return None

    crossed_up = s_prev <= l_prev and s_now > l_now
    crossed_down = s_prev >= l_prev and s_now < l_now

    if crossed_up:
        return "BUY"
    if crossed_down:
        return "SELL"
    return None


def get_position_qty(symbol: str) -> int:
    # Current position quantity (signed). 0 if flat.
    try:
        pos = trading.get_open_position(symbol)
        return int(float(pos.qty))  # qty can be a string
    except Exception:
        return 0


def place_market_order(symbol: str, side: str, notional_usd: float = None, qty: int = None):

    # Send a market order using notional (dollar-based) or quantity.
 
    if notional_usd is None and qty is None:
        raise ValueError("Provide notional_usd or qty")

    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    req = None
    if notional_usd is not None:
        # Dollar notional order (fractional shares allowed on Alpaca)
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional_usd, 2),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
    else:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=abs(int(qty)),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

    order = trading.submit_order(req)
    return order


def sync_once():
    log(f"Fetching bars for {SYMBOL} @ {TIMEFRAME_STR} ...")
    df = get_bars(SYMBOL, TIMEFRAME)
    px = df["close"].iloc[-1]
    signal = compute_signals(df, SHORT_SMA, LONG_SMA)
    pos_qty = get_position_qty(SYMBOL)

    log(f"Last close: {px:.2f} | SMA{SHORT_SMA}={df['sma_short'].iloc[-1]:.2f} "
        f"| SMA{LONG_SMA}={df['sma_long'].iloc[-1]:.2f} | Position={pos_qty} shares")

    if signal == "BUY":
        if pos_qty <= 0:
            log(f"Signal=BUY → placing market buy for ~${DOLLARS_PER_TRADE:.2f}")
            order = place_market_order(SYMBOL, "buy", notional_usd=DOLLARS_PER_TRADE)
            log(f"Buy submitted: id={order.id} status={order.status}")
        else:
            log("Signal=BUY but already long; holding.")
    elif signal == "SELL":
        if pos_qty > 0:
            log(f"Signal=SELL → closing long position ({pos_qty} shares)")
            order = place_market_order(SYMBOL, "sell", qty=pos_qty)
            log(f"Sell submitted: id={order.id} status={order.status}")
        else:
            log("Signal=SELL but not long; holding.")
    else:
        log("No new signal; holding.")


def market_is_open_now() -> bool:
    # Simple check using account clock; avoids trading outside NYSE market hours.
    clock = trading.get_clock()
    return bool(clock.is_open)


def main_loop():
    log("Starting SMA crossover bot (paper) ...")
    log(f"Symbol={SYMBOL} | TF={TIMEFRAME_STR} | ShortSMA={SHORT_SMA} | LongSMA={LONG_SMA}")
    while True:
        try:
            if market_is_open_now():
                sync_once()
            else:
                log("Market closed; skipping.")

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    # Quick sanity check for credentials
    acct = trading.get_account()
    mode = "PAPER" if "paper" in BASE_URL else "LIVE"
    log(f"Connected to Alpaca ({mode}). Equity: ${acct.equity}, Buying power: ${acct.buying_power}")
    main_loop()
