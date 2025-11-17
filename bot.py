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

# List of symbols to trade
SYMBOLS = ["NVDA", "TSLA", "META", "PLTR"]

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
SHORT_EMA = 9
LONG_EMA = 21
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
    Pull enough history to compute the EMA comfortably
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


def compute_signals(df: pd.DataFrame, short_ema: int, long_ema: int):
    """
    Compute EMA cross signals. Returns 'BUY', 'SELL', or None for hold.
    Signal triggers *only when a crossover happens* on the most recent bar.
    """
    if len(df) < max(short_ema, long_ema) + 2:
        return None

    df["ema_short"] = df["close"].ewm(span=short_ema, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=long_ema, adjust=False).mean()

    # Use last two points to detect fresh cross
    e_short_prev = df["ema_short"].iloc[-2]
    e_long_prev = df["ema_long"].iloc[-2]
    e_short_now = df["ema_short"].iloc[-1]
    e_long_now = df["ema_long"].iloc[-1]

    # Require both EMAs to be valid
    if any(math.isnan(x) for x in [e_short_prev, e_long_prev, e_short_now, e_long_now]):
        return None

    crossed_up = e_short_prev <= e_long_prev and e_short_now > e_long_now
    crossed_down = e_short_prev >= e_long_prev and e_short_now < e_long_now

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


def sync_once_symbol(symbol: str):
    """Process a single symbol for trading signals"""
    try:
        log(f"Fetching bars for {symbol} @ {TIMEFRAME_STR} ...")
        df = get_bars(symbol, TIMEFRAME)
        px = df["close"].iloc[-1]
        signal = compute_signals(df, SHORT_EMA, LONG_EMA)
        pos_qty = get_position_qty(symbol)

        log(f"{symbol} | Last close: {px:.2f} | EMA{SHORT_EMA}={df['ema_short'].iloc[-1]:.2f} "
            f"| EMA{LONG_EMA}={df['ema_long'].iloc[-1]:.2f} | Position={pos_qty} shares")

        if signal == "BUY":
            if pos_qty <= 0:
                log(f"{symbol}: Signal=BUY (9 EMA crossed above 21 EMA) → placing market buy for ~${DOLLARS_PER_TRADE:.2f}")
                order = place_market_order(symbol, "buy", notional_usd=DOLLARS_PER_TRADE)
                log(f"{symbol}: Buy submitted: id={order.id} status={order.status}")
            else:
                log(f"{symbol}: Signal=BUY but already long; holding.")
        elif signal == "SELL":
            if pos_qty > 0:
                log(f"{symbol}: Signal=SELL (9 EMA crossed below 21 EMA) → closing long position ({pos_qty} shares)")
                order = place_market_order(symbol, "sell", qty=pos_qty)
                log(f"{symbol}: Sell submitted: id={order.id} status={order.status}")
            else:
                log(f"{symbol}: Signal=SELL but not long; holding.")
        else:
            log(f"{symbol}: No new signal; holding.")
    except Exception as e:
        log(f"{symbol}: ERROR: {e}")


def sync_once():
    """Process all symbols"""
    for symbol in SYMBOLS:
        sync_once_symbol(symbol)


def market_is_open_now() -> bool:
    # Simple check using account clock; avoids trading outside market hours.
    clock = trading.get_clock()
    return bool(clock.is_open)


def main_loop():
    log("Starting 9/21 EMA crossover bot (paper) ...")
    log(f"Symbols={', '.join(SYMBOLS)} | TF={TIMEFRAME_STR} | ShortEMA={SHORT_EMA} | LongEMA={LONG_EMA}")
    while True:
        try:
            if market_is_open_now():
                sync_once()
            else:
                log("Market closed; skipping.")

        except Exception as e:
            log(f"FATAL ERROR: {e}")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    # Quick sanity check for credentials
    acct = trading.get_account()
    mode = "PAPER" if "paper" in BASE_URL else "LIVE"
    log(f"Connected to Alpaca ({mode}). Equity: ${acct.equity}, Buying power: ${acct.buying_power}")
    main_loop()
