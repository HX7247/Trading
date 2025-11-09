"""
To run this bot, set the following environment variables in a `.env` file or your shell:
$env:APCA_API_KEY_ID="YOUR_KEY_ID"
$env:APCA_API_SECRET_KEY="YOUR_SECRET_KEY"
$env:APCA_BASE_URL="YOUR_ENDPOINT_URL"  # e.g. https://paper-api.alpaca.markets

python bot.py
"""

import os
import time
import math
import pytz
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Alpaca SDK
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

SYMBOL = os.getenv("SYMBOL", "AAPL").upper()
TIMEFRAME_STR = os.getenv("TIMEFRAME", "1Hour")
SHORT_SMA = int(os.getenv("SHORT_SMA", 50))
LONG_SMA = int(os.getenv("LONG_SMA", 200))
DOLLARS_PER_TRADE = float(os.getenv("DOLLARS_PER_TRADE", 200))
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", 900))

# Map timeframe string to alpaca TimeFrame
TF_MAP = {
    "Minute": TimeFrame.Minute,
    "5Minute": TimeFrame(5, "Minute"),
    "15Minute": TimeFrame(15, "Minute"),
    "1Hour": TimeFrame.Hour,
    "1Day": TimeFrame.Day,
}

if TIMEFRAME_STR not in TF_MAP:
    raise ValueError(f"Unsupported TIMEFRAME '{TIMEFRAME_STR}'. Use one of: {list(TF_MAP.keys())}")

TIMEFRAME = TF_MAP[TIMEFRAME_STR]

# Alpaca clients
trading = TradingClient(API_KEY, API_SECRET, paper="paper" in BASE_URL)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

TZ_NY = pytz.timezone("America/New_York")


def log(msg: str):
    now = datetime.now(TZ_NY).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{now}] {msg}", flush=True)


def get_bars(symbol: str, timeframe: TimeFrame, limit: int = 400) -> pd.DataFrame:
    """
    Fetch recent bars into a pandas DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    Pull enough history to compute the long SMA comfortably
    """
    req = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=timeframe,
    limit=limit,
    start=(datetime.now(TZ_NY) - timedelta(days=90)),
    end=datetime.now(TZ_NY),
    adjustment="raw",
    feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(req)

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
    # Simple check using account clock; avoids trading outside market hours.
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
