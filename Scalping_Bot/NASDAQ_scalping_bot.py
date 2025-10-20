"""
NASDAQ100 Momentum Scalping Bot — 1‑Minute Strategy
-------------------------------------------------

Features
- Works in two modes: BACKTEST (default) and LIVE (Alpaca paper trading stub)
- Indicators: EMA(20/50), RSI(14), MACD(12,26,9), ATR(14), rolling volume mean
- Entry (Long-only, momentum trend continuation):
  * EMA20 > EMA50 and Close > EMA20
  * MACD histogram > 0 and increasing vs prior bar
  * RSI between 60 and 85 (avoid overbought blow‑offs above 85)
  * Volume > 1.5 × rolling 20‑bar volume mean
- Risk: fixed fractional sizing or ATR‑based, hard stop and take‑profit, time stop, daily loss cap, cooldown after exits
- Backtest with 1‑minute data from yfinance; includes slippage & fees
- Live trading stub using Alpaca API (requires alpaca-py library and environment variables for credentials)
-------------------------------------------------
"""

from __future__ import annotations
import os
import math
import time
import enum
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # for backtesting data
except Exception:
    yf = None

# =============== Utilities ===============

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =============== Configuration ===============

class Mode(enum.Enum):
    BACKTEST = "BACKTEST"
    LIVE = "LIVE"

@dataclass
class StrategyConfig:
    symbols: List[str] = None  # e.g., ["QQQ"] or Nasdaq-100 list
    timeframe: str = "1m"  # backtest granularity
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    rsi_min: float = 60.0
    rsi_max: float = 85.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vol_window: int = 20
    vol_multiplier: float = 1.5
    use_atr_size: bool = False
    risk_per_trade: float = 0.003  # 0.3% of equity per trade
    sl_pct: float = 0.0025  # 0.25% stop
    tp_pct: float = 0.0040  # 0.40% take profit
    time_stop_bars: int = 5  # exit after 5 minutes if neither SL/TP hits
    cooldown_bars: int = 3
    slippage_bps: float = 1.5  # 1.5 bps ~ 0.015%
    fee_per_share: float = 0.0005  # $0.0005 per share
    daily_loss_limit_pct: float = 0.02  # stop trading for day at -2%

    def __post_init__(self):
        if self.symbols is None:
            # Default to QQQ (Nasdaq-100 ETF) as a liquid Nasdaq proxy for scalping
            self.symbols = ["QQQ"]

@dataclass
class AccountState:
    equity: float = 25_000.0
    cash: float = 25_000.0
    position_size: int = 0
    position_entry: float = 0.0
    last_exit_index: Optional[int] = None
    daily_pnl: float = 0.0
    last_trade_day: Optional[pd.Timestamp] = None

# =============== Strategy Logic ===============

class MomentumScalper:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['EMA_F'] = ema(df['Close'], self.cfg.ema_fast)
        df['EMA_S'] = ema(df['Close'], self.cfg.ema_slow)
        macd_line, signal_line, hist = macd(df['Close'], self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal)
        df['MACD'] = macd_line
        df['MACD_SIG'] = signal_line
        df['MACD_H'] = hist
        df['RSI'] = rsi(df['Close'], self.cfg.rsi_period)
        df['ATR'] = atr(df, 14)
        df['VolMA'] = df['Volume'].rolling(self.cfg.vol_window).mean()
        return df

    def entry_signal(self, df: pd.DataFrame, i: int) -> bool:
        row = df.iloc[i]
        prev = df.iloc[i-1] if i > 0 else row
        cond_trend = (row['EMA_F'] > row['EMA_S']) and (row['Close'] > row['EMA_F'])
        cond_macd = (row['MACD_H'] > 0) and (row['MACD_H'] > prev['MACD_H'])
        cond_rsi = (self.cfg.rsi_min <= row['RSI'] <= self.cfg.rsi_max)
        cond_vol = row['Volume'] > (self.cfg.vol_multiplier * (row['VolMA'] if not math.isnan(row['VolMA']) else 0))
        return bool(cond_trend and cond_macd and cond_rsi and cond_vol)

    def size_shares(self, equity: float, price: float, atr_val: float) -> int:
        if self.cfg.use_atr_size and atr_val and atr_val > 0:
            risk_dollars = equity * self.cfg.risk_per_trade
            stop_dollars = max(self.cfg.sl_pct * price, 0.6 * atr_val)  # use 0.6 ATR or pct
            shares = int(risk_dollars / stop_dollars)
        else:
            shares = int((equity * self.cfg.risk_per_trade) / (self.cfg.sl_pct * price))
        return max(shares, 0)

# =============== Backtester ===============

class Backtester:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.strategy = MomentumScalper(cfg)
        self.results_by_symbol: Dict[str, pd.DataFrame] = {}
        self.accounts: Dict[str, AccountState] = {}
        self.trades_by_symbol: Dict[str, List[Dict]] = {}

    def load_data(self, symbol: str, start_days: int = 5) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed. pip install yfinance")
        df = yf.download(symbol, period=f"{start_days}d", interval=self.cfg.timeframe, auto_adjust=False, progress=False)
        df = df.rename(columns=str.title)
        df.dropna(inplace=True)
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Use run_multi(symbols)")

    def run_multi(self, start_days: int = 5) -> Dict[str, pd.DataFrame]:
        for sym in self.cfg.symbols:
            df = self.load_data(sym, start_days=start_days)
            acc = AccountState()
            self.accounts[sym] = acc
            in_pos = False
            entry_index: Optional[int] = None
            cooldown_until: Optional[int] = None

            dfp = self.strategy.prepare(df)
            daily_open_equity = acc.equity
            acc.daily_pnl = 0.0
            acc.last_trade_day = None

            results = []
            trades: List[Dict] = []

            for i in range(len(dfp)):
                ts = dfp.index[i]
                price_open = float(dfp.iloc[i]['Open'])
                price_high = float(dfp.iloc[i]['High'])
                price_low = float(dfp.iloc[i]['Low'])
                price_close = float(dfp.iloc[i]['Close'])

                day = ts.normalize()
                if acc.last_trade_day is None or day != acc.last_trade_day:
                    daily_open_equity = acc.equity
                    acc.daily_pnl = 0.0
                    acc.last_trade_day = day

                if cooldown_until is not None and i < cooldown_until:
                    results.append(self._snapshot_symbol(dfp, i, acc, in_pos, sym))
                    continue

                if acc.equity <= daily_open_equity * (1 - self.cfg.daily_loss_limit_pct):
                    results.append(self._snapshot_symbol(dfp, i, acc, in_pos, sym, note="Daily loss limit reached"))
                    continue

                if not in_pos:
                    if i > max(self.cfg.ema_slow, self.cfg.vol_window) and self.strategy.entry_signal(dfp, i):
                        shares = self.strategy.size_shares(acc.equity, price_open, float(dfp.iloc[i]['ATR']))
                        if shares > 0:
                            fill_price = price_open * (1 + self.cfg.slippage_bps / 10000)
                            cost = shares * fill_price + shares * self.cfg.fee_per_share
                            if acc.cash >= cost:
                                acc.cash -= cost
                                acc.position_size = shares
                                acc.position_entry = fill_price
                                in_pos = True
                                entry_index = i
                                results.append(self._snapshot_symbol(dfp, i, acc, in_pos, sym, note=f"ENTER {shares}@{fill_price:.2f}"))
                                continue
                else:
                    tp = acc.position_entry * (1 + self.cfg.tp_pct)
                    sl = acc.position_entry * (1 - self.cfg.sl_pct)
                    exit_reason = None
                    exit_price = None

                    if price_high >= tp:
                        exit_price = tp * (1 - self.cfg.slippage_bps / 10000)
                        exit_reason = 'TP'
                    elif price_low <= sl:
                        exit_price = sl * (1 - self.cfg.slippage_bps / 10000)
                        exit_reason = 'SL'
                    elif entry_index is not None and (i - entry_index) >= self.cfg.time_stop_bars:
                        exit_price = price_close * (1 - self.cfg.slippage_bps / 10000)
                        exit_reason = 'TIME'

                    if exit_price is not None:
                        proceeds = acc.position_size * exit_price - acc.position_size * self.cfg.fee_per_share
                        trade_pnl = proceeds - (acc.position_size * acc.position_entry)
                        acc.cash += proceeds
                        acc.equity += trade_pnl
                        acc.daily_pnl += trade_pnl
                        trades.append({
                            'symbol': sym,
                            'time': ts,
                            'reason': exit_reason,
                            'shares': acc.position_size,
                            'entry': acc.position_entry,
                            'exit': exit_price,
                            'pnl': trade_pnl
                        })
                        in_pos = False
                        acc.position_size = 0
                        acc.position_entry = 0.0
                        entry_index = None
                        cooldown_until = i + self.cfg.cooldown_bars
                        results.append(self._snapshot_symbol(dfp, i, acc, in_pos, sym, note=f"EXIT {exit_reason} pnl={trade_pnl:.2f}"))
                        continue

                results.append(self._snapshot_symbol(dfp, i, acc, in_pos, sym))

            res_df = pd.DataFrame(results).set_index('time')
            self.results_by_symbol[sym] = res_df
            self.trades_by_symbol[sym] = trades
        return self.results_by_symbol

    def _snapshot_symbol(self, df: pd.DataFrame, i: int, acc: AccountState, in_pos: bool, sym: str, note: str = "") -> Dict:
        ts = df.index[i]
        price_close = float(df.iloc[i]['Close'])
        mtm_equity = acc.cash + (acc.position_size * price_close)
        return {
            'time': ts,
            'symbol': sym,
            'close': price_close,
            'in_pos': in_pos,
            'cash': acc.cash,
            'equity': mtm_equity,
            'note': note
        }

    def stats(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for sym, trades in self.trades_by_symbol.items():
            if not trades:
                out[sym] = {"trades": 0}
                continue
            pnl = np.array([t['pnl'] for t in trades])
            wins = (pnl > 0).sum()
            losses = (pnl <= 0).sum()
            total = len(pnl)
            avg_win = pnl[pnl > 0].mean() if wins else 0.0
            avg_loss = pnl[pnl <= 0].mean() if losses else 0.0
            expectancy = pnl.mean() if total else 0.0
            cum = pnl.cumsum()
            mdd = 0.0
            peak = 0.0
            for x in cum:
                peak = max(peak, x)
                mdd = min(mdd, x - peak)
            out[sym] = {
                'trades': total,
                'win_rate': float(wins) / total,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'expectancy': expectancy,
                'max_drawdown': mdd,
                'gross_pnl': pnl.sum()
            }
        return out

    def _snapshot(self, df: pd.DataFrame, i: int, in_pos: bool, note: str = "") -> Dict:
        ts = df.index[i]
        price_close = float(df.iloc[i]['Close'])
        # mark-to-market equity
        acc = self.account
        mtm_equity = acc.cash + (acc.position_size * price_close)
        return {
            'time': ts,
            'close': price_close,
            'in_pos': in_pos,
            'cash': acc.cash,
            'equity': mtm_equity,
            'note': note
        }

    def stats(self) -> Dict:
        if not self.trades:
            return {"trades": 0}
        pnl = np.array([t['pnl'] for t in self.trades])
        wins = (pnl > 0).sum()
        losses = (pnl <= 0).sum()
        total = len(pnl)
        avg_win = pnl[pnl > 0].mean() if wins else 0.0
        avg_loss = pnl[pnl <= 0].mean() if losses else 0.0
        expectancy = pnl.mean() if total else 0.0
        cum = pnl.cumsum()
        mdd = 0.0
        peak = 0.0
        for x in cum:
            peak = max(peak, x)
            mdd = min(mdd, x - peak)
        return {
            'trades': total,
            'win_rate': float(wins) / total,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'max_drawdown': mdd,
            'gross_pnl': pnl.sum()
        }

# =============== Live Trading (Alpaca) ===============

"""
LIVE TRADING OVERVIEW (Alpaca)
- Uses alpaca-py TradingClient + MarketData (REST poll, 1-minute cadence)
- Per-symbol state machine (one position per symbol)
- Order type: MARKET entry with attached OCO bracket (TP/SL)
- Risk controls: per-trade risk, daily loss cap, cooldown, time-stop (market exit)
- IMPORTANT: Set credentials via environment variables. Never hard-code secrets.

Env Vars
    ALPACA_KEY_ID
    ALPACA_SECRET_KEY
    ALPACA_BASE_URL (e.g., https://paper-api.alpaca.markets)
    DATA_BASE_URL (optional; defaults to https://data.alpaca.markets)
"""

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:
    TradingClient = None

@dataclass
class LiveSymbolState:
    account: AccountState
    in_pos: bool = False
    entry_time: Optional[pd.Timestamp] = None
    cooldown_until: Optional[pd.Timestamp] = None
    last_bar_time: Optional[pd.Timestamp] = None
    position_qty: int = 0
    position_avg: float = 0.0

class LiveAlpaca:
    def __init__(self, cfg: StrategyConfig):
        if TradingClient is None:
            raise RuntimeError("alpaca-py is not installed. pip install alpaca-py")
        self.cfg = cfg
        self.strategy = MomentumScalper(cfg)
        self.states: Dict[str, LiveSymbolState] = {sym: LiveSymbolState(AccountState()) for sym in cfg.symbols}
        key = os.getenv('ALPACA_KEY_ID')
        secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        data_url = os.getenv('DATA_BASE_URL', 'https://data.alpaca.markets')
        if not key or not secret:
            raise RuntimeError("Missing ALPACA_KEY_ID/ALPACA_SECRET_KEY environment variables.")
        self.trading = TradingClient(api_key=key, secret_key=secret, paper=('paper' in base_url))
        self.data = StockHistoricalDataClient(api_key=key, secret_key=secret)
        self.buffers: Dict[str, pd.DataFrame] = {sym: pd.DataFrame(columns=['Open','High','Low','Close','Volume']) for sym in cfg.symbols}

    def fetch_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        now = pd.Timestamp.utcnow().floor('min')
        start = now - pd.Timedelta(minutes=5)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start.to_pydatetime(), end=now.to_pydatetime(), limit=5)
        bars = self.data.get_stock_bars(req)
        if symbol not in bars.data or not bars.data[symbol]:
            return None
        b = bars.data[symbol][-1]
        return pd.Series({'Open': float(b.open), 'High': float(b.high), 'Low': float(b.low), 'Close': float(b.close), 'Volume': int(b.volume)}, name=pd.to_datetime(b.timestamp))

    def place_entry(self, symbol: str, qty: int, tp_price: float, sl_price: float):
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            take_profit={'limit_price': round(tp_price, 2)},
            stop_loss={'stop_price': round(sl_price, 2)}
        )
        order = self.trading.submit_order(req)
        return order

    def close_position(self, symbol: str):
        try:
            self.trading.close_position(symbol)
        except Exception:
            pass

    def run(self):
        print(f"LIVE trading for {self.cfg.symbols} on 1-minute bars.")
        while True:
            now = pd.Timestamp.utcnow()
            for sym in self.cfg.symbols:
                state = self.states[sym]
                # Cooldown check
                if state.cooldown_until and now < state.cooldown_until:
                    continue

                bar = self.fetch_latest_bar(sym)
                if bar is None:
                    continue
                if state.last_bar_time is not None and bar.name <= state.last_bar_time:
                    continue
                state.last_bar_time = bar.name

                # Update buffer and indicators
                buf = self.buffers[sym]
                buf.loc[bar.name] = bar
                buf = buf.sort_index().tail(1000)
                self.buffers[sym] = buf
                dfp = self.strategy.prepare(buf)
                i = len(dfp) - 1

                # New day reset
                day = bar.name.normalize()
                acc = state.account
                if acc.last_trade_day is None or day != acc.last_trade_day:
                    acc.daily_pnl = 0.0
                    acc.last_trade_day = day

                # Daily loss cap
                if acc.daily_pnl <= -acc.equity * self.cfg.daily_loss_limit_pct:
                    continue

                if not state.in_pos:
                    if i > max(self.cfg.ema_slow, self.cfg.vol_window) and self.strategy.entry_signal(dfp, i):
                        price = float(dfp.iloc[i]['Close'])
                        shares = self.strategy.size_shares(acc.equity, price, float(dfp.iloc[i]['ATR']))
                        if shares <= 0:
                            continue
                        tp = price * (1 + self.cfg.tp_pct)
                        sl = price * (1 - self.cfg.sl_pct)
                        try:
                            order = self.place_entry(sym, shares, tp, sl)
                            state.in_pos = True
                            state.entry_time = bar.name
                            state.position_qty = shares
                            state.position_avg = price
                            print(f"{bar.name} ENTER {sym} {shares}@{price:.2f} TP {tp:.2f} SL {sl:.2f}")
                        except Exception as e:
                            print(f"Order error {sym}: {e}")
                else:
                    # Time-stop management — close after N bars if neither TP/SL hit
                    if state.entry_time and (bar.name - state.entry_time) >= pd.Timedelta(minutes=self.cfg.time_stop_bars):
                        try:
                            self.close_position(sym)
                            state.in_pos = False
                            state.position_qty = 0
                            state.position_avg = 0.0
                            state.entry_time = None
                            state.cooldown_until = bar.name + pd.Timedelta(minutes=self.cfg.cooldown_bars)
                            print(f"{bar.name} TIME EXIT {sym}")
                        except Exception as e:
                            print(f"Exit error {sym}: {e}")
            # align to next minute
            sleep_sec = max(1, 60 - (time.time() % 60))
            time.sleep(sleep_sec)

# =============== Nasdaq-100 Universe ===============


NASDAQ_100_LIST = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
    "NFLX","PEP","ADBE","CSCO","AMD","INTC","TXN","QCOM","AMAT","BKNG",
    "LIN","SBUX","ISRG","INTU","PDD","GILD","REGN","VRTX","MU","ABNB",
    "CHTR","LRCX","MAR","PANW","CRWD","ADP","MDLZ","PYPL","MELI","KLAC",
    "CSX","ADSK","MRVL","AMGN","AEP","HON","ORLY","AZN","KDP","KHC",
    "SNPS","CDNS","NXPI","ODFL","FTNT","IDXX","MNST","ROP","PAYX","CTAS",
    "PCAR","WDAY","CEG","CTSH","ADI","EXC","TEAM","DDOG","ZS","SPLK",
    "ABNB","ANSS","FAST","DXCM","CHTR","CPRT","EA","LULU","ROST","DLTR",
    "VRSK","CSGP","GEHC","ON","LCID","RIVN","MRNA","BIDU","JD","KLAY",
    "FANG","WBD","PDD","ZM","DOCU","OKTA","MDB","BKR","TTD","VRSN"
]

# =============== CLI ===============

def backtest_main():
    # Parse symbols
    env_syms = os.getenv('SYMBOLS')
    use_ndx = os.getenv('USE_NDX', '0')
    if env_syms:
        symbols = [s.strip().upper() for s in env_syms.split(',') if s.strip()]
    elif use_ndx == '1':
        symbols = NASDAQ_100_LIST
    else:
        symbols = [os.getenv('SYMBOL', 'QQQ').upper()]

    cfg = StrategyConfig(
        symbols=symbols,
        timeframe=os.getenv('TIMEFRAME', '1m'),
        rsi_min=float(os.getenv('RSI_MIN', 60)),
        rsi_max=float(os.getenv('RSI_MAX', 85)),
        tp_pct=float(os.getenv('TP_PCT', 0.0040)),
        sl_pct=float(os.getenv('SL_PCT', 0.0025)),
    )
    bt = Backtester(cfg)
    print(f"Downloading {symbols} {cfg.timeframe} data…")
    res_map = bt.run_multi(start_days=int(os.getenv('DAYS', '5')))
    print("Running backtests per symbol… done.")
    stats_map = bt.stats()
    print("
=== RESULTS BY SYMBOL ===")
    for sym, s in stats_map.items():
        print(f"
[{sym}]")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"{k:>15}: {v:.4f}")
            else:
                print(f"{k:>15}: {v}")

    out_dir = os.getenv('OUT', '.')
    os.makedirs(out_dir, exist_ok=True)
    # Save trades and equity per symbol
    for sym, trades in bt.trades_by_symbol.items():
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_path = os.path.join(out_dir, f"trades_{sym}_{cfg.timeframe}.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"Saved trades -> {trades_path}")
        eq_path = os.path.join(out_dir, f"equity_{sym}_{cfg.timeframe}.csv")
        res_map[sym][['equity']].to_csv(eq_path)
        print(f"Saved equity curve -> {eq_path}")

if __name__ == "__main__":
    mode = os.getenv('MODE', 'BACKTEST').upper()
    if mode == 'BACKTEST':
        backtest_main()
    else:
        print("LIVE mode stub. Integrate broker API (e.g., alpaca-py) before use.")

    mode = os.getenv('MODE', 'BACKTEST').upper()
    if mode == 'BACKTEST':
        backtest_main()
    else:
        print("LIVE mode stub. Integrate broker API (e.g., alpaca-py) before use.")
