"""
Momentum Scalping Bot (1‑Minute)

Education-only example showing:
- Backtesting on 1‑minute bars (yfinance)
- Live paper trading (Alpaca Markets)
- Momentum filters: EMA(20/50), RSI(14), MACD(12,26,9), recent‑high breakout
- Risk mgmt: fixed % risk, ATR stop, trailing stop, cooldown, daily loss cap

Requirements (install):
    pip install pandas numpy yfinance python-dotenv alpaca-trade-api

IMPORTANT:
- Use at your own risk. Markets involve risk. This is not financial advice.
- 1m data from yfinance is only available for recent days; for older intraday
  testing you will need a different data source.

Quickstart (Backtest):
    python momentum_scalper.py --mode backtest --symbol AAPL --days 5

Quickstart (Live Paper on Alpaca):
    # Set env vars in a .env file in the same folder:
    # ALPACA_API_KEY_ID=...
    # ALPACA_API_SECRET_KEY=...
    # ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
    python momentum_scalper.py --mode live --symbol AAPL
"""
from __future__ import annotations
import os
import time
import math
import argparse
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

# Alpaca is optional for backtests
try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

# -------------------------
# Indicator Implementations
# -------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean().fillna(method='bfill')

# -------------------------
# Config
# -------------------------

@dataclass
class StrategyConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    rsi_min: float = 55.0  # momentum confirmation
    rsi_max: float = 85.0  # avoid extreme overbought entries
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    breakout_lookback: int = 20  # bars to define recent high/low
    atr_len: int = 14
    atr_mult_sl: float = 1.2     # initial stop distance
    atr_mult_tp: float = 2.0     # take profit distance
    trail_atr_mult: float = 1.0  # trailing stop once in profit
    cooldown_minutes: int = 3
    max_daily_loss_pct: float = 2.0  # as % of start equity

@dataclass
class ExecutionConfig:
    starting_equity: float = 25_000.0
    risk_per_trade_pct: float = 0.25  # of equity
    max_position_pct: float = 33.0    # cap position size as % of equity
    slippage_bps: float = 1.0         # 1 bp = 0.01%
    commission_per_order: float = 0.0

# -------------------------
# Data Utilities
# -------------------------

class DataProvider:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def fetch_yfinance_1m(self, days: int = 5) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")
        df = yf.download(self.symbol, period=f"{days}d", interval="1m", auto_adjust=False, prepost=False)
        if df.empty:
            raise ValueError("Received empty dataframe from yfinance. Check symbol or interval availability.")
        df.index = df.index.tz_localize(None)
        df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
        return df

# -------------------------
# Strategy
# -------------------------

class MomentumScalper:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.last_trade_time: Optional[pd.Timestamp] = None
        self.daily_pnl: Dict[str, float] = {}

    def _daily_key(self, ts: pd.Timestamp) -> str:
        return ts.date().isoformat()

    def _in_cooldown(self, ts: pd.Timestamp) -> bool:
        if self.last_trade_time is None:
            return False
        return (ts - self.last_trade_time) < pd.Timedelta(minutes=self.cfg.cooldown_minutes)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['EMA_F'] = ema(out['Close'], self.cfg.ema_fast)
        out['EMA_S'] = ema(out['Close'], self.cfg.ema_slow)
        out['RSI']   = rsi(out['Close'], self.cfg.rsi_len)
        macd_line, signal, hist = macd(out['Close'], self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal)
        out['MACD']  = macd_line
        out['MACDS'] = signal
        out['MACDH'] = hist
        out['ATR']   = atr(out, self.cfg.atr_len)
        out['HighN'] = out['High'].rolling(self.cfg.breakout_lookback).max()
        out['LowN']  = out['Low'].rolling(self.cfg.breakout_lookback).min()
        return out

    def entry_signal_long(self, row: pd.Series, prev_row: pd.Series) -> bool:
        # Trend & momentum filters
        ema_trend = row['EMA_F'] > row['EMA_S'] and row['EMA_F'] > prev_row['EMA_F'] and row['EMA_S'] >= prev_row['EMA_S']
        rsi_ok = self.cfg.rsi_min <= row['RSI'] <= self.cfg.rsi_max
        macd_up = row['MACDH'] > 0 and row['MACDH'] > prev_row['MACDH']
        breakout = row['Close'] >= (row['HighN'] - 1e-9)
        return bool(ema_trend and rsi_ok and macd_up and breakout)

    def exit_rules(self, position: Dict, row: pd.Series) -> Dict:
        # Update trailing stop
        if position['side'] == 'long':
            trail = row['Close'] - self.cfg.trail_atr_mult * row['ATR']
            position['trail_stop'] = max(position.get('trail_stop', -math.inf), trail)
            # Hard SL/TP
            if row['Low'] <= position['stop']:
                position['exit_reason'] = 'stop'
            elif row['High'] >= position['take']:
                position['exit_reason'] = 'take'
            elif row['Close'] <= position['trail_stop']:
                position['exit_reason'] = 'trail'
        return position

    def can_trade_today(self, ts: pd.Timestamp, equity: float) -> bool:
        key = self._daily_key(ts)
        if key not in self.daily_pnl:
            self.daily_pnl[key] = 0.0
        # cap daily loss
        loss_cap = - (self.cfg.max_daily_loss_pct / 100.0) * equity
        return self.daily_pnl[key] > loss_cap

    def register_pnl(self, ts: pd.Timestamp, pnl: float):
        key = self._daily_key(ts)
        self.daily_pnl[key] = self.daily_pnl.get(key, 0.0) + pnl

# -------------------------
# Backtester
# -------------------------

class Backtester:
    def __init__(self, strat: MomentumScalper, exe: ExecutionConfig, symbol: str):
        self.strat = strat
        self.exe = exe
        self.symbol = symbol
        self.trades: List[Dict] = []

    def position_size(self, equity: float, atr_value: float, price: float) -> int:
        risk_amt = equity * (self.exe.risk_per_trade_pct / 100.0)
        stop_dist = max(atr_value * self.strat.cfg.atr_mult_sl, 0.01)
        shares = int(risk_amt / stop_dist)
        # cap by max position % of equity
        max_shares_by_equity = int((equity * (self.exe.max_position_pct/100.0)) / price)
        return max(0, min(shares, max_shares_by_equity))

    def run(self, df: pd.DataFrame) -> Dict:
        df = self.strat.compute_indicators(df)
        equity = self.exe.starting_equity
        position: Optional[Dict] = None
        last_day = None

        for i in range(max(60, self.strat.cfg.breakout_lookback)+1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            ts: pd.Timestamp = row.name
            day = ts.date()

            if last_day is None or day != last_day:
                last_day = day

            # Flatten at session end (optional; assume 20:00 ET / 01:00 UTC for US stocks)
            session_end = (ts.hour == 20 and ts.minute == 0)

            # Manage open position
            if position is not None:
                position = self.strat.exit_rules(position, row)
                if 'exit_reason' in position or session_end:
                    # Execute exit at close
                    exit_px = row['Close'] * (1 - self.exe.slippage_bps/10000.0)
                    pnl = (exit_px - position['entry']) * position['qty'] - self.exe.commission_per_order
                    equity += pnl
                    self.trades.append({**position, 'exit_time': ts, 'exit_px': exit_px, 'pnl': pnl})
                    self.strat.register_pnl(ts, pnl)
                    position = None
                    self.strat.last_trade_time = ts

            # New entry
            if position is None and not self.strat._in_cooldown(ts) and self.strat.can_trade_today(ts, equity):
                if self.strat.entry_signal_long(row, prev):
                    qty = self.position_size(equity, row['ATR'], row['Close'])
                    if qty > 0:
                        entry_px = row['Close'] * (1 + self.exe.slippage_bps/10000.0)
                        stop = entry_px - self.strat.cfg.atr_mult_sl * row['ATR']
                        take = entry_px + self.strat.cfg.atr_mult_tp * row['ATR']
                        position = {
                            'side':'long','entry_time':ts,'entry':entry_px,'qty':qty,
                            'stop':stop,'take':take,'trail_stop':-math.inf,
                            'symbol': self.symbol
                        }

        # Close at the end if still open
        if position is not None:
            last_row = df.iloc[-1]
            exit_px = last_row['Close']
            pnl = (exit_px - position['entry']) * position['qty'] - self.exe.commission_per_order
            equity += pnl
            self.trades.append({**position, 'exit_time': df.index[-1], 'exit_px': exit_px, 'pnl': pnl})

        results = self._summarize(equity)
        return results

    def _summarize(self, final_equity: float) -> Dict:
        trades = pd.DataFrame(self.trades)
        if trades.empty:
            return {"final_equity": final_equity, "n_trades": 0}
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] <= 0]
        avg_win = wins['pnl'].mean() if not wins.empty else 0.0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0.0
        winrate = len(wins) / len(trades) if len(trades) else 0
        expectancy = winrate*avg_win + (1-winrate)*avg_loss
        max_dd = self._max_drawdown(pd.Series([self.exe.starting_equity] + list(trades['pnl'].cumsum() + self.exe.starting_equity)))
        return {
            "final_equity": round(final_equity, 2),
            "n_trades": int(len(trades)),
            "winrate": round(winrate*100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "max_drawdown": round(max_dd, 2),
            "trades": trades
        }

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        roll_max = equity_curve.cummax()
        drawdown = equity_curve/roll_max - 1.0
        return drawdown.min()*100.0

# -------------------------
# Live Trading (Alpaca Paper)
# -------------------------

class LiveTrader:
    def __init__(self, strat: MomentumScalper, exe: ExecutionConfig, symbol: str):
        if tradeapi is None:
            raise RuntimeError("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
        load_dotenv()
        key = os.getenv('ALPACA_API_KEY_ID')
        sec = os.getenv('ALPACA_API_SECRET_KEY')
        base_url = os.getenv('ALPACA_PAPER_BASE_URL', 'https://paper-api.alpaca.markets')
        if not key or not sec:
            raise RuntimeError("Alpaca API keys missing. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .env")
        self.api = tradeapi.REST(key_id=key, secret_key=sec, base_url=base_url, api_version='v2')
        self.symbol = symbol
        self.strat = strat
        self.exe = exe
        self.position: Optional[Dict] = None
        self.equity = float(self.api.get_account().equity)
        self.last_bar_time: Optional[pd.Timestamp] = None

    def get_latest_bars(self, limit: int = 200) -> pd.DataFrame:
        # Using Alpaca data API v2 for the last N minute bars
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(minutes=limit*2)
        bars = self.api.get_bars(self.symbol, timeframe=tradeapi.TimeFrame.Minute, start=start.isoformat()+"Z", end=end.isoformat()+"Z").df
        if bars.empty:
            raise RuntimeError("No bars returned from Alpaca")
        df = bars.tz_convert(None)
        df = df[['open','high','low','close','volume']].rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        return df

    def position_size(self, atr_value: float, price: float) -> int:
        risk_amt = self.equity * (self.exe.risk_per_trade_pct / 100.0)
        stop_dist = max(atr_value * self.strat.cfg.atr_mult_sl, 0.01)
        shares = int(risk_amt / stop_dist)
        cap = int((self.equity * (self.exe.max_position_pct/100.0)) / price)
        return max(0, min(shares, cap))

    def trade_loop(self, poll_seconds: int = 5):
        print("Starting live loop. Ctrl+C to exit.")
        while True:
            try:
                df = self.get_latest_bars(limit=400)
                df = self.strat.compute_indicators(df)
                row = df.iloc[-1]
                prev = df.iloc[-2]
                ts = row.name

                # Avoid double-processing the same bar
                if self.last_bar_time is not None and ts <= self.last_bar_time:
                    time.sleep(poll_seconds)
                    continue
                self.last_bar_time = ts

                # Update equity from account
                self.equity = float(self.api.get_account().equity)

                # Exit logic
                if self.position is not None:
                    self.position = self.strat.exit_rules(self.position, row)
                    if 'exit_reason' in self.position:
                        self._close_position(row)
                        self.strat.last_trade_time = ts

                # Entry logic
                if self.position is None and not self.strat._in_cooldown(ts) and self.strat.can_trade_today(ts, self.equity):
                    if self.strat.entry_signal_long(row, prev):
                        qty = self.position_size(row['ATR'], row['Close'])
                        if qty > 0:
                            self._open_long(qty, row)

            except KeyboardInterrupt:
                print("Exiting loop.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(poll_seconds)

    def _open_long(self, qty: int, row: pd.Series):
        # Market buy for simplicity
        order = self.api.submit_order(symbol=self.symbol, qty=qty, side='buy', type='market', time_in_force='day')
        # Assume fill near last price
        entry_px = float(row['Close']) * (1 + self.exe.slippage_bps/10000.0)
        stop = entry_px - self.strat.cfg.atr_mult_sl * float(row['ATR'])
        take = entry_px + self.strat.cfg.atr_mult_tp * float(row['ATR'])
        self.position = {'side':'long','entry_time':row.name,'entry':entry_px,'qty':qty,'stop':stop,'take':take,'trail_stop':-math.inf,'symbol':self.symbol,'alpaca_order_id':order.id}
        print(f"Opened LONG {qty} {self.symbol} @ ~{entry_px:.2f}")

    def _close_position(self, row: pd.Series):
        if self.position is None:
            return
        qty = self.position['qty']
        order = self.api.submit_order(symbol=self.symbol, qty=qty, side='sell', type='market', time_in_force='day')
        exit_px = float(row['Close']) * (1 - self.exe.slippage_bps/10000.0)
        pnl = (exit_px - self.position['entry']) * qty - self.exe.commission_per_order
        print(f"Closed LONG {qty} {self.symbol} @ ~{exit_px:.2f} | PnL {pnl:.2f} ({self.position.get('exit_reason')})")
        self.position = None

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Momentum Scalping Bot (1m)")
    p.add_argument('--mode', choices=['backtest','live'], default='backtest')
    p.add_argument('--symbol', type=str, required=True, help='Ticker, e.g., AAPL')
    p.add_argument('--days', type=int, default=5, help='Backtest: number of recent days for 1m data')
    p.add_argument('--starting-equity', type=float, default=25_000.0)
    p.add_argument('--risk', type=float, default=0.25, help='% equity risk per trade')
    p.add_argument('--maxpos', type=float, default=33.0, help='Max position % of equity')
    return p.parse_args()


def main():
    args = parse_args()

    strat_cfg = StrategyConfig()
    exe_cfg = ExecutionConfig(starting_equity=args.starting_equity, risk_per_trade_pct=args.risk, max_position_pct=args.maxpos)
    strat = MomentumScalper(strat_cfg)

    if args.mode == 'backtest':
        dp = DataProvider(args.symbol)
        df = dp.fetch_yfinance_1m(days=args.days)
        bt = Backtester(strat, exe_cfg, args.symbol)
        results = bt.run(df)
        print("\n=== Backtest Summary ===")
        print({k:v for k,v in results.items() if k != 'trades'})
        trades = results.get('trades')
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            csv_path = f"trades_{args.symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades.to_csv(csv_path, index=False)
            print(f"Saved trades to {csv_path}")
    else:
        trader = LiveTrader(strat, exe_cfg, args.symbol)
        trader.trade_loop()


if __name__ == '__main__':
    main()
