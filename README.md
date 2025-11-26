# Trading Projects — Overview

## Projects

### 1) Scalping Bot
- File: `bot.py`
- Purpose: Real-time 9‑EMA / 21‑EMA crossover strategy on 5‑minute bars.
- Features:
  - Trades multiple symbols concurrently.
  - Places orders with stop loss and take profit.
  - Risk management: position sizing targets 2% of account equity.
  - Stop loss: 1% below entry; Take profit: 2% above entry.
  - Polls every 5 minutes and only trades when market is open.
- Notes:
  - Uses Alpaca (alpaca-py) for data and trading.
  - Robust handling for different alpaca-py versions; normalizes TimeFrame for compatibility.
  - Logs actions and warnings to stdout.

### 2) Stock Price Forecasting / Model Testing
- File: `stock_price_forecasting.py`
- Purpose: Historical price fitting and model selection.
- Features:
  - Fetches historical data via `yfinance`.
  - Fits polynomials (orders 1–9) with parameter covariance, χ²/DOF, and BIC.
  - Evaluates exponential model (y = A * exp(b x_s)) and compares metrics.
  - Plots fits and model-selection metrics (χ²/DOF and BIC).
  - Prints best-model parameters, covariance matrix, and uncertainties.

## Important files
- `bot.py` — trading bot (requires Alpaca credentials).
- `stock_price_forecasting.py` — offline forecasting and model testing.
- `.env` — environment variables (API keys, timeframe, EMA spans, risk settings).
- `README.md` — this file.


## Dependencies
Install in the project virtualenv:
- pip install alpaca-py pandas numpy python-dotenv pytz yfinance matplotlib

Check installed versions if you hit compatibility issues (especially `alpaca-py`).


## .env example
Place a `.env` in this folder (do NOT commit real keys to VCS):
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
APCA_BASE_URL=https://paper-api.alpaca.markets

TIMEFRAME=5Minute
SHORT_EMA=9
LONG_EMA=21
DOLLARS_PER_TRADE=200
POLL_INTERVAL_SEC=300

## How to run

1. Trading bot (paper mode recommended for testing):
   - Activate your virtualenv, ensure `.env` set.
   - python bot.py
   - Keep the process running; the bot polls and trades during market hours.

2. Forecasting script:
   - python stock_price_forecasting.py
   - Follow prompts to choose a ticker. Script will fetch data, fit models, plot results, and print analysis.





