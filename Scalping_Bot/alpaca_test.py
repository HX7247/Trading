import os
from alpaca.trading.client import TradingClient

key = os.getenv("ALPACA_KEY_ID")
secret = os.getenv("ALPACA_SECRET_KEY")
base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

tc = TradingClient(api_key=key, secret_key=secret, paper=("paper" in base_url))
acct = tc.get_account()

print("Account ID:", acct.id)
print("Status:", acct.status)
print("Buying power:", acct.buying_power)
