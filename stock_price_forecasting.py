import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from numpy.polynomial import polynomial as P

# Array of stock tickers to choose from
STOCK_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']

def display_ticker_options():
    # Display available stock tickers
    print("\n" + "="*50)
    print("Available Stock Tickers:")
    print("="*50)
    for i, ticker in enumerate(STOCK_TICKERS, 1):
        print(f"{i}. {ticker}")
    print("="*50)

def get_user_ticker():
    # Get user's stock ticker choice
    while True:
        display_ticker_options()
        try:
            choice = int(input("\nEnter the number of your choice (1-10): "))
            if 1 <= choice <= len(STOCK_TICKERS):
                return STOCK_TICKERS[choice - 1]
            else:
                print("Invalid choice. Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def fetch_stock_data(ticker):
    # Fetch historical stock data from Yahoo Finance
    print(f"\nFetching data for {ticker}...")
    
    # Fetch data - yfinance will go back as far as available
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    
    if hist.empty:
        print(f"Error: Could not fetch data for {ticker}")
        return None
    
    # Get the last 10 years or all available data
    ten_years_ago = datetime.now() - timedelta(days=365*10)
    hist = hist[hist.index >= ten_years_ago]
    
    print(f"Data fetched: {len(hist)} trading days from {hist.index[0].date()} to {hist.index[-1].date()}")
    return hist

def prepare_data(hist):
    # Prepare data for polynomial fitting
    # Use daily closing prices
    prices = hist['Close'].values
    
    # Create x-axis as days from start
    days = np.arange(len(prices))
    
    return days, prices

def fit_polynomials(x, y):
    # Fit polynomials of order 1-9 to the data
    fits = {}
    for order in range(1, 10):
        coeffs = np.polyfit(x, y, order)
        poly = np.poly1d(coeffs)
        fits[order] = poly
        
        # Calculate R-squared
        y_pred = poly(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Order {order}: R² = {r_squared:.6f}")
    
    return fits

def forecast_polynomials(fits, current_x, future_days=365*10):
    # Forecast future prices using fitted polynomials
    forecasts = {}
    future_x = np.arange(current_x[-1], current_x[-1] + future_days)
    
    for order, poly in fits.items():
        forecasts[order] = poly(future_x)
    
    return future_x, forecasts

def plot_results(x, y, fits, future_x, forecasts, ticker):
    # Plot historical data and forecasts
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'{ticker} - Polynomial Forecasting (Orders 1-9)', fontsize=16)
    
    axes = axes.flatten()
    
    for idx, order in enumerate(range(1, 10)):
        ax = axes[idx]
        
        # Plot historical data
        ax.scatter(x, y, alpha=0.3, s=10, label='Historical Data', color='blue')
        
        # Plot fitted polynomial
        y_fit = fits[order](x)
        ax.plot(x, y_fit, 'g-', linewidth=2, label='Fit')
        
        # Plot forecast
        ax.plot(future_x, forecasts[order], 'r--', linewidth=2, label='Forecast')
        
        ax.set_title(f'Order {order}')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Main program flow
    print("\n" + "="*50)
    print("Stock Price Forecasting with Polynomial Models")
    print("="*50)
    
    # Get user's ticker choice
    ticker = get_user_ticker()
    
    # Fetch historical data
    hist = fetch_stock_data(ticker)
    if hist is None:
        return
    
    # Prepare data
    x, y = prepare_data(hist)
    
    # Fit polynomials
    print(f"\nFitting polynomials to {ticker} data...")
    fits = fit_polynomials(x, y)
    
    # Forecast future prices
    print(f"\nForecasting next 10 years...")
    future_x, forecasts = forecast_polynomials(fits, x)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(x, y, fits, future_x, forecasts, ticker)
    
    print("\nDone!")

if __name__ == "__main__":
    main()