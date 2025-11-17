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
    ten_years_ago = datetime.now(hist.index.tz) - timedelta(days=365*10)
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
    chi_squared_dof = {}
    bic_scores = {}
    n = len(x)
    
    print("\nFitting polynomials...")
    print("-" * 70)
    
    for order in range(1, 10):
        coeffs = np.polyfit(x, y, order)
        poly = np.poly1d(coeffs)
        fits[order] = poly
        
        # Calculate R-squared
        y_pred = poly(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)1
        
        # Calculate chi-squared per degree of freedom
        # DOF = n - (order + 1) where order+1 is number of parameters
        dof = n - (order + 1)
        chi_squared = ss_res / dof
        chi_squared_dof[order] = chi_squared
        
        # Calculate BIC (Bayesian Information Criterion)
        # BIC = n*ln(SS_res/n) + k*ln(n)
        # where k = order + 1 (number of parameters including intercept)
        k = order + 1
        bic = n * np.log(ss_res / n) + k * np.log(n)
        bic_scores[order] = bic
        
        print(f"Order {order}: R² = {r_squared:.6f}, χ²/DOF = {chi_squared:.6f}, BIC = {bic:.2f}")
    
    print("-" * 70)
    return fits, chi_squared_dof, bic_scores

def forecast_polynomials(fits, current_x, future_days=365*10):
    # Forecast future prices using fitted polynomials
    forecasts = {}
    future_x = np.arange(current_x[-1], current_x[-1] + future_days)
    
    for order, poly in fits.items():
        forecasts[order] = poly(future_x)
    
    return future_x, forecasts

def plot_results(x, y, fits, future_x, forecasts, ticker, chi_squared_dof, bic_scores):
    # Plot historical data and forecasts
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.5, top=0.92, bottom=0.08)
    
    # Add main title
    fig.suptitle(f'{ticker} - Polynomial Forecasting (Orders 1-9)', fontsize=11, fontweight='bold')
    
    # Plot polynomial fits (9 subplots)
    for idx, order in enumerate(range(1, 10)):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Plot historical data
        ax.scatter(x, y, alpha=0.3, s=3, label='Historical Data', color='blue')
        
        # Plot fitted polynomial
        y_fit = fits[order](x)
        ax.plot(x, y_fit, 'g-', linewidth=1, label='Fit')
        
        # Plot forecast
        ax.plot(future_x, forecasts[order], 'r--', linewidth=1, label='Forecast')
        
        ax.set_title(f'Order {order}', fontsize=7)
        ax.set_xlabel('Days', fontsize=6)
        ax.set_ylabel('Price', fontsize=6)
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=5)
    
    plt.show()
    
    # Plot chi-squared per DOF and BIC on separate page
    fig_metrics = plt.figure(figsize=(14, 6))
    
    orders = list(chi_squared_dof.keys())
    chi_values = list(chi_squared_dof.values())
    bic_values = list(bic_scores.values())
    
    # Find best models
    best_chi_order = min(chi_squared_dof, key=chi_squared_dof.get)
    best_bic_order = min(bic_scores, key=bic_scores.get)
    
    # Plot 1: Chi-squared per DOF
    ax1 = fig_metrics.add_subplot(121)
    ax1.plot(orders, chi_values, 'bo-', linewidth=2, markersize=8, markerfacecolor='lightblue', markeredgewidth=2)
    ax1.plot(best_chi_order, chi_squared_dof[best_chi_order], 'r*', markersize=20, label=f'Best (Order {best_chi_order})', zorder=5)
    ax1.set_xlabel('Polynomial Order', fontsize=10, fontweight='bold')
    ax1.set_ylabel('χ²/DOF', fontsize=10, fontweight='bold')
    ax1.set_title('χ² per Degree of Freedom vs Polynomial Order', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(orders)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=9)
    
    # Plot 2: BIC
    ax2 = fig_metrics.add_subplot(122)
    ax2.plot(orders, bic_values, 'go-', linewidth=2, markersize=8, markerfacecolor='lightgreen', markeredgewidth=2)
    ax2.plot(best_bic_order, bic_scores[best_bic_order], 'r*', markersize=20, label=f'Best (Order {best_bic_order})', zorder=5)
    ax2.set_xlabel('Polynomial Order', fontsize=10, fontweight='bold')
    ax2.set_ylabel('BIC', fontsize=10, fontweight='bold')
    ax2.set_title('BIC vs Polynomial Order', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(orders)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=9)
    
    plt.suptitle(f'{ticker} - Model Selection Metrics', fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Summary analysis
    print("\n" + "="*70)
    print("MODEL COMPARISON ANALYSIS")
    print("="*70)
    print(f"\nBest model by χ²/DOF: Order {best_chi_order}")
    print(f"  χ²/DOF = {chi_squared_dof[best_chi_order]:.6f}")
    print(f"\nBest model by BIC: Order {best_bic_order}")
    print(f"  BIC = {bic_scores[best_bic_order]:.2f}")
    print(f"\nComparison:")
    print(f"  χ²/DOF penalizes goodness-of-fit without model complexity penalty")
    print(f"  BIC penalizes both residual error AND model complexity (order)")
    print(f"  Lower BIC → better balance between fit quality and simplicity")
    print(f"  Lower χ²/DOF → better fit but may overfit with higher orders")
    
    if best_bic_order == best_chi_order:
        print(f"\n✓ Both metrics agree: Order {best_bic_order} is optimal")
    else:
        print(f"\n⚠ Metrics disagree: BIC prefers Order {best_bic_order} (simpler), χ²/DOF prefers Order {best_chi_order}")
    
    print("="*70 + "\n")

def main():
    # Main program flow
    print("\n" + "="*50)
    print("Stock Price Forecasting with Polynomial Models")
    print("Fitting & Forecasting Activity: 3 Model Testing")
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
    fits, chi_squared_dof, bic_scores = fit_polynomials(x, y)
    
    # Forecast future prices
    print(f"\nForecasting next 10 years...")
    future_x, forecasts = forecast_polynomials(fits, x)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(x, y, fits, future_x, forecasts, ticker, chi_squared_dof, bic_scores)
    
    print("Done!")

if __name__ == "__main__":
    main()