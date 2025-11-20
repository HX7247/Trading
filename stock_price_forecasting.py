import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from numpy.polynomial import polynomial as P
import numpy.linalg as la

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

def _fit_polynomial_with_cov(x, y, degree):
    """
    Fit polynomial on scaled x, return coeffs (lowest->highest), cov matrix, ss_res, dof.
    Uses linear least squares with design matrix and computes parameter covariance:
      cov = sigma2 * (A^T A)^{-1},  sigma2 = ss_res / (n - p)
    x is 1D array of sample indices (e.g. days). We scale x to improve conditioning.
    """
    n = len(x)
    p = degree + 1

    # scale x
    x0 = x.mean()
    sx = x.std() if x.std() != 0 else 1.0
    x_s = (x - x0) / sx

    # design matrix: columns [1, x, x^2, ..., x^d]
    A = np.column_stack([x_s**k for k in range(p)])  # shape (n, p)

    # least squares
    coeffs, residuals, rank, svals = la.lstsq(A, y, rcond=None)
    y_pred = A.dot(coeffs)
    ss_res = float(np.sum((y - y_pred) ** 2))

    dof = n - p
    if dof <= 0:
        sigma2 = np.nan
        cov = np.full((p, p), np.nan)
    else:
        sigma2 = ss_res / dof
        # compute (A^T A)^{-1}
        ATA_inv = la.inv(A.T.dot(A))
        cov = sigma2 * ATA_inv

    # return coeffs (a0 + a1 x + ...), cov, scaling params, metrics
    return {
        "degree": degree,
        "coeffs": coeffs,                 # a0..ad for scaled x
        "cov": cov,                       # covariance matrix of coeffs
        "x0": x0,
        "sx": sx,
        "ss_res": ss_res,
        "dof": dof,
        "sigma2": sigma2,
        "n": n,
        "p": p
    }

def _poly_eval(fit, x_new):
    """Evaluate polynomial fit dict on raw x_new values (unscaled)."""
    x_s = (x_new - fit["x0"]) / fit["sx"]
    powers = np.column_stack([x_s**k for k in range(fit["p"])])
    return powers.dot(fit["coeffs"])

def fit_polynomials_with_stats(x, y, max_order=9):
    """
    Replace previous fit_polynomials: returns dict of fits keyed by order and computed BIC, chi2/dof.
    """
    fits = {}
    chi2_dof = {}
    bic = {}

    for order in range(1, max_order + 1):
        res = _fit_polynomial_with_cov(x, y, order)
        fits[order] = res

        ss_res = res["ss_res"]
        n = res["n"]
        p = res["p"]
        dof = res["dof"]

        chi2_per_dof = ss_res / dof if dof > 0 else np.nan
        chi2_dof[order] = chi2_per_dof

        # BIC = n * ln(SS_res/n) + k * ln(n), where k = p
        bic_val = n * np.log(ss_res / n) + p * np.log(n)
        bic[order] = bic_val

    return fits, chi2_dof, bic

def fit_exponential(x, y):
    """
    Fit y = A * exp(b * x_s) by linearizing: ln(y) = ln(A) + b * x_s
    Returns params {A, b}, covariance matrix for [A, b], ss_res (on original y), BIC, chi2/dof.
    Requires y > 0 for log transform.
    """
    mask = y > 0
    if mask.sum() < 3:
        raise RuntimeError("Not enough positive y values for exponential fit")

    x_pos = x[mask]
    y_pos = y[mask]

    # scale x same way as polynomial fits
    x0 = x_pos.mean()
    sx = x_pos.std() if x_pos.std() != 0 else 1.0
    x_s = (x_pos - x0) / sx

    # design for linear regression on log(y)
    A = np.column_stack([np.ones_like(x_s), x_s])  # [1, x_s]
    ly = np.log(y_pos)

    coeffs_lin, residuals, rank, svals = la.lstsq(A, ly, rcond=None)
    lnA, b = coeffs_lin
    y_pred = np.exp(lnA + b * x_s)
    ss_res = float(np.sum((y_pos - y_pred) ** 2))

    n = len(y_pos)
    p = 2
    dof = n - p
    sigma2 = ss_res / dof if dof > 0 else np.nan

    # covariance in log-space: cov_ln = sigma2_log * (A^T A)^{-1}
    # but we computed residuals in original space; to approximate covariance of lnA and b,
    # we instead compute covariance from linear regression on ln(y):
    ly_pred = A.dot(coeffs_lin)
    ss_res_log = float(np.sum((ly - ly_pred) ** 2))
    sigma2_log = ss_res_log / (n - p) if n - p > 0 else np.nan
    ATA_inv = la.inv(A.T.dot(A))
    cov_ln = sigma2_log * ATA_inv  # covariance of [lnA, b]

    # transform covariance to [A, b] using delta method
    A_param = np.exp(lnA)
    # var(A) ≈ (dA/dlnA)^2 var(lnA) = A^2 * var(lnA)
    var_A = (A_param**2) * cov_ln[0, 0]
    cov_Ab = A_param * cov_ln[0, 1]  # cov(A, b) ≈ A * cov(lnA, b)
    cov_mat = np.array([[var_A, cov_Ab], [cov_Ab, cov_ln[1, 1]]])

    # BIC using ss_res on original y
    bic = n * np.log(ss_res / n) + p * np.log(n) if ss_res > 0 else np.nan
    chi2dof = ss_res / dof if dof > 0 else np.nan

    return {
        "A": A_param,
        "b": b,
        "cov": cov_mat,
        "ss_res": ss_res,
        "n": n,
        "p": p,
        "dof": dof,
        "chi2dof": chi2dof,
        "bic": bic,
        "x0": x0,
        "sx": sx
    }

def analyze_and_report(x, y, fits_dict, chi2_dof, bic_scores):
    """
    For the best model by BIC: print parameter values, covariance matrix, uncertainties.
    Also fit exponential model and compare metrics (chi2/DOF and BIC).
    """
    # choose best polynomial by BIC
    best_poly = min(bic_scores, key=bic_scores.get)
    poly_fit = fits_dict[best_poly]

    coeffs = poly_fit["coeffs"]
    cov = poly_fit["cov"]
    p = poly_fit["p"]

    # parameter uncertainties (std dev)
    uncert = np.sqrt(np.diag(cov)) if cov is not None else np.full(p, np.nan)

    print("\n" + "="*60)
    print(f"BEST POLYNOMIAL MODEL: degree = {best_poly}")
    print("-" * 60)
    for i, (c, u) in enumerate(zip(coeffs, uncert)):
        print(f"coeff a{i:02d} (x^{i}): {c:.6e} ± {u:.6e}")
    print("\nCovariance matrix (rows/cols correspond to a0..a{d}):")
    with np.printoptions(precision=4, suppress=True):
        print(cov)
    print(f"\nχ²/DOF = {chi2_dof[best_poly]:.6e}")
    print(f"BIC      = {bic_scores[best_poly]:.6e}")
    print("="*60)

    # Exponential fit
    try:
        exp_res = fit_exponential(x, y)
        print("\nExponential model fit: y = A * exp(b * x_s)")
        print(f" A = {exp_res['A']:.6e} ± {np.sqrt(exp_res['cov'][0,0]):.6e}")
        print(f" b = {exp_res['b']:.6e} ± {np.sqrt(exp_res['cov'][1,1]):.6e}")
        print("\nCovariance matrix for [A, b]:")
        with np.printoptions(precision=4, suppress=True):
            print(exp_res["cov"])
        print(f"\nExponential χ²/DOF = {exp_res['chi2dof']:.6e}")
        print(f"Exponential BIC    = {exp_res['bic']:.6e}")
    except Exception as e:
        print("\nExponential fit failed:", e)
        exp_res = None

    # Compare BIC and χ²/DOF
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("-" * 60)
    print(f"Polynomial (degree {best_poly}): χ²/DOF = {chi2_dof[best_poly]:.6e}, BIC = {bic_scores[best_poly]:.6e}")
    if exp_res:
        print(f"Exponential                : χ²/DOF = {exp_res['chi2dof']:.6e}, BIC = {exp_res['bic']:.6e}")
        if exp_res['bic'] < bic_scores[best_poly]:
            print("\n-> Exponential model has lower BIC (preferred by BIC).")
        elif exp_res['bic'] > bic_scores[best_poly]:
            print("\n-> Polynomial model has lower BIC (preferred by BIC).")
        else:
            print("\n-> BIC equal (tie).")
    else:
        print("Exponential model not available for comparison.")

    print("="*60 + "\n")

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
        r_squared = 1 - (ss_res / ss_tot)
        
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
    
    # Fit polynomials (with stats)
    print(f"\nFitting polynomials to {ticker} data...")
    fits, chi_squared_dof, bic_scores = fit_polynomials_with_stats(x, y, max_order=9)
    
    # Analyze best model and compare to exponential
    analyze_and_report(x, y, fits, chi_squared_dof, bic_scores)
    
    # Forecast future prices using the fitted models
    print(f"\nForecasting next 10 years...")
    # build callable evaluators from the fit dicts (use _poly_eval which handles scaling)
    poly_callables = {k: (lambda xx, f=fits[k]: _poly_eval(f, xx)) for k in fits}
    
    # get future_x and forecasts (forecast_polynomials expects callables)
    future_x, forecasts = forecast_polynomials(poly_callables, x)
    
    # Plot results (pass the same callables as 'fits' so plotting uses them for y_fit)
    print("\nGenerating plots...")
    plot_results(x, y, poly_callables, future_x, forecasts, ticker, chi_squared_dof, bic_scores)
    
    print("Done!")

if __name__ == "__main__":
    main()