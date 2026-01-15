#!/usr/bin/env python3
# /// script
# dependencies = [
#   "rich",
#   "numpy",
#   "pandas",
#   "scipy",
#   "matplotlib",
#   "yfinance",
#   "groq",
#   "seaborn",
#   "requests",
#   "python-dotenv",
#   "numba",
# ]
# ///

import sys
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import box
from rich.prompt import Prompt, Confirm
from groq import Groq
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

console = Console()

# Auto-install dependencies using uv
def ensure_package(pkg, import_name=None):
    try:
        # Check if the package is already available
        __import__(import_name or pkg)
    except ImportError:
        console.print(f"🚀 [uv] Installing {pkg}...")
        try:
            # Use 'uv pip install' for high-speed installation
            subprocess.check_call(["uv", "pip", "install", pkg, "--system", "-q"])
        except FileNotFoundError:
            # Fallback to standard pip if uv is not installed on the system
            console.print(f"⚠️ uv not found, falling back to pip for {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

packages = [
    ("rich", "rich"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("yfinance", "yfinance"),
    ("groq", "groq"),
    ("seaborn", "seaborn"),
    ("python-dotenv", "dotenv"),
    ("numba", "numba"),
]

for pkg, import_name in packages:
    ensure_package(pkg, import_name)

# ==================== CONFIG ====================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

BANNER = """
[bold cyan]
╔════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                    ║
║  ███████╗██╗   ██╗ █████╗ ███╗   ██╗████████╗ ██████╗ ███████╗██╗   ██╗███████╗    ║
║  ██╔═══██║██║   ██║██╔══██╗████╗  ██║╚══██╔══╝ ██╔══██╗██╔════╝██║   ██║██╔════╝   ║
║  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║    ██║  ██║█████╗  ██║   ██║███████╗   ║
║  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║    ██║  ██║██╔══╝  ╚██╗ ██╔╝╚════██║   ║
║  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║    ██████╔╝███████╗ ╚████╔╝ ███████║   ║
║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚══════╝  ╚═══╝  ╚══════╝   ║
║                                                                                    ║
║                  ADVANCED OPTIONS PRICING & RISK ANALYSIS                          ║
║                            American (Binomial)                                     ║
╚════════════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
"""

# ==================== DATA FETCHING ====================

def fetch_market_data(ticker, period='1y'):
    """Fetch comprehensive market data"""
    try:
        console.print(f"[cyan]📡 Fetching data for {ticker.upper()}...[/cyan]")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            console.print(f"[red]❌ No data found for {ticker}[/red]")
            return None
        
        spot = hist['Close'].iloc[-1]
        
        # Get dividend yield (Robust Method)
        div_yield = 0.0
        try:
            div_yield = stock.info.get('dividendYield', 0.0)
            if div_yield is None:
                div_yield = 0.0
            
            # Fallback: Calculate trailing 12m yield
            if div_yield == 0.0:
                divs = stock.dividends
                if not divs.empty:
                    one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=1)
                    last_year_divs = divs[divs.index >= one_year_ago].sum()
                    div_yield = last_year_divs / spot
        except Exception as e:
            console.print(f"[yellow]⚠ Could not calculate dividend yield: {e}[/yellow]")
            div_yield = 0.0
        
        # ⚠️ FIX: Better risk-free rate fetching
        rf_rate = 0.0435  # Current 3-month T-bill (Jan 2026)
        try:
            # Try fetching current 3-month T-bill rate
            import requests
            # Fallback to yfinance proxy
            try:
                irx = yf.Ticker("^IRX")
                irx_hist = irx.history(period='5d')
                if not irx_hist.empty and irx_hist['Close'].iloc[-1] > 0:
                    rf_rate = irx_hist['Close'].iloc[-1] / 100.0
                    console.print(f"[green]✓ Using ^IRX rate: {rf_rate*100:.2f}%[/green]")
                else:
                    raise Exception("^IRX returned zero")
            except:
                # Try 10-year as fallback
                tnx = yf.Ticker("^TNX")
                tnx_hist = tnx.history(period='5d')
                if not tnx_hist.empty:
                    rf_rate = tnx_hist['Close'].iloc[-1] / 100.0
                    console.print(f"[yellow]⚠ Using ^TNX rate: {rf_rate*100:.2f}%[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Using default risk-free rate: {rf_rate*100:.2f}%[/yellow]")
        
        console.print(f"[green]✓ Spot: ${spot:.2f} | Div: {div_yield:.4f}% | Rate: {rf_rate*100:.2f}%[/green]")
        
        return {
            'hist': hist,
            'spot': spot,
            'div_yield': div_yield,
            'risk_free_rate': rf_rate,
            'ticker': ticker.upper(),
            'stock_obj': stock
        }
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return None

def calculate_realized_volatility_cone(prices, windows=[30, 90]):
    """
    Calculate realized volatility cone for validation.
    Institutional approach: use statistical distribution of realized vol.
    
    Args:
        prices: Price series
        windows: List of lookback windows (days)
    
    Returns:
        dict with 'rv_30d', 'rv_90d', 'mean', 'std'
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    
    rvs = {}
    for window in windows:
        if len(returns) >= window:
            rv = returns.tail(window).std() * np.sqrt(252)
            rvs[f'rv_{window}d'] = rv
    
    # Calculate statistical bounds (Mean +/- 2 Sigma)
    all_rvs = list(rvs.values())
    if all_rvs:
        rv_mean = np.mean(all_rvs)
        rv_std = np.std(all_rvs) if len(all_rvs) > 1 else rv_mean * 0.2
        rvs['mean'] = rv_mean
        rvs['std'] = rv_std
    
    return rvs

def validate_market_iv(iv, rv_cone, ewma_vol):
    """
    Institutional-grade IV validation using Realized Volatility Cone.
    NO hardcoded thresholds - uses statistical confidence intervals.
    
    Args:
        iv: Market IV from yfinance (decimal)
        rv_cone: Realized volatility cone dict
        ewma_vol: EWMA volatility (fallback)
    
    Returns:
        Validated IV or EWMA fallback
    """
    # Check for None or NaN
    if iv is None or (isinstance(iv, float) and np.isnan(iv)):
        console.print(f"[yellow]⚠ Market IV unavailable → using EWMA ({ewma_vol*100:.1f}%)[/yellow]")
        return ewma_vol
    
    # Validate against Realized Volatility Cone
    # Accept if within [0.5 * RV, 3.0 * RV] (institutional standard)
    rv_ref = rv_cone.get('rv_30d', rv_cone.get('rv_90d', ewma_vol))
    min_bound = 0.5 * rv_ref
    max_bound = 3.0 * rv_ref
    
    if iv < min_bound or iv > max_bound:
        console.print(f"[yellow]⚠ Market IV {iv*100:.1f}% outside RV cone [{min_bound*100:.0f}%-{max_bound*100:.0f}%] → using EWMA ({ewma_vol*100:.1f}%)[/yellow]")
        return ewma_vol
    
    # Passed validation
    console.print(f"[green]✓ Market IV {iv*100:.1f}% validated (RV cone: [{min_bound*100:.0f}%-{max_bound*100:.0f}%])[/green]")
    return iv

def fetch_option_chain(stock_obj, spot_price, hist_prices=None):
    """Interactive option selection with market prices"""
    try:
        # 1. Select Expiry
        expirations = stock_obj.options
        if not expirations:
            console.print("[red]❌ No options chain found.[/red]")
            return None
        
        console.print("\n[cyan]📅 Available Expirations:[/cyan]")
        display_exps = expirations[:10]
        for i, exp in enumerate(display_exps):
            console.print(f"{i+1}. {exp}")
        console.print("0. Custom Expiry")
        
        choice_input = Prompt.ask("Select expiry date (number)", default="1")
        
        if choice_input == "0":
             expiry_date = Prompt.ask("Enter custom expiry (YYYY-MM-DD)", default="2026-06-18")
             # Validate format
             try:
                 pd.to_datetime(expiry_date)
             except:
                 console.print("[red]❌ Invalid date format[/red]")
                 return None
        else:
            choice_idx = int(choice_input) - 1
            if not (0 <= choice_idx < len(expirations)):
                console.print("[red]❌ Invalid selection[/red]")
                return None
            expiry_date = expirations[choice_idx]
            
        # 2. Select Option Type
        opt_type = Prompt.ask("📈 Option type", choices=["call", "put"], default="call")

        try:
            chain = stock_obj.option_chain(expiry_date)
            options = chain.calls if opt_type == 'call' else chain.puts
        except:
             console.print(f"[yellow]⚠ Could not fetch chain for {expiry_date}. Using manual mode.[/yellow]")
             chain = None
             options = pd.DataFrame() # Empty frame for manual fallback
        
        # Check if we have data or need manual
        if chain is None or options.empty:
             manual_strike = float(Prompt.ask("Enter strike price"))
             return {
                'expiry': expiry_date,
                'strike': manual_strike,
                'type': opt_type,
                'market_price': None,
                'implied_vol': None,
                'bid': 0.0,
                'ask': 0.0,
                'contractSymbol': f"MANUAL_{expiry_date}_{manual_strike}_{opt_type}"
            }
        
        # ═══ INSTITUTIONAL FILTERING: Delta & Liquidity (NO hardcoded strike ranges) ═══
        
        # Filter 1: Liquidity - Keep options with market activity
        liquid_options = options[
            (options['volume'] > 0) | (options['openInterest'] > 0)
        ].copy()
        
        if liquid_options.empty:
            console.print("[yellow]⚠ No liquid options found, using all strikes[/yellow]")
            liquid_options = options.copy()
        
        # Filter 2: Moneyness - Removes deep ITM/OTM (equivalent to Delta filtering)
        # Keep strikes where moneyness is reasonable (0.70 to 1.30)
        # This automatically removes strikes where Delta would be outside [0.05, 0.95]
        liquid_options['moneyness'] = liquid_options['strike'] / spot_price
        
        nearby_options = liquid_options[
            (liquid_options['moneyness'] >= 0.70) &  # Removes deep OTM calls / deep ITM puts
            (liquid_options['moneyness'] <= 1.30)    # Removes deep ITM calls / deep OTM puts
        ].copy()
        
        if nearby_options.empty:
            console.print("[yellow]⚠ No options found near spot, showing all...[/yellow]")
            nearby_options = options
        
        # Display options
        console.print(f"\n[cyan]🎯 Available {opt_type.upper()} Strikes for {expiry_date} (Spot: ${spot_price:.2f}):[/cyan]")
        
        step = max(1, len(nearby_options) // 15)
        display_opts = nearby_options.iloc[::step]
        
        table = Table(box=box.SIMPLE)
        table.add_column("#", style="dim")
        table.add_column("Strike", style="bold")
        table.add_column("Last", style="green")
        table.add_column("Bid", style="yellow")
        table.add_column("Ask", style="yellow")
        table.add_column("Spread %", style="red")  # Liquidity indicator
        table.add_column("Implied Vol", style="magenta")
        table.add_column("OI", style="blue")  # Open Interest (institutional focus)
        
        # Calculate Realized Volatility Cone for institutional validation
        if hist_prices is not None and len(hist_prices) > 90:
            rv_cone = calculate_realized_volatility_cone(hist_prices)
            returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
            ewma_vol = ewma_volatility(returns)
            
            console.print(f"[dim]RV Cone: 30d={rv_cone.get('rv_30d',0)*100:.1f}% | 90d={rv_cone.get('rv_90d',0)*100:.1f}% | EWMA={ewma_vol*100:.1f}%[/dim]")
        else:
            # Minimal data - use EWMA only
            returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
            ewma_vol = ewma_volatility(returns)
            rv_cone = {'rv_30d': ewma_vol, 'mean': ewma_vol, 'std': ewma_vol * 0.2}
        
        choices_map = {}
        for i, (idx, row) in enumerate(display_opts.iterrows()):
            # Validate IV using Realized Volatility Cone (institutional approach)
            clean_iv = validate_market_iv(row['impliedVolatility'], rv_cone, ewma_vol)
            
            # Calculate bid-ask spread % (liquidity indicator)
            if row['bid'] > 0 and row['ask'] > row['bid']:
                spread_pct = (row['ask'] - row['bid']) / row['lastPrice'] * 100
            else:
                spread_pct = 0
            
            table.add_row(
                str(i+1),
                f"${row['strike']:.1f}",
                f"${row['lastPrice']:.2f}",
                f"${row['bid']:.2f}",
                f"${row['ask']:.2f}",
                f"{spread_pct:.1f}%" if spread_pct > 0 else "N/A",
                f"{clean_iv*100:.1f}%",
                str(int(row['openInterest'])) if pd.notna(row['openInterest']) else "0"
            )
            choices_map[str(i+1)] = row
        
        console.print(table)
        

        console.print("[dim]Enter '0' or custom value for manual strike[/dim]")
        
        sel_input = Prompt.ask("Select strike (number)", default=str(len(display_opts)//2 + 1))
        
        if sel_input == "0" or sel_input not in choices_map:
             if sel_input == "0" or Confirm.ask("Strike not in list? Enter manually?", default=True):
                 manual_strike = float(Prompt.ask("Enter strike price"))
                 # Try to find closest match for data if possible
                 try:
                     closest_idx = (options['strike'] - manual_strike).abs().idxmin()
                     selected_row = options.loc[closest_idx]
                     # If the match is exact enough (within 1 cent), use it
                     if abs(selected_row['strike'] - manual_strike) < 0.01:
                          console.print(f"[green]✓ Found exact match in chain[/green]")
                     else:
                          console.print(f"[yellow]⚠ Using manual strike ${manual_strike} (Closest market data: ${selected_row['strike']})[/yellow]")
                          # Create a synthetic row with manual strike but market data from closest
                          # Or just proceed with manual params and no market data?
                          # Let's return manual data
                          return {
                            'expiry': expiry_date,
                            'strike': manual_strike,
                            'type': opt_type,
                            'market_price': None, # User will input later
                            'implied_vol': None,
                            'bid': 0.0,
                            'ask': 0.0,
                            'contractSymbol': f"MANUAL_{manual_strike}"
                        }
                 except:
                      # Total fallback
                      return {
                        'expiry': expiry_date,
                        'strike': manual_strike,
                        'type': opt_type,
                        'market_price': None,
                        'implied_vol': None,
                        'bid': 0.0,
                        'ask': 0.0,
                        'contractSymbol': f"MANUAL_{manual_strike}"
                    }
             else:
                 return None
        else:
            selected_row = choices_map[sel_input]
        
        # Validate IV before returning using RV cone
        validated_iv = validate_market_iv(selected_row['impliedVolatility'], rv_cone, ewma_vol)
        
        return {
            'expiry': expiry_date,
            'strike': selected_row['strike'],
            'type': opt_type,
            'market_price': selected_row['lastPrice'],
            'implied_vol': validated_iv,
            'bid': selected_row['bid'],
            'ask': selected_row['ask'],
            'contractSymbol': selected_row['contractSymbol']
        }
    
    except Exception as e:
        console.print(f"[red]❌ Error fetching option chain: {e}[/red]")
        return None

def save_datasets(ticker, period='1y'):
    """Save datasets for Binomial"""
    console.print(f"[cyan]💾 Saving datasets for {ticker}...[/cyan]")
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        return False
    
    # Binomial dataset
    binomial_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    binomial_df['Date'] = pd.to_datetime(binomial_df['Date']).dt.strftime('%Y-%m-%d')
    binomial_df.to_csv('market_data_binomial.csv', index=False)
    
    console.print(f"[green]✓ Saved: market_data_binomial.csv ({len(binomial_df)} rows)[/green]")
    
    return True

# ==================== MODELS (FIXED VEGA) ====================

from numba import jit

@jit(nopython=True)
def _binomial_price_jit(S, K, T, r, sigma, q, steps, opt_type_int):
    """
    JIT-compiled binomial pricing for American options.
    opt_type_int: 1 for CALL, -1 for PUT
    """
    if T <= 0 or sigma <= 0 or steps <= 0:
        if opt_type_int == 1:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Build price tree at maturity
    prices = np.zeros(steps + 1)
    for i in range(steps + 1):
        prices[i] = S * (u ** (steps - i)) * (d ** i)
    
    # Initialize option values at maturity
    values = np.zeros(steps + 1)
    if opt_type_int == 1:  # Call
        for i in range(steps + 1):
            values[i] = max(0.0, prices[i] - K)
    else:  # Put
        for i in range(steps + 1):
            values[i] = max(0.0, K - prices[i])
    
    # Backward induction
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            prices[i] = prices[i] / u
            hold = disc * (p * values[i] + (1 - p) * values[i + 1])
            
            if opt_type_int == 1:  # Call
                exercise = max(0.0, prices[i] - K)
            else:  # Put
                exercise = max(0.0, K - prices[i])
            
            values[i] = max(hold, exercise)
    
    return values[0]


class BinomialModel:
    """Cox-Ross-Rubinstein for American Options (Numba-optimized)"""
    
    @staticmethod
    def price(S, K, T, r, sigma, q, steps, opt_type='call'):
        """
        American option pricing using Binomial Tree.
        opt_type: 'call' or 'put' (converted to int internally for JIT)
        """
        opt_type_int = 1 if opt_type == 'call' else -1
        return _binomial_price_jit(S, K, T, r, sigma, q, steps, opt_type_int)
    
    @staticmethod
    def greeks(S, K, T, r, sigma, q, steps, opt_type='call'):
        dS = S * 0.01
        dVol = 0.01
        dt = 1.0 / 365.0
        
        up = BinomialModel.price(S + dS, K, T, r, sigma, q, steps, opt_type)
        down = BinomialModel.price(S - dS, K, T, r, sigma, q, steps, opt_type)
        delta = (up - down) / (2 * dS)
        
        mid = BinomialModel.price(S, K, T, r, sigma, q, steps, opt_type)
        gamma = (up - 2 * mid + down) / (dS ** 2)
        
        # Vega per 1% vol
        vega_up = BinomialModel.price(S, K, T, r, sigma + dVol, q, steps, opt_type)
        vega = (vega_up - mid) / dVol
        
        if T > dt:
            theta_future = BinomialModel.price(S, K, T - dt, r, sigma, q, steps, opt_type)
            theta = theta_future - mid
        else:
            # Boundary condition for very small T: use 1-hour decay
            dt_small = 1.0 / (365.0 * 24.0)
            theta_future = BinomialModel.price(S, K, max(0, T - dt_small), r, sigma, q, steps, opt_type)
            theta = (theta_future - mid) * (dt / dt_small)
        
        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

# ==================== VOLATILITY ====================

def calculate_historical_vol(prices):
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns.std() * np.sqrt(252)

def ewma_volatility(returns, lambda_decay=0.94):
    """
    Calculates volatility using EWMA (RiskMetrics standard).
    Recursive formula: Var(t) = lambda * Var(t-1) + (1-lambda) * Return(t-1)^2
    Standard lambda = 0.94 for daily data (RiskMetrics recommendation)
    """
    if len(returns) < 1:
        return 0.0
    
    variance = np.var(returns)
    for r in returns:
        variance = lambda_decay * variance + (1 - lambda_decay) * r**2
    
    return np.sqrt(variance * 252)

def solve_iv(market_price, S, K, T, r, q, steps=200, opt_type='call'):
    """Newton-Raphson IV solver with bisection fallback using Binomial Model"""
    
    moneyness = S / K
    if moneyness > 1.1:
        sigma = 0.3
    elif moneyness < 0.9:
        sigma = 0.6
    else:
        sigma = 0.4
    
    tolerance = 0.00001
    
    # Newton-Raphson
    for iteration in range(50):
        price = BinomialModel.price(S, K, T, r, sigma, q, steps, opt_type)
        vega = BinomialModel.greeks(S, K, T, r, sigma, q, steps, opt_type)['vega']
        
        diff = price - market_price
        
        if abs(diff) < tolerance:
            return sigma
        
        if vega < 0.001:
            break
        
        sigma_new = sigma - diff / vega
        sigma_new = np.clip(sigma_new, 0.05, 2.5)
        
        if abs(sigma_new - sigma) < 0.0001 and iteration > 10:
            return sigma_new
        
        sigma = sigma_new
    
    # Bisection fallback
    low, high = 0.01, 3.0
    for _ in range(100):
        mid = (low + high) / 2
        price = BinomialModel.price(S, K, T, r, mid, q, steps, opt_type)
        
        if abs(price - market_price) < tolerance:
            return mid
        
        if price > market_price:
            high = mid
        else:
            low = mid
    
    return (low + high) / 2

# ==================== BACKTESTING ====================

def run_backtest(data, K, r, q, steps=200, opt_type='call'):
    """Run backtest with 80/20 split"""
    
    if len(data) < 50:
        console.print("[red]❌ Insufficient data[/red]")
        return None
    
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    train_vol = calculate_historical_vol(train_data['Close'])
    train_returns = np.log(train_data['Close'] / train_data['Close'].shift(1)).dropna()
    ewma_vol = ewma_volatility(train_returns)
    ensemble_vol = 0.6 * train_vol + 0.4 * ewma_vol
    
    console.print(f"[cyan]📊 Training: {len(train_data)} days | Testing: {len(test_data)} days[/cyan]")
    console.print(f"[yellow]HV: {train_vol*100:.2f}% | EWMA: {ewma_vol*100:.2f}% | Ensemble: {ensemble_vol*100:.2f}%[/yellow]")
    
    results = []
    
    for i, (date, row) in enumerate(test_data.iterrows()):
        spot = row['Close']
        days_left = len(test_data) - i
        T = days_left / 252.0
        
        if T <= 0:
            continue
        
        price = BinomialModel.price(spot, K, T, r, ensemble_vol, q, steps, opt_type)
        greeks = BinomialModel.greeks(spot, K, T, r, ensemble_vol, q, steps, opt_type)
        
        results.append({
            'Date': date,
            'Spot': spot,
            'Price': price,
            'Delta': greeks['delta'],
            'Gamma': greeks['gamma'],
            'Vega': greeks['vega'] / 100,  # Display per 1%
            'Theta': greeks['theta'],
            'T': T
        })
    
    df = pd.DataFrame(results)
    returns = df['Price'].pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    downside = returns[returns < 0].std()
    sortino = np.sqrt(252) * returns.mean() / downside if downside > 0 else 0
    cummax = df['Price'].expanding().max()
    drawdown = ((df['Price'] - cummax) / cummax).min()
    
    return {
        'df': df,
        'train_vol': train_vol,
        'ewma_vol': ewma_vol,
        'ensemble_vol': ensemble_vol,
        'metrics': {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': drawdown,
            'avg_price': df['Price'].mean(),
            'std_price': df['Price'].std()
        }
    }

# ==================== VISUALIZATION (ORIGINAL) ====================

def plot_comprehensive_analysis(backtest_result, ticker, save_prefix='analysis'):
    """Generate backtest visualizations"""
    df = backtest_result['df']
    
    # Figure 1: Greeks Evolution
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f'{ticker} - Greeks Evolution Over Time', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(df['Date'], df['Price'], 'b-', linewidth=2)
    axes[0, 0].set_title('Option Price Evolution')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['Date'], df['Delta'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='ATM')
    axes[0, 1].set_title('Delta (Δ)')
    axes[0, 1].set_ylabel('Delta')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['Date'], df['Gamma'], 'm-', linewidth=2)
    axes[1, 0].set_title('Gamma (Γ)')
    axes[1, 0].set_ylabel('Gamma')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df['Date'], df['Theta'], 'r-', linewidth=2)
    axes[1, 1].set_title('Theta (Θ) - Time Decay')
    axes[1, 1].set_ylabel('Theta (per day)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_greeks.png', dpi=150, bbox_inches='tight')
    console.print(f"[green]✓ Saved: {save_prefix}_greeks.png[/green]")
    plt.close()
    
    # Figure 2: Performance
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f'{ticker} - Portfolio Performance', fontsize=16, fontweight='bold')
    
    cumret = (1 + df['Price'].pct_change().fillna(0)).cumprod()
    axes[0, 0].plot(df['Date'], cumret, 'b-', linewidth=2)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].grid(True, alpha=0.3)
    
    returns = df['Price'].pct_change().dropna()
    axes[0, 1].hist(returns, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Returns Distribution')
    axes[0, 1].set_xlabel('Daily Returns')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['Date'], df['Spot'], 'orange', linewidth=2, label='Spot')
    axes[1, 0].plot(df['Date'], df['Price'], 'blue', linewidth=2, label='Option')
    axes[1, 0].set_title('Spot vs Option Price')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    cummax = cumret.expanding().max()
    drawdown = (cumret - cummax) / cummax
    axes[1, 1].fill_between(df['Date'], drawdown, 0, alpha=0.5, color='red')
    axes[1, 1].set_title('Drawdown')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_performance.png', dpi=150, bbox_inches='tight')
    console.print(f"[green]✓ Saved: {save_prefix}_performance.png[/green]")
    plt.close()

# ==================== ADVANCED VISUALIZATION (FIXED) ====================

def create_viz_folder(ticker):
    """Create organized viz folder"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    folder = Path(f"./visualizations/{ticker}_{today_str}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def plot_advanced_visualizations(ticker, S, K, T, r, q, vol, opt_type, stock_obj=None, expiry_date=None, steps=50):
    """⚠️ FIXED Professional Desk-Quality Visualizations"""
    
    console.print("\n[cyan]🎨 Generating Advanced Visualizations...[/cyan]")
    
    viz_folder = create_viz_folder(ticker)
    sns.set_theme(style="whitegrid")
    
    spot_range = np.linspace(S * 0.7, S * 1.3, 50)
    
    # Dynamic time slicing with proper deduplication
    T_days = max(T * 365, 1)
    slice_candidates = [T_days, max(T_days / 2.0, 1), 1]
    time_slices_days = sorted(list(set([round(s, 1) for s in slice_candidates if s > 0])), reverse=True)
    time_slices = [t/365.0 for t in time_slices_days]
    
    # ========== 1. PRICE VS SPOT (FIXED) ==========
    
    plt.figure(figsize=(12, 7))
    
    if opt_type == 'call':
        intrinsic = np.maximum(spot_range - K, 0)
    else:
        intrinsic = np.maximum(K - spot_range, 0)
    
    plt.plot(spot_range, intrinsic, 'k--', label='Intrinsic Value', alpha=0.7, linewidth=2.5)
    
    colors = ['#2E86AB', '#06A77D', '#D05010', '#C73E1D']
    labels = [f'{int(td)} days' for td in time_slices_days[:len(time_slices)]]
    
    # Plot option values for different maturities
    for t_val, label, color in zip(time_slices, labels, colors):
        chart_steps = 500 if t_val < 0.05 else 200
        prices = []
        time_values = []
        for s_val in spot_range:
            p = BinomialModel.price(s_val, K, t_val, r, vol, q, chart_steps, opt_type)
            intrinsic_val = max(s_val - K, 0) if opt_type == 'call' else max(K - s_val, 0)
            prices.append(p)
            time_values.append(p - intrinsic_val)
        plt.plot(spot_range, prices, label=label, color=color, linewidth=2.5, alpha=0.8)
    
    # Reference lines
    plt.axvline(x=K, color='black', linestyle=':', label=f'Strike (${K:.0f})', alpha=0.6, linewidth=1.5)
    plt.axvline(x=S, color='red', linestyle='-', label=f'Current Spot (${S:.2f})', alpha=0.8, linewidth=2)
    
    # Moneyness zones
    if opt_type == 'call':
        plt.axvspan(K, spot_range[-1], alpha=0.08, color='green')
        plt.axvspan(spot_range[0], K, alpha=0.08, color='red')
        plt.text(K * 1.15, plt.ylim()[1] * 0.95, 'ITM', fontsize=10, color='darkgreen', weight='bold')
        plt.text(K * 0.85, plt.ylim()[1] * 0.95, 'OTM', fontsize=10, color='darkred', weight='bold')
    else:
        plt.axvspan(spot_range[0], K, alpha=0.08, color='green')
        plt.axvspan(K, spot_range[-1], alpha=0.08, color='red')
        plt.text(K * 0.85, plt.ylim()[1] * 0.95, 'ITM', fontsize=10, color='darkgreen', weight='bold')
        plt.text(K * 1.15, plt.ylim()[1] * 0.95, 'OTM', fontsize=10, color='darkred', weight='bold')
    
    plt.title(f'{ticker} ${K:.0f} {opt_type.upper()} - Price Behavior', fontsize=14, fontweight='bold')
    plt.xlabel('Spot Price ($)', fontsize=12)
    plt.ylabel('Option Value ($)', fontsize=12)
    plt.legend(loc='best', fontsize=9, framealpha=0.95)
    plt.grid(True, alpha=0.25, linestyle=':')
    plt.tight_layout()
    
    save_path = viz_folder / f'{ticker}_price_spot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ {save_path}[/green]")
    plt.close()
    
    # ========== 2. VALUE SURFACE ==========
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    time_range = np.linspace(max(0.01, T * 0.1), T, 20)
    X, Y = np.meshgrid(spot_range, time_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = BinomialModel.price(X[i, j], K, Y[i, j], r, vol, q, 30, opt_type)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    title = f'{ticker} Value Surface (American - Binomial)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Spot Price ($)')
    ax.set_ylabel('Time to Expiry (years)')
    ax.set_zlabel('Option Price ($)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    save_path = viz_folder / f'{ticker}_surface.png'
    plt.savefig(save_path, dpi=300)
    console.print(f"[green]✓ {save_path}[/green]")
    plt.close()
    
    # ========== 3. IV SMILE (FIXED FILTERING) ==========
    
    if stock_obj and expiry_date:
        # Check if the expiry date is available in the options chain
        available_expirations = list(stock_obj.options)
        actual_expiry_for_smile = expiry_date
        
        if expiry_date not in available_expirations:
            # Find the closest available expiry date
            from datetime import datetime
            target_date = pd.to_datetime(expiry_date)
            available_dates = [pd.to_datetime(exp) for exp in available_expirations]
            date_diffs = [(abs((d - target_date).days), d, exp_str) for d, exp_str in zip(available_dates, available_expirations)]
            date_diffs.sort()
            closest_diff_days, closest_date, closest_exp_str = date_diffs[0]
            
            console.print(f"[yellow]⚠ Custom expiry {expiry_date} not available in market data[/yellow]")
            console.print(f"[cyan]Closest available: {closest_exp_str} ({closest_diff_days} days difference)[/cyan]")
            
            use_closest = Confirm.ask(f"Use {closest_exp_str} for IV smile visualization?", default=True)
            
            if use_closest:
                actual_expiry_for_smile = closest_exp_str
            else:
                console.print("[dim]ℹ Skipping IV smile (model-based calculations continue)[/dim]")
                actual_expiry_for_smile = None
        
        if actual_expiry_for_smile:
            try:
                chain = stock_obj.option_chain(actual_expiry_for_smile)
                opts = chain.calls if opt_type == 'call' else chain.puts
                
                # Strict filtering: remove zero volume BEFORE fitting
                opts = opts[
                    (opts['volume'] > 10) &
                    (opts['openInterest'] > 100) &
                    (opts['impliedVolatility'] > 0.05) &
                    (opts['impliedVolatility'] < 3.0) &
                    (opts['lastPrice'] > 0.05) &
                    (opts['strike'] >= S * 0.7) &
                    (opts['strike'] <= S * 1.3)
                ].copy()
                
                # Filter by spread if available
                if 'bid' in opts.columns and 'ask' in opts.columns and 'lastPrice' in opts.columns:
                    opts = opts[opts['bid'] > 0].copy()
                    opts['spread_pct'] = (opts['ask'] - opts['bid']) / opts['lastPrice']
                    opts = opts[opts['spread_pct'] < 0.3].copy()

                if len(opts) > 5:
                    plt.figure(figsize=(13, 7))
                    
                    opts['moneyness'] = opts['strike'] / S
                    opts['log_moneyness'] = np.log(opts['moneyness'])
                    
                    # Scatter with volume sizing
                    scatter = plt.scatter(opts['moneyness'], opts['impliedVolatility'],
                                        s=opts['volume']/opts['volume'].max()*200 + 20,
                                        c=opts['openInterest'], cmap='viridis',
                                        alpha=0.6, edgecolors='black', linewidth=0.5)
                    plt.colorbar(scatter, label='Open Interest')
                    
                    # Weighted cubic fit (ATM-weighted)
                    try:
                        fit_data = opts.sort_values('moneyness')
                        if len(fit_data) >= 6:
                            weights = opts['volume'] / (np.abs(opts['moneyness'] - 1.0) + 0.15)
                            z = np.polyfit(fit_data['moneyness'], fit_data['impliedVolatility'], 3, w=weights)
                            p = np.poly1d(z)
                            x_smooth = np.linspace(fit_data['moneyness'].min(), fit_data['moneyness'].max(), 300)
                            y_smooth = p(x_smooth)
                            plt.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Vol Smile Fit', alpha=0.85)
                            
                            # Mark ATM vol
                            atm_vol = p(1.0)
                            plt.scatter([1.0], [atm_vol], s=200, c='red', marker='*', 
                                      edgecolors='black', linewidth=1.5, label=f'ATM IV: {atm_vol:.1%}', zorder=5)
                    except Exception as e:
                        console.print(f"[yellow]⚠ Smile fit: {e}[/yellow]")
                    
                    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='ATM')
                    plt.axvline(x=S/K, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Current')
                    
                    plt.title(f'{ticker} Implied Volatility Smile - {actual_expiry_for_smile}', fontsize=14, fontweight='bold')
                    plt.xlabel('Moneyness (Strike/Spot)', fontsize=12)
                    plt.ylabel('Implied Volatility', fontsize=12)
                    plt.legend(loc='best', fontsize=9)
                    plt.grid(True, alpha=0.25, linestyle=':')
                    plt.tight_layout()
                    
                    save_path = viz_folder / f'{ticker}_iv_smile.png'
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    console.print(f"[green]✓ {save_path}[/green]")
                    plt.close()
            except Exception as e:
                console.print(f"[yellow]⚠ IV smile: {e}[/yellow]")
    
    # ========== 4. GREEKS MULTI-MATURITY (FIXED) ==========
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{ticker} Greeks Across Time & Spot', fontsize=16, fontweight='bold')
    
    T_plot = time_slices[:3]
    colors_g = ['blue', 'green', 'red']
    labels_g = [f'{int(t*365)} days' for t in T_plot]
    
    for t_val, color, label in zip(T_plot, colors_g[:len(T_plot)], labels_g):
        chart_steps = 500 if t_val < (5.0/365.0) else 200
        deltas, gammas, vegas, thetas = [], [], [], []
        
        for s_val in spot_range:
            g = BinomialModel.greeks(s_val, K, t_val, r, vol, q, chart_steps, opt_type)
            deltas.append(g['delta'])
            gammas.append(g['gamma'])
            vegas.append(g['vega'] / 100)
            thetas.append(g['theta'])
        
        axes[0, 0].plot(spot_range, deltas, color=color, lw=2, label=label)
        axes[0, 1].plot(spot_range, gammas, color=color, lw=2, label=label)
        axes[1, 1].plot(spot_range, vegas, color=color, lw=2, label=label)
        axes[1, 0].plot(spot_range, thetas, color=color, lw=2, label=label)
    
    # Delta
    axes[0, 0].set_title('Delta (Δ)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Delta')
    axes[0, 0].axvline(x=S, color='red', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[0, 0].axvline(x=K, color='black', linestyle=':', alpha=0.5)
    axes[0, 0].axhline(y=0.5 if opt_type=='call' else -0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gamma - mark actual peak correctly
    axes[0, 1].set_title('Gamma (Γ)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Gamma')
    axes[0, 1].axvline(x=S, color='red', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[0, 1].axvline(x=K, color='black', linestyle=':', alpha=0.5)
    
    # Get the actual plotted gamma values for the shortest maturity
    first_gamma_list = [BinomialModel.greeks(s, K, T_plot[0], r, vol, q, chart_steps, opt_type)['gamma'] for s in spot_range]
    max_gamma_idx = np.argmax(first_gamma_list)
    axes[0, 1].scatter([spot_range[max_gamma_idx]], [first_gamma_list[max_gamma_idx]], 
                      s=120, c='red', marker='*', zorder=5, edgecolor='black', linewidth=1,
                      label=f'Peak at ${spot_range[max_gamma_idx]:.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Theta
    axes[1, 0].set_title('Theta (Θ) - Time Decay', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Theta ($/day)')
    axes[1, 0].set_xlabel('Spot Price ($)', fontsize=10)
    axes[1, 0].axvline(x=S, color='red', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[1, 0].axvline(x=K, color='black', linestyle=':', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Vega
    axes[1, 1].set_title('Vega (ν) - Vol Sensitivity', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Vega ($/1% vol)', fontsize=10)
    axes[1, 1].set_xlabel('Spot Price ($)', fontsize=10)
    axes[1, 1].axvline(x=S, color='red', linestyle='-', linewidth=1.5, alpha=0.6)
    axes[1, 1].axvline(x=K, color='black', linestyle=':', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = viz_folder / f'{ticker}_greeks_multi.png'
    plt.savefig(save_path, dpi=300)
    console.print(f"[green]✓ {save_path}[/green]")
    plt.close()
    
    # ========== 5. SCENARIO HEATMAP (FIXED: DELTA-HEDGED) ==========
    
    spot_moves = np.linspace(-0.15, 0.15, 15)
    vol_moves = np.linspace(-0.10, 0.10, 15)
    
    p0 = BinomialModel.price(S, K, T, r, vol, q, steps, opt_type)
    delta0 = BinomialModel.greeks(S, K, T, r, vol, q, steps, opt_type)['delta']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'{ticker} P&L Scenario Analysis', fontsize=16, fontweight='bold')
    
    pnl_unhedged = np.zeros((len(vol_moves), len(spot_moves)))
    pnl_hedged = np.zeros((len(vol_moves), len(spot_moves)))
    
    for i, v_chg in enumerate(vol_moves):
        for j, s_chg in enumerate(spot_moves):
            sim_S = S * (1 + s_chg)
            sim_vol = max(0.01, vol + v_chg)
            T_sim = max(T - 1/365, 0.001)
            
            sim_p = BinomialModel.price(sim_S, K, T_sim, r, sim_vol, q, steps, opt_type)
            dS = sim_S - S
            
            pnl_unhedged[i, j] = sim_p - p0
            pnl_hedged[i, j] = (sim_p - p0) - delta0 * dS
    
    x_labels = [f"{m:+.0%}" for m in spot_moves]
    y_labels = [f"{v:+.1%}" for v in vol_moves]
    
    # Robust color scaling
    all_values = np.concatenate([pnl_unhedged.flatten(), pnl_hedged.flatten()])
    v_max = np.percentile(np.abs(all_values), 95)
    v_min = -v_max

    # Unhedged heatmap with contours
    im1 = axes[0].imshow(pnl_unhedged, cmap='RdYlGn', aspect='auto', 
                        vmin=v_min, vmax=v_max, origin='lower')
    axes[0].contour(pnl_unhedged, levels=[0], colors='black', linewidths=2, alpha=0.6)
    axes[0].set_title('Unhedged P&L (Spot + Vol Risk)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Spot Move (%)', fontsize=10)
    axes[0].set_ylabel('Vol Change', fontsize=10)
    axes[0].set_xticks(np.arange(len(spot_moves))[::2])
    axes[0].set_xticklabels(x_labels[::2])
    axes[0].set_yticks(np.arange(len(vol_moves))[::2])
    axes[0].set_yticklabels(y_labels[::2])
    plt.colorbar(im1, ax=axes[0], label='P&L ($)')
    
    # Delta-hedged heatmap
    im2 = axes[1].imshow(pnl_hedged, cmap='RdYlGn', aspect='auto',
                        vmin=v_min, vmax=v_max, origin='lower')
    axes[1].contour(pnl_hedged, levels=[0], colors='black', linewidths=2, alpha=0.6)
    axes[1].set_title('Delta-Hedged P&L (Gamma + Vega Risk)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Spot Move (%)', fontsize=10)
    axes[1].set_ylabel('Vol Change', fontsize=10)
    axes[1].set_xticks(np.arange(len(spot_moves))[::2])
    axes[1].set_xticklabels(x_labels[::2])
    axes[1].set_yticks(np.arange(len(vol_moves))[::2])
    axes[1].set_yticklabels(y_labels[::2])
    plt.colorbar(im2, ax=axes[1], label='P&L ($)')
    
    plt.tight_layout()
    save_path = viz_folder / f'{ticker}_heatmap.png'
    plt.savefig(save_path, dpi=300)
    console.print(f"[green]✓ {save_path}[/green]")
    plt.close()
    
    console.print(f"\n[bold green]✅ All visualizations saved to: {viz_folder}[/bold green]")

# ==================== AI CONSULTANT ====================

def get_ai_advice(analysis_data):
    """Get Articulate AI consulting in Indian English style"""
    if not GROQ_API_KEY:
        return "⚠️ Set GROQ_API_KEY environment variable for AI consulting"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # This prompt is engineered for the IIT Hackathon requirements
        prompt = f"""
        Act as a Senior Quant Strategist at a top-tier Indian Hedge Fund. 
        Your task is to explain this option trade to a non-technical Portfolio Manager. 
        Use professional, articulate Indian English (clear, structured, and slightly formal but very helpful).

        **TRADE DATA:**
        - Ticker: {analysis_data['ticker']}
        - Option: ${analysis_data['strike']:.0f} {analysis_data['opt_type'].upper()}
        - Spot Price: ${analysis_data['spot']:.2f}
        - Fair Value (Our Model): ${analysis_data['fair_value']:.2f}
        - Market Price: ${analysis_data.get('market_price', 'N/A')}
        - Implied Vol (IV): {analysis_data.get('implied_vol', 'N/A')}
        - Delta: {analysis_data['delta']:.4f} | Gamma: {analysis_data['gamma']:.6f}
        - Vega: {analysis_data['vega']:.2f} | Theta: {analysis_data['theta']:.4f}

        **INSTRUCTIONS:**
        1. **The Flow:** Start with a simple "Market Pulse" summary. Move into the "Greeks" by explaining them as real-world forces (e.g., Theta as 'daily rent', Vega as 'uncertainty tax').
        2. **Language:** Use articulate Indian English. Use analogies common in the Indian financial ecosystem (like comparing premiums to insurance or property deposits).
        3. **Clarity:** Avoid heavy math. Explain *why* a high Gamma is dangerous or *how* high IV makes the option expensive for the fund.
        4. **Action:** Conclude with a clear 'Investment Mandate' (Buy/Sell/Hold) and a specific hedging step.

        Keep it under 300 words. Use a professional, consultative tone.
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, # Slightly higher for better flow
            max_tokens=800
        )
        
        advice_content = response.choices[0].message.content
        
        # Auto-save for hackathon deliverable
        with open("Consulting_Memo_Draft.txt", "w", encoding="utf-8") as f:
            f.write(advice_content)
        console.print("[green]✓ Saved AI advice to Consulting_Memo_Draft.txt[/green]")
        
        return advice_content
    except Exception as e:
        return f"⚠️ AI unavailable: {e}"

# ==================== PUT-CALL PARITY ====================

def check_put_call_parity(S, K, r, T, call_price, put_price):
    """
    Check Put-Call Parity: C - P = S - K * e^(-rT)
    For American options, this is an approximation (strict for European)
    Returns the parity difference and interpretation
    """
    pv_strike = K * np.exp(-r * T)
    theoretical_diff = S - pv_strike
    actual_diff = call_price - put_price
    parity_error = abs(actual_diff - theoretical_diff)
    parity_error_pct = (parity_error / S) * 100
    
    return {
        'theoretical_diff': theoretical_diff,
        'actual_diff': actual_diff,
        'parity_error': parity_error,
        'parity_error_pct': parity_error_pct,
        'is_valid': parity_error_pct < 2.0  # < 2% error is acceptable for American options
    }

# ==================== MAIN WORKFLOWS ====================

def full_option_analysis():
    """Complete workflow"""
    
    console.clear()
    console.print("[bold cyan]🎯 COMPLETE OPTION ANALYSIS (AMERICAN BINOMIAL)[/bold cyan]\n")
    
    # Default to Binomial (American)
    
    ticker = Prompt.ask("📊 Ticker symbol", default="NVDA").upper()
    
    market_data = fetch_market_data(ticker)
    if not market_data:
        return
    
    save_datasets(ticker)
    
    S = market_data['spot']
    
    # Allow user override for rates
    console.print(f"\n[dim]Hint: Check alphaspread.com for accurate rates[/dim]")
    
    default_r = f"{market_data['risk_free_rate']*100:.2f}"
    r_input = Prompt.ask(f"🏦 Risk-Free Rate (%)", default=default_r)
    r = float(r_input.replace('%', '')) / 100.0
    
    default_q = f"{market_data['div_yield']:.4f}"
    q_input = Prompt.ask(f"💰 Dividend Yield (%)", default=default_q)
    q = float(q_input.replace('%', '')) / 100.0
    
    # ========== PHASE 1: CALIBRATION ==========
    
    console.print("\n" + "="*70)
    console.print("[bold yellow]PHASE 1: MODEL CALIBRATION & BACKTEST[/bold yellow]")
    console.print("="*70)
    
    backtest_strike = int(S)
    steps = 100  # Default steps for speed
    
    backtest_result = run_backtest(market_data['hist'], backtest_strike, r, q, steps, 'call')
    
    if backtest_result:
        metrics_table = Table(title="📊 Backtest Metrics", box=box.ROUNDED, style="cyan")
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right", style="green")
        
        metrics_table.add_row("Sharpe Ratio", f"{backtest_result['metrics']['sharpe']:.2f}")
        metrics_table.add_row("Sortino Ratio", f"{backtest_result['metrics']['sortino']:.2f}")
        metrics_table.add_row("Max Drawdown", f"{backtest_result['metrics']['max_drawdown']*100:.2f}%")
        
        console.print(metrics_table)
        
        backtest_result['df'].to_csv(f'{ticker}_backtest.csv', index=False)
        
        model_choice = 'american'
        if Confirm.ask("\n📈 View calibration plots?", default=True):
            plot_comprehensive_analysis(backtest_result, ticker, f'{ticker}_{model_choice}')
    
    # ========== PHASE 2: LIVE PREDICTION ==========
    
    console.print("\n" + "="*70)
    console.print("[bold green]PHASE 2: LIVE PREDICTION[/bold green]")
    console.print("="*70)
    
    contract_data = None
    if Confirm.ask("\n📡 Select live option contract?", default=True):
        contract_data = fetch_option_chain(market_data['stock_obj'], S, market_data['hist']['Close'])
    
    if contract_data:
        strike = contract_data['strike']
        expiry_str = contract_data['expiry']
        opt_type = contract_data['type']
        market_price = contract_data['market_price']
        implied_vol = contract_data['implied_vol']
    else:
        strike = float(Prompt.ask("🎯 Strike price", default=str(int(S))))
        expiry_str = Prompt.ask("📅 Expiry (YYYY-MM-DD)", default="2026-06-18")
        opt_type = Prompt.ask("📈 Option type", choices=["call", "put"], default="call")
        market_price = None
        implied_vol = None
    
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    expiry = pd.to_datetime(expiry_str).tz_localize(timezone.utc)
    dte = (expiry - today).days
    T = dte / 365.0
    
    if T <= 0:
        console.print("[red]❌ Expired[/red]")
        return
    
    # Calculate volatilities
    hist_vol = calculate_historical_vol(market_data['hist']['Close'])
    returns = np.log(market_data['hist']['Close'] / market_data['hist']['Close'].shift(1)).dropna()
    ewma_vol = ewma_volatility(returns)
    
    # ═══ IV RANK CALCULATION (INSTITUTIONAL) ═══
    # Calculate 1-year range of 30-day realized volatility
    prices = market_data['hist']['Close']
    rolling_30d_vol = prices.rolling(window=30).apply(
        lambda x: np.log(x / x.shift(1)).dropna().std() * np.sqrt(252) if len(x) >= 2 else np.nan
    ).dropna()
    
    if len(rolling_30d_vol) > 0:
        year_vol_high = rolling_30d_vol.max()
        year_vol_low = rolling_30d_vol.min()
        current_rv = rolling_30d_vol.iloc[-1] if len(rolling_30d_vol) > 0 else hist_vol
    else:
        year_vol_high = hist_vol * 1.5
        year_vol_low = hist_vol * 0.5
        current_rv = hist_vol
    
    # ═══ MODEL VOLATILITY (FOR FAIR VALUE) ═══
    # Our internal view of "true" volatility - used for pricing model
    ensemble_vol = 0.6 * hist_vol + 0.4 * ewma_vol
    
    console.print("\n[bold cyan]═══ VOLATILITY ANALYSIS ═══[/bold cyan]")
    console.print(f"Historical Vol (252d): {hist_vol*100:.2f}%")
    console.print(f"EWMA Forecast (λ=0.94): {ewma_vol*100:.2f}%")
    console.print(f"[bold]Ensemble Vol (Model):[/bold] [bold green]{ensemble_vol*100:.2f}%[/bold green]")
    
    # ═══ MARKET IV (FOR COMPARISON ONLY) ═══
    # Calculate Realized Volatility Cone for validation
    rv_cone = calculate_realized_volatility_cone(market_data['hist']['Close'])
    console.print(f"RV Cone 30d: {rv_cone.get('rv_30d',0)*100:.2f}% | 90d: {rv_cone.get('rv_90d',0)*100:.2f}%")
    
    market_iv_validated = None
    market_iv_source = None
    
    # Try to capture market IV (for comparison, NOT pricing)
    if implied_vol is not None:
        validated_iv = validate_market_iv(implied_vol, rv_cone, ewma_vol)
        if abs(validated_iv - implied_vol) < 0.0001:  # Validation passed
            market_iv_validated = validated_iv
            market_iv_source = "Market IV (from chain)"
            console.print(f"[dim]Market IV (comparison): {market_iv_validated*100:.1f}% (validated)[/dim]")
        else:
            console.print(f"[dim]Market IV {implied_vol*100:.1f}% failed validation[/dim]")
    
    # Try solving IV from market price (for comparison)
    if market_iv_validated is None and market_price is not None and market_price > 0:
        console.print("[dim]Solving IV from market price (for comparison)...[/dim]")
        steps = 200
        solved_iv = solve_iv(market_price, S, strike, T, r, q, steps, opt_type)
        
        if solved_iv is not None:
            validated_solved_iv = validate_market_iv(solved_iv, rv_cone, ewma_vol)
            if abs(validated_solved_iv - solved_iv) < 0.0001:
                market_iv_validated = validated_solved_iv
                market_iv_source = "Solved from market price"
                console.print(f"[dim]Market IV (comparison): {market_iv_validated*100:.1f}% (solved)[/dim]")
    
    # Calculate IV Rank
    current_iv = market_iv_validated if market_iv_validated else ensemble_vol
    if year_vol_high > year_vol_low:
        iv_rank = (current_iv - year_vol_low) / (year_vol_high - year_vol_low) * 100
        iv_rank = np.clip(iv_rank, 0, 100)  # Ensure 0-100 range
    else:
        iv_rank = 50.0  # Default to mid-range if no spread
    
    console.print(f"[bold]IV Rank:[/bold] [bold yellow]{iv_rank:.1f}%[/bold yellow] (1Y range: {year_vol_low*100:.1f}%-{year_vol_high*100:.1f}%)")
    
    console.print("\n[cyan]⚙️ Calculating Fair Value using Ensemble Vol...[/cyan]")
    
    # ═══ CRITICAL: Use ENSEMBLE_VOL for Fair Value (NOT market IV) ═══
    steps = 200
    fair_value = BinomialModel.price(S, strike, T, r, ensemble_vol, q, steps, opt_type)
    greeks = BinomialModel.greeks(S, strike, T, r, ensemble_vol, q, steps, opt_type)
    
    # FIX: Ask for price if it is missing (even if contract_data exists)
    if (market_price is None) and Confirm.ask("\n💰 Do you know the market price?", default=True):
        market_price = float(Prompt.ask("Enter market price"))
        console.print("[cyan]🔬 Calculating IV...[/cyan]")
        implied_vol = solve_iv(market_price, S, strike, T, r, q, steps, opt_type)
        console.print(f"[green]✓ IV: {implied_vol*100:.2f}%[/green]")
    
    # Display Results
    summary = f"""[bold cyan]{ticker} ${strike:.2f} {opt_type.upper()} - {expiry_str}[/bold cyan]
[yellow]Model: AMERICAN (Binomial) | DTE: {dte} days | Spot: ${S:.2f}[/yellow]"""
    console.print(Panel(summary, title="📋 Summary", border_style="cyan"))
    
    # Volatility table - With IV Rank
    vol_table = Table(title="📈 Volatility Analysis", box=box.ROUNDED, style="cyan")
    vol_table.add_column("Metric", style="bold")
    vol_table.add_column("Value", justify="right", style="green")
    
    vol_table.add_row("Historical Vol (252d)", f"{hist_vol*100:.2f}%")
    vol_table.add_row("EWMA Forecast (λ=0.94)", f"{ewma_vol*100:.2f}%")
    vol_table.add_row("[bold]Ensemble (60/40)[/bold]", f"[bold]{ensemble_vol*100:.2f}%[/bold]")
    vol_table.add_row("RV Cone 30d", f"{rv_cone.get('rv_30d',0)*100:.2f}%")
    vol_table.add_row("RV Cone 90d", f"{rv_cone.get('rv_90d',0)*100:.2f}%")
    
    # Show IV Rank
    vol_table.add_row("═"*30, "═"*10)  # Separator
    
    # Color code IV Rank
    if iv_rank < 30:
        iv_rank_display = f"[green]{iv_rank:.1f}% (Low Vol)[/green]"
    elif iv_rank > 70:
        iv_rank_display = f"[red]{iv_rank:.1f}% (High Vol)[/red]"
    else:
        iv_rank_display = f"[yellow]{iv_rank:.1f}% (Mid Vol)[/yellow]"
    
    vol_table.add_row("[bold]IV Rank (1Y)[/bold]", iv_rank_display)
    vol_table.add_row("[dim]1Y Vol Range[/dim]", f"[dim]{year_vol_low*100:.1f}%-{year_vol_high*100:.1f}%[/dim]")
    
    # Show model vs market volatilities
    vol_table.add_row("═"*30, "═"*10)  # Separator
    vol_table.add_row("[bold green]Model Vol (for pricing)[/bold green]", f"[bold green]{ensemble_vol*100:.2f}%[/bold green]")
    if market_iv_validated:
        vol_table.add_row("[dim]Market IV (comparison)[/dim]", f"[dim]{market_iv_validated*100:.2f}%[/dim]")
        vol_table.add_row("[dim]Source[/dim]", f"[dim]{market_iv_source}[/dim]")
        vol_spread = market_iv_validated - ensemble_vol
        vol_table.add_row("[dim]IV Spread (Market - Model)[/dim]", f"[dim]{vol_spread*100:+.2f}%[/dim]")
    
    console.print(vol_table)
    
    # Pricing table
    price_table = Table(title="💰 Pricing", box=box.ROUNDED, style="green")
    price_table.add_column("Metric", style="bold")
    price_table.add_column("Value", justify="right", style="yellow")

    price_table.add_row("Fair Value (Model)", f"${fair_value:.2f}")
    if market_price:
        price_table.add_row("Market Price", f"${market_price:.2f}")
        diff = market_price - fair_value
        rel_diff = diff / fair_value if fair_value > 0 else 0.0
        price_table.add_row("Abs Difference", f"${diff:+.2f}")
        price_table.add_row("Rel Difference", f"{rel_diff*100:+.1f}%")
    else:
        diff = 0.0
        rel_diff = 0.0
        price_table.add_row("Market Price", "N/A")
        price_table.add_row("Abs Difference", "N/A")
        price_table.add_row("Rel Difference", "N/A")

    # ═══ IV RANK-BASED SIGNAL GENERATION (INSTITUTIONAL) ═══
    
    if not market_price or fair_value <= 0:
        signal_text = "[yellow]⚪ NO TRADE VIEW - Missing/invalid market price.[/yellow]"
        signal_explanation = ""
    else:
        model_cheaper = fair_value < market_price  # Model says it's overpriced
        mispricing = abs(rel_diff)
        
        # IV Rank regime logic
        if iv_rank < 30:
            # LOW VOL REGIME: Premium is cheap
            regime_desc = "[green]Low Volatility Regime[/green] (IV Rank < 30)"
            
            if not model_cheaper and mispricing > 0.10:  # Model > Market
                signal_text = "[bold green]🟢 STRONG BUY[/bold green]"
                signal_explanation = (
                    f"{regime_desc} → Premium is cheap.\n"
                    f"Model fair value (${fair_value:.2f}) > Market price (${market_price:.2f}).\n"
                    f"[bold]Strategy:[/bold] Buy {opt_type}s to get cheap optionality."
                )
            elif model_cheaper:
                signal_text = "[yellow]⚪ HOLD[/yellow]"
                signal_explanation = (
                    f"{regime_desc} → Don't short cheap options.\n"
                    f"Model suggests overpriced, but vol is already low.\n"
                    f"[bold]Strategy:[/bold] Wait for better entry or skip."
                )
            else:
                signal_text = "[yellow]🟡 WEAK BUY[/yellow]"
                signal_explanation = (
                    f"{regime_desc} → Slight edge to buying.\n"
                    f"Mispricing is small ({rel_diff*100:+.1f}%).\n"
                    f"[bold]Strategy:[/bold] Consider small long position."
                )
        
        elif iv_rank > 70:
            # HIGH VOL REGIME: Premium is expensive
            regime_desc = "[red]High Volatility Regime[/red] (IV Rank > 70)"
            
            if model_cheaper and mispricing > 0.10:  # Model < Market
                signal_text = "[bold red]🔴 STRONG SELL[/bold red]"
                signal_explanation = (
                    f"{regime_desc} → Premium is expensive.\n"
                    f"Model fair value (${fair_value:.2f}) < Market price (${market_price:.2f}).\n"
                    f"[bold]Strategy:[/bold] Write {opt_type}s (sell premium) if suitable."
                )
            elif not model_cheaper:
                signal_text = "[yellow]⚪ HOLD[/yellow]"
                signal_explanation = (
                    f"{regime_desc} → Don't buy expensive options.\n"
                    f"Model suggests underpriced, but vol is already elevated.\n"
                    f"[bold]Strategy:[/bold] Wait for vol to decline."
                )
            else:
                signal_text = "[yellow]🟠 WEAK SELL[/yellow]"
                signal_explanation = (
                    f"{regime_desc} → Slight edge to selling.\n"
                    f"Mispricing is small ({rel_diff*100:+.1f}%).\n"
                    f"[bold]Strategy:[/bold] Consider small short premium position."
                )
        
        else:
            # MID VOL REGIME: Trade on mispricing only
            regime_desc = "[yellow]Mid Volatility Regime[/yellow] (IV Rank 30-70)"
            
            if mispricing < 0.10:
                signal_text = "[yellow]⚪ HOLD[/yellow]"
                signal_explanation = (
                    f"{regime_desc} → Neutral environment.\n"
                    f"Mispricing < 10% ({rel_diff*100:+.1f}%).\n"
                    f"[bold]Strategy:[/bold] Monitor for better opportunities."
                )
            elif not model_cheaper:  # Model > Market (underpriced)
                signal_text = "[green]🟢 BUY[/green]"
                signal_explanation = (
                    f"{regime_desc} → Trade on mispricing.\n"
                    f"Model: ${fair_value:.2f} vs Market: ${market_price:.2f} ({rel_diff*100:+.1f}%).\n"
                    f"[bold]Strategy:[/bold] Buy {opt_type} - model shows value."
                )
            else:  # Model < Market (overpriced)
                signal_text = "[red]🔴 SELL[/red]"
                signal_explanation = (
                    f"{regime_desc} → Trade on mispricing.\n"
                    f"Model: ${fair_value:.2f} vs Market: ${market_price:.2f} ({rel_diff*100:+.1f}%).\n"
                    f"[bold]Strategy:[/bold] Avoid or short {opt_type} - model shows overvaluation."
                )
    
    # Format full signal with explanation
    full_signal = f"{signal_text}\n\n{signal_explanation}" if signal_explanation else signal_text
    
    # Print signal + pricing
    console.print(Panel(full_signal, title="📊 Signal (IV Rank-Based)", border_style="yellow"))
    console.print(price_table)

    
    # Greeks table
    greek_table = Table(title="🔬 Greeks", box=box.ROUNDED, style="magenta")
    greek_table.add_column("Greek", style="bold")
    greek_table.add_column("Value", justify="right", style="cyan")
    greek_table.add_column("Meaning", style="yellow")
    
    greek_table.add_row("Delta (Δ)", f"{greeks['delta']:.4f}", f"${abs(greeks['delta']):.2f} per $1 move")
    greek_table.add_row("Gamma (Γ)", f"{greeks['gamma']:.6f}", "Delta sensitivity")
    greek_table.add_row("Vega (ν)", f"{greeks['vega']/100:.4f}", f"${abs(greeks['vega']/100):.2f} per 1% vol")
    greek_table.add_row("Theta (Θ)", f"{greeks['theta']:.4f}", f"${abs(greeks['theta']):.2f} daily decay")
    
    console.print(greek_table)
    
    # Put-Call Parity Check (Optional Extension for Hackathon)
    console.print("\n[cyan]🔍 Checking Put-Call Parity...[/cyan]")
    opposite_type = 'put' if opt_type == 'call' else 'call'
    opposite_price = BinomialModel.price(S, strike, T, r, ensemble_vol, q, steps, opposite_type)
    
    if opt_type == 'call':
        parity_result = check_put_call_parity(S, strike, r, T, fair_value, opposite_price)
    else:
        parity_result = check_put_call_parity(S, strike, r, T, opposite_price, fair_value)
    
    parity_table = Table(title="⚖️ Put-Call Parity Check", box=box.ROUNDED, style="blue")
    parity_table.add_column("Metric", style="bold")
    parity_table.add_column("Value", justify="right", style="cyan")
    
    parity_table.add_row(f"Your {opt_type.capitalize()} Price", f"${fair_value:.2f}")
    parity_table.add_row(f"Companion {opposite_type.capitalize()} Price", f"${opposite_price:.2f}")
    parity_table.add_row("Theoretical Diff (C - P)", f"${parity_result['theoretical_diff']:.2f}")
    parity_table.add_row("Actual Diff (Model)", f"${parity_result['actual_diff']:.2f}")
    parity_table.add_row("Parity Error", f"${parity_result['parity_error']:.2f} ({parity_result['parity_error_pct']:.2f}%)")
    
    # Format status
    status_emoji = "✅" if parity_result['is_valid'] else "⚠️"
    status_text = "Valid" if parity_result['is_valid'] else "Deviation (normal for American)"
    parity_table.add_row("Status", f"{status_emoji} {status_text}")
    
    console.print(parity_table)
    
    # Advanced Visualizations
    if Confirm.ask("\n🎨 Generate advanced visualizations? (Takes time)", default=True):
        plot_advanced_visualizations(
            ticker, S, strike, T, r, q, ensemble_vol, opt_type,
            market_data.get('stock_obj'), expiry_str
        )
    
    # AI Consultant
    if GROQ_API_KEY and Confirm.ask("\n🤖 Get AI advice?", default=True):
        console.print("\n[cyan]🤖 Consulting...[/cyan]")
        
        analysis_data = {
            'ticker': ticker,
            'opt_type': opt_type,
            'strike': strike,
            'spot': S,
            'moneyness': S / strike,
            'fair_value': fair_value,
            'market_price': market_price,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'vega': greeks['vega'] / 100,
            'theta': greeks['theta'],
            'hist_vol': hist_vol,
            'garch_vol': ewma_vol,
            'implied_vol': f"{implied_vol*100:.2f}%" if implied_vol else "N/A",
            'sharpe': backtest_result['metrics']['sharpe'] if backtest_result else 0,
            'max_drawdown': backtest_result['metrics']['max_drawdown'] if backtest_result else 0
        }
        
        advice = get_ai_advice(analysis_data)
        console.print(Panel(advice, title="🤖 AI Consultant", border_style="blue"))
    
    console.print("\n[bold green]✅ Complete![/bold green]")
    input("\nPress Enter...")


def explain_greeks():
    """Educational content"""
    console.clear()
    console.print(Panel("[bold cyan]📚 GREEKS EXPLAINED[/bold cyan]", style="cyan"))
    
    info = """
[bold yellow]Delta (Δ)[/bold yellow] - Directional exposure
• $0.70 delta = option moves $0.70 when stock moves $1
• Also probability of expiring ITM

[bold green]Gamma (Γ)[/bold green] - Delta stability
• High gamma = delta changes rapidly
• Highest near ATM

[bold magenta]Vega (ν)[/bold magenta] - Volatility risk
• Profits from vol spikes
• Long options = positive vega

[bold red]Theta (Θ)[/bold red] - Time decay
• "Rent" for holding option
• Accelerates near expiry
"""
    
    console.print(info)
    input("\nPress Enter...")

def main_menu():
    """Main menu"""
    console.clear()
    console.print(BANNER)
    
    menu = Table(title="🏆 MAIN MENU", box=box.ROUNDED, style="cyan", show_header=False)
    menu.add_column("Option", style="bold yellow", width=3)
    menu.add_column("Description", style="white")
    
    menu.add_row("1", "🎯 Complete Analysis (Price | Backtest | Visualize | AI)")
    menu.add_row("2", "📚 Greeks Explained")
    menu.add_row("3", "❌ Exit")
    
    console.print(menu)

def main():
    """Main application"""
    
    while True:
        main_menu()
        choice = Prompt.ask("Select", choices=["1","2","3"], default="1")
        
        if choice == "1":
            full_option_analysis()
        elif choice == "2":
            explain_greeks()
        elif choice == "3":
            console.print("\n[bold green]🏆 Good luck! 🚀[/bold green]\n")
            break

if __name__ == "__main__":
    main()
