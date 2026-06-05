"""
fetch_data.py
-------------
Fetches real data from Tiingo + getFamaFrenchFactors.
Run this once to populate data/raw/ then run run.py --real-data

Usage:
    python fetch_data.py --api-key YOUR_TIINGO_KEY
"""

import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import getFamaFrenchFactors as gff

API_KEY = ""   # paste your key here, or pass via --api-key

# Tickers: S&P 500 sector ETFs + broad market — no survivorship bias
TICKERS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLU",   # Utilities
    "XLB",   # Materials
    "GLD",   # Gold
    "TLT",   # Long Treasuries
    "IEF",   # Intermediate Treasuries
    "HYG",   # High Yield
    "EEM",   # Emerging Markets
    "EFA",   # Developed International
    "VNQ",   # Real Estate
    "DIA",   # Dow Jones
    "MDY",   # Mid Cap
    "IJR",   # Small Cap
    "IVW",   # S&P 500 Growth
    "IVE",   # S&P 500 Value
    "VGT",   # Vanguard Tech
    "VHT",   # Vanguard Healthcare
    "VFH",   # Vanguard Financials
    "VDE",   # Vanguard Energy
    "VPU",   # Vanguard Utilities
    "VAW",   # Vanguard Materials
]

Path("data/raw").mkdir(parents=True, exist_ok=True)


# ── Tiingo price fetcher ──────────────────────────────────────────────────────

def fetch_tiingo_prices(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    api_key: str = API_KEY,
) -> pd.DataFrame:
    """
    Fetch monthly adjusted closing prices from Tiingo for a list of tickers.
    Returns a DataFrame with Date index and ticker columns (simple monthly returns).
    """
    all_prices = {}
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}

    for i, ticker in enumerate(tickers):
        url = (
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            f"?startDate={start}&endDate={end}&resampleFreq=monthly&token={api_key}"
        )
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: HTTP {r.status_code} — skipping")
                continue

            data = r.json()
            if not data:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: no data — skipping")
                continue

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
            df = df.set_index("date")["adjClose"].rename(ticker)
            all_prices[ticker] = df
            print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(df)} months ✓")

        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: error — {e}")
            continue

    if not all_prices:
        raise ValueError("No price data fetched — check your API key")

    prices = pd.DataFrame(all_prices)
    prices.index.name = "Date"

    # Convert prices to simple monthly returns
    returns = prices.pct_change().dropna(how="all")

    print(f"\nReturns panel: {returns.shape[0]} months × {returns.shape[1]} tickers")
    print(f"Date range: {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"Missing values: {returns.isna().sum().sum():,}")

    return returns


# ── French factors ────────────────────────────────────────────────────────────

def fetch_french_factors(start: str = "2000-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Fetch FF5 + Momentum factors via getFamaFrenchFactors."""
    print("Fetching Fama-French 5 factors...")
    ff5 = gff.famaFrench5Factor(frequency="m")
    ff5 = ff5.rename(columns={"date_ff_factors": "Date", "Mkt-RF": "Mkt_RF"})
    ff5 = ff5.set_index("Date")

    print("Fetching Momentum factor...")
    mom = gff.momentumFactor(frequency="m")
    mom = mom.rename(columns={"date_ff_factors": "Date", "MOM": "Mom"})
    mom = mom.set_index("Date")

    factors = ff5.join(mom[["Mom"]], how="inner")

    # Add ST_Rev as negative of prior month return on low-vol stocks
    # (approximation — true ST_Rev requires the French CSV)
    # For now, placeholder of zeros; swap in real CSV if available
    factors["ST_Rev"] = 0.0

    # Filter date range
    factors.index = pd.to_datetime(factors.index)
    factors = factors.loc[start:end]

    print(f"Factors: {factors.shape[0]} months, columns: {list(factors.columns)}")
    return factors


# ── Main ──────────────────────────────────────────────────────────────────────

def main(api_key: str):
    print("=" * 50)
    print("Fetching real data")
    print("=" * 50)

    # Factors
    print("\n[1/2] Fetching French factors...")
    factors = fetch_french_factors()
    factors.to_csv("data/raw/factors.csv")
    print(f"Saved: data/raw/factors.csv")
    print(factors.tail(3))

    # Returns
    print(f"\n[2/2] Fetching Tiingo prices for {len(TICKERS)} tickers...")
    returns = fetch_tiingo_prices(TICKERS, api_key=api_key)
    returns.to_csv("data/raw/returns.csv")
    print(f"Saved: data/raw/returns.csv")
    print(returns.tail(3))

    print("\nDone. Now run:")
    print("  python run.py --real-data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=API_KEY, help="Tiingo API key")
    args = parser.parse_args()
    main(api_key=args.api_key)