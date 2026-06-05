"""
macro.py
--------
Fetches and prepares macroeconomic state variables.

Variables:
    term_spread    : 10Y - 3M Treasury yield (slope of yield curve)
    credit_spread  : Moody's BAA - AAA corporate bond yield spread
    vix            : CBOE VIX (market uncertainty)
    inflation      : Year-over-year CPI (CPIAUCSL)

Real data: fetched from FRED via fredapi or requests.
Synthetic: generated to match real statistical properties (default).

To use real data:
    1. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
    2. Run: python src/macro.py --api-key YOUR_KEY
       This saves data/raw/macro.csv
    3. load_macro(use_synthetic=False) reads from that CSV
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path

RAW = Path("data/raw")


# ── FRED fetcher ──────────────────────────────────────────────────────────────

FRED_SERIES = {
    "GS10":         "t10y",          # 10-year Treasury constant maturity
    "TB3MS":        "t3m",           # 3-month Treasury bill secondary market
    "BAA":          "baa",           # Moody's BAA corporate bond yield
    "AAA":          "aaa",           # Moody's AAA corporate bond yield
    "VIXCLS":       "vix",           # CBOE VIX close
    "CPIAUCSL":     "cpi",           # CPI all urban consumers (seasonally adj.)
}


def fetch_fred_series(series_id: str, api_key: str,
                      start: str = "2000-01-01",
                      end: str = "2024-12-31") -> pd.Series:
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
        f"&observation_start={start}&observation_end={end}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        {o["date"]: float(o["value"]) if o["value"] != "." else np.nan
         for o in obs},
        name=series_id,
    )
    s.index = pd.to_datetime(s.index)
    return s


def fetch_and_save_macro(api_key: str,
                         start: str = "2000-01-01",
                         end: str = "2024-12-31") -> pd.DataFrame:
    """Fetch all macro series from FRED and save to data/raw/macro.csv."""
    print("Fetching macro series from FRED...")
    raw = {}
    for fred_id, name in FRED_SERIES.items():
        print(f"  {fred_id}...", end=" ")
        try:
            raw[name] = fetch_fred_series(fred_id, api_key, start, end)
            print("ok")
        except Exception as e:
            print(f"failed: {e}")

    df = pd.DataFrame(raw)
    # Resample to month-end
    df = df.resample("ME").last()

    # Construct derived variables
    df["term_spread"]   = df["t10y"] - df["t3m"]
    df["credit_spread"] = df["baa"]  - df["aaa"]
    df["vix"]           = df["vix"]

    # YoY inflation from CPI level
    df["inflation"] = df["cpi"].pct_change(12) * 100

    macro = df[["term_spread", "credit_spread", "vix", "inflation"]].dropna()
    RAW.mkdir(parents=True, exist_ok=True)
    macro.to_csv(RAW / "macro.csv")
    print(f"\nSaved: data/raw/macro.csv  ({macro.shape})")
    print(macro.describe().round(3))
    return macro


# ── Synthetic data ────────────────────────────────────────────────────────────

def generate_synthetic_macro(
    start: str = "2000-01",
    end: str   = "2024-12",
    seed: int  = 99,
) -> pd.DataFrame:
    """
    Synthetic macro variables calibrated to FRED statistics (2000-2024):
        term_spread  : mean 1.52%, std 1.12%  (10Y-3M)
        credit_spread: mean 0.93%, std 0.32%  (BAA-AAA)
        vix          : mean 19.5,  std 8.2
        inflation    : mean 2.57%, std 1.84%
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="ME")
    T     = len(dates)

    # AR(1) processes to add persistence (macro variables are autocorrelated)
    def ar1(mean, std, phi, T, rng):
        x = np.zeros(T)
        x[0] = mean
        eps = rng.normal(0, std * np.sqrt(1 - phi**2), T)
        for t in range(1, T):
            x[t] = mean + phi * (x[t-1] - mean) + eps[t]
        return x

    term_spread   = ar1(1.52,  1.12, 0.92, T, rng)
    credit_spread = np.abs(ar1(0.93, 0.32, 0.95, T, rng))
    vix           = np.abs(ar1(19.5, 8.2,  0.88, T, rng))
    inflation     = ar1(2.57,  1.84, 0.96, T, rng)

    macro = pd.DataFrame({
        "term_spread":   term_spread,
        "credit_spread": credit_spread,
        "vix":           vix,
        "inflation":     inflation,
    }, index=dates)
    macro.index.name = "Date"
    return macro


# ── Main loader ───────────────────────────────────────────────────────────────

def load_macro(use_synthetic: bool = True) -> pd.DataFrame:
    """
    Load macro state variables.

    Parameters
    ----------
    use_synthetic : bool
        True  → generate synthetic data (default)
        False → load from data/raw/macro.csv (run fetch_and_save_macro first)
    """
    if not use_synthetic and (RAW / "macro.csv").exists():
        macro = pd.read_csv(RAW / "macro.csv", index_col=0, parse_dates=True)
        macro.index = macro.index + pd.offsets.MonthEnd(0)
        macro.index.name = "Date"
        print(f"  Macro: {macro.shape}  "
              f"({macro.index[0].date()} → {macro.index[-1].date()})")
        return macro.astype(float)

    if not use_synthetic:
        print("  data/raw/macro.csv not found — using synthetic macro data")

    macro = generate_synthetic_macro()
    macro.index = macro.index + pd.offsets.MonthEnd(0)
    return macro


# ── CLI for fetching real data ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="FRED API key")
    args = parser.parse_args()
    fetch_and_save_macro(api_key=args.api_key)
