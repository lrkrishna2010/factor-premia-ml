"""
features.py
-----------
Constructs the feature matrix (X) for model training.

For each asset at each time t, we compute predictors using only
information available at time t (no look-ahead bias).

Features:
    Factor exposures    — rolling 12-month betas to each FF5 factor + Mom + ST_Rev
    Return-based        — prior 12-1 month momentum, prior 1-month reversal, 
                          trailing 12-month realised volatility
    Cross-sectional     — all features are cross-sectionally ranked and normalised
                          at each date (standard in the literature; removes the
                          effect of time-varying market-wide levels)
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ── Rolling beta estimation ───────────────────────────────────────────────────

def rolling_betas(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 36,
    factor_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute rolling OLS betas for each asset against each factor.

    Parameters
    ----------
    returns  : (T, N) asset returns
    factors  : (T, K) factor returns
    window   : rolling window length (months)
    factor_cols : which factor columns to use

    Returns
    -------
    dict mapping factor_name -> DataFrame(T, N) of rolling betas
    """
    if factor_cols is None:
        factor_cols = [c for c in factors.columns if c != "RF"]

    betas = {f: pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
             for f in factor_cols}

    factor_vals = factors[factor_cols].values  # (T, K)

    for t in range(window, len(returns) + 1):
        X = factor_vals[t - window: t]          # (window, K)
        Y = returns.iloc[t - window: t].values  # (window, N)

        # OLS: beta = (X'X)^{-1} X'Y
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(len(factor_cols)))
            beta = XtX_inv @ X.T @ Y             # (K, N)
        except np.linalg.LinAlgError:
            continue

        date_idx = returns.index[t - 1]
        for i, f in enumerate(factor_cols):
            betas[f].loc[date_idx] = beta[i]

    return betas


# ── Return-based features ─────────────────────────────────────────────────────

def momentum_12_1(returns: pd.DataFrame) -> pd.DataFrame:
    """Prior 12-1 month return (skip 1 month, sum 11)."""
    cum = (1 + returns).rolling(12).apply(np.prod, raw=True) - 1
    lagged = cum.shift(1)  # skip most recent month
    return lagged


def reversal_1(returns: pd.DataFrame) -> pd.DataFrame:
    """Prior 1-month return (short-term reversal signal)."""
    return returns.shift(1)


def realised_vol(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Rolling realised volatility (annualised)."""
    return returns.rolling(window).std() * np.sqrt(12)


# ── Cross-sectional standardisation ──────────────────────────────────────────

def cs_rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    At each date, rank assets cross-sectionally and normalise to [-0.5, 0.5].
    This removes the effect of time-varying factor means and scales all
    features to a common range, which helps tree models and regularised
    regressions equally.
    """
    def _rank_row(row):
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            return row
        ranked = np.full_like(row, np.nan, dtype=float)
        ranked[valid] = rankdata(row[valid]) / valid.sum() - 0.5
        return ranked

    return df.apply(_rank_row, axis=1, raw=True)


# ── Main feature builder ──────────────────────────────────────────────────────

def build_feature_matrix(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    beta_window: int = 36,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix.

    Parameters
    ----------
    returns     : (T, N) asset returns
    factors     : (T, K) factor returns
    beta_window : rolling window for beta estimation
    macro       : (T, M) macro state variables (term_spread, credit_spread,
                  vix, inflation). If provided, added as time-series features
                  that vary by date but not cross-sectionally. These are
                  standardised (z-scored) over the training period rather
                  than cross-sectionally ranked, since they have no
                  cross-sectional dimension.

    Returns
    -------
    X : DataFrame with MultiIndex (Date, Asset) and feature columns
    y : Series with MultiIndex (Date, Asset) — next-month return rank
    """
    print("Computing rolling betas...")
    # ST_Rev excluded — placeholder zeros in real data corrupt the feature
    available = [c for c in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
                 if c in factors.columns and factors[c].std() > 0]
    factor_cols = available
    betas = rolling_betas(returns, factors, window=beta_window, factor_cols=factor_cols)

    print("Computing return-based features...")
    mom   = momentum_12_1(returns)
    rev   = reversal_1(returns)
    rvol  = realised_vol(returns)

    print("Cross-sectional rank normalisation...")
    feature_frames = {}
    for name, betas_df in betas.items():
        feature_frames[f"beta_{name}"] = cs_rank_normalise(betas_df)

    feature_frames["mom_12_1"] = cs_rank_normalise(mom)
    feature_frames["rev_1"]    = cs_rank_normalise(rev)
    feature_frames["rvol_12"]  = cs_rank_normalise(rvol)

    # Macro state variables: broadcast to all assets at each date
    # These are time-series features (same value for all assets on a given date)
    # normalised over expanding window to avoid look-ahead
    if macro is not None:
        print("Adding macro state variables...")
        macro_cols = [c for c in ["term_spread", "credit_spread", "vix", "inflation"]
                      if c in macro.columns]
        for col in macro_cols:
            # Broadcast: same value for all assets at date t
            macro_broadcast = pd.DataFrame(
                np.tile(macro[col].values[:, None], (1, len(returns.columns))),
                index=macro.index,
                columns=returns.columns,
            )
            # Expanding z-score (use only past data to normalise)
            expanding_mean = macro[col].expanding().mean()
            expanding_std  = macro[col].expanding().std().clip(lower=1e-8)
            macro_z = ((macro[col] - expanding_mean) / expanding_std)
            macro_z_broadcast = pd.DataFrame(
                np.tile(macro_z.values[:, None], (1, len(returns.columns))),
                index=macro.index,
                columns=returns.columns,
            )
            feature_frames[f"macro_{col}"] = macro_z_broadcast

    # Target: next-month return, cross-sectionally ranked
    fwd_return = returns.shift(-1)
    y_ranked = cs_rank_normalise(fwd_return)

    # Stack into long format (Date, Asset)
    print("Stacking into panel format...")
    feature_dfs = []
    for name, df in feature_frames.items():
        long = df.stack(future_stack=True).rename(name)
        feature_dfs.append(long)

    X = pd.concat(feature_dfs, axis=1)
    y = y_ranked.stack(future_stack=True).rename("target")

    # Align and drop rows with any NaN
    panel = X.join(y, how="inner").dropna()

    # Drop the last month (no forward return available)
    last_date = returns.index[-1]
    panel = panel[panel.index.get_level_values("Date") < last_date]

    print(f"Feature matrix: {panel.shape[0]:,} observations, {X.shape[1]} features")
    print(f"  Dates: {panel.index.get_level_values('Date').min().date()} → "
          f"{panel.index.get_level_values('Date').max().date()}")

    X_out = panel.drop(columns=["target"])
    y_out = panel["target"]

    return X_out, y_out


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data import load_data

    factors, returns = load_data(use_synthetic=True)
    X, y = build_feature_matrix(returns, factors, beta_window=36)

    print("\nFeature matrix sample:")
    print(X.head())
    print("\nTarget distribution:")
    print(y.describe().round(4))
    print("\nFeature correlations (mean abs):", X.corr().abs().values[np.triu_indices(X.shape[1], k=1)].mean().round(3))
