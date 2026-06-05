"""
shap_regression.py
------------------
Models dynamic SHAP importance values as a function of macro state variables.

Research question:
    Are factor premia (as captured by XGBoost SHAP values) state-dependent?
    Does CMA importance rise when credit spreads widen?
    Does momentum importance rise with VIX?

Method:
    For each feature f, regress annual SHAP importance on lagged macro variables:
        SHAP_{f,t} = alpha_f + beta_f * Macro_{t-1} + epsilon_{f,t}

    Uses Newey-West HAC standard errors to account for serial correlation
    in the annual SHAP series (T=15 is small, so results are suggestive
    rather than definitive).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


# ── OLS with Newey-West SEs ────────────────────────────────────────────────────

def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 2):
    """
    OLS with Newey-West heteroskedasticity and autocorrelation consistent
    standard errors.

    Parameters
    ----------
    y    : (T,) dependent variable
    X    : (T, K) regressors (including constant)
    lags : number of lags for NW kernel

    Returns
    -------
    dict with keys: coef, se, tstat, pval, r2
    """
    T, K = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta    = XtX_inv @ X.T @ y
    resid   = y - X @ beta

    # Newey-West covariance
    S = np.zeros((K, K))
    for t in range(T):
        xt = X[t:t+1].T
        S += resid[t]**2 * (xt @ xt.T)

    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        for t in range(lag, T):
            xt  = X[t:t+1].T
            xtl = X[t-lag:t-lag+1].T
            gamma = resid[t] * resid[t-lag] * (xt @ xtl.T + xtl @ xt.T)
            S += weight * gamma

    V    = XtX_inv @ S @ XtX_inv
    se   = np.sqrt(np.diag(V))
    tstat = beta / se
    pval  = 2 * (1 - stats.t.cdf(np.abs(tstat), df=T - K))

    ybar = y.mean()
    r2   = 1 - np.sum(resid**2) / np.sum((y - ybar)**2)

    return {"coef": beta, "se": se, "tstat": tstat, "pval": pval, "r2": r2}


# ── SHAP regression ───────────────────────────────────────────────────────────

def run_shap_regression(
    shap_df: pd.DataFrame,
    macro: pd.DataFrame,
    macro_cols: list[str] | None = None,
    lags: int = 2,
) -> pd.DataFrame:
    """
    Regress each feature's annual SHAP importance on contemporaneous
    and lagged macro variables.

    Parameters
    ----------
    shap_df   : DataFrame (year x feature) of normalised SHAP values
    macro     : DataFrame (Date x macro_var) of monthly macro variables
    macro_cols: which macro variables to include (default: all)
    lags      : Newey-West lags

    Returns
    -------
    results_df : DataFrame with (feature, macro_var) MultiIndex and
                 columns [coef, se, tstat, pval, r2]
    """
    if macro_cols is None:
        macro_cols = [c for c in ["term_spread", "credit_spread", "vix", "inflation"]
                      if c in macro.columns]

    # Annual macro: average each macro variable within each calendar year
    macro_annual = macro.copy()
    macro_annual.index = pd.to_datetime(macro_annual.index)
    macro_annual = macro_annual.resample("YE").mean()
    macro_annual.index = macro_annual.index.year

    # Align on common years
    common_years = shap_df.index.intersection(macro_annual.index)
    shap_aligned = shap_df.loc[common_years]
    macro_aligned = macro_annual.loc[common_years, macro_cols]

    # Standardise macro variables
    macro_std = (macro_aligned - macro_aligned.mean()) / macro_aligned.std().clip(lower=1e-8)

    rows = []
    for feature in shap_aligned.columns:
        y = shap_aligned[feature].values

        for mcol in macro_cols:
            X = np.column_stack([np.ones(len(y)), macro_std[mcol].values])
            res = newey_west_ols(y, X, lags=lags)

            rows.append({
                "Feature":    feature,
                "Macro var":  mcol,
                "Coef":       round(res["coef"][1], 4),
                "SE":         round(res["se"][1], 4),
                "t-stat":     round(res["tstat"][1], 3),
                "p-value":    round(res["pval"][1], 3),
                "R²":         round(res["r2"], 3),
                "Sig":        "***" if res["pval"][1] < 0.01
                              else "**" if res["pval"][1] < 0.05
                              else "*"  if res["pval"][1] < 0.10
                              else "",
            })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_shap_vs_macro(
    shap_df: pd.DataFrame,
    macro: pd.DataFrame,
    top_features: list[str] | None = None,
    save_path: str = "results/figures/shap_vs_macro.png",
):
    """
    Scatter plots of SHAP importance vs macro variables for key features.
    Each panel shows one (feature, macro_var) pair with OLS fit line.
    """
    if top_features is None:
        # Default to the three features with most variance in SHAP
        top_features = shap_df.std().nlargest(3).index.tolist()

    macro_cols = [c for c in ["term_spread", "credit_spread", "vix", "inflation"]
                  if c in macro.columns]

    # Annual macro
    macro_annual = macro.resample("YE").mean()
    macro_annual.index = macro_annual.index.year
    common_years = shap_df.index.intersection(macro_annual.index)
    shap_a = shap_df.loc[common_years]
    macro_a = macro_annual.loc[common_years, macro_cols]

    n_features = len(top_features)
    n_macro    = len(macro_cols)
    fig, axes  = plt.subplots(
        n_features, n_macro,
        figsize=(4 * n_macro, 3.5 * n_features),
        squeeze=False,
    )

    macro_labels = {
        "term_spread":   "Term spread (%)",
        "credit_spread": "Credit spread (%)",
        "vix":           "VIX",
        "inflation":     "Inflation (%)",
    }

    for i, feat in enumerate(top_features):
        for j, mcol in enumerate(macro_cols):
            ax = axes[i][j]
            x = macro_a[mcol].values
            y = shap_a[feat].values

            ax.scatter(x, y, color="steelblue", s=40, alpha=0.8, zorder=3)

            # Annotate years
            for yr, xi, yi in zip(common_years, x, y):
                ax.annotate(str(int(yr))[-2:], (xi, yi),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points", color="gray")

            # OLS fit line
            if len(x) > 3:
                m, b, r, p, _ = stats.linregress(x, y)
                xfit = np.linspace(x.min(), x.max(), 100)
                color = "firebrick" if p < 0.10 else "gray"
                ls    = "-" if p < 0.10 else "--"
                ax.plot(xfit, m * xfit + b, color=color, linewidth=1.5,
                        linestyle=ls, label=f"p={p:.2f}, r={r:.2f}")
                ax.legend(fontsize=7, loc="best")

            ax.set_xlabel(macro_labels.get(mcol, mcol), fontsize=9)
            ax.set_ylabel(f"SHAP importance\n({feat})", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("SHAP feature importance vs macro state variables",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_time_series(
    shap_df: pd.DataFrame,
    macro: pd.DataFrame,
    feature: str = "beta_CMA",
    macro_col: str = "credit_spread",
    save_path: str = "results/figures/shap_macro_dual.png",
):
    """
    Dual-axis plot: SHAP importance (left) vs macro variable (right)
    over time, for a single (feature, macro_var) pair.
    """
    macro_annual = macro.resample("YE").mean()
    macro_annual.index = macro_annual.index.year
    common_years = shap_df.index.intersection(macro_annual.index)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    years = common_years.astype(int)
    shap_vals  = shap_df.loc[common_years, feature].values
    macro_vals = macro_annual.loc[common_years, macro_col].values

    ax1.plot(years, shap_vals, color="steelblue", linewidth=2,
             marker="o", markersize=5, label=f"SHAP: {feature}")
    ax2.plot(years, macro_vals, color="firebrick", linewidth=1.5,
             linestyle="--", marker="s", markersize=4, alpha=0.8,
             label=macro_col.replace("_", " ").title())

    ax1.set_ylabel(f"SHAP importance ({feature})", color="steelblue", fontsize=10)
    ax2.set_ylabel(macro_col.replace("_", " ").title(), color="firebrick", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="firebrick")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title(f"SHAP importance ({feature}) vs {macro_col.replace('_',' ')}",
                  fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.macro import load_macro
    from src.data import load_data
    from src.features import build_feature_matrix
    from src.models import OLS_FF5, XGBModel, walk_forward
    from src.evaluation import compute_dynamic_shap

    print("Loading data...")
    factors, returns = load_data(use_synthetic=True)
    macro = load_macro(use_synthetic=True)
    X, y = build_feature_matrix(returns, factors, macro=macro)
    feature_names = list(X.columns)

    fwd_raw    = returns.shift(-1)
    raw_panel  = fwd_raw.stack(future_stack=True).rename("raw_return")
    raw_panel.index.names = ["Date", "Asset"]
    raw_panel  = raw_panel.reindex(y.index).dropna()
    common_idx = y.index.intersection(raw_panel.index)
    y_al, X_al, raw_al = y.loc[common_idx], X.loc[common_idx], raw_panel.loc[common_idx]

    print("Running walk-forward (XGBoost only)...")
    results = walk_forward(X_al, y_al, [XGBModel()], raw_returns=raw_al,
                           train_years=5, val_years=1, test_years=1)

    shap_df = compute_dynamic_shap(
        results["XGBoost"]["shap_inputs"], feature_names
    )
    print("\nSHAP df:\n", shap_df.round(3))

    print("\nRunning SHAP regression...")
    reg = run_shap_regression(shap_df, macro)
    print(reg[reg["p-value"] < 0.15].to_string())

    plot_shap_vs_macro(shap_df, macro)
    plot_shap_time_series(shap_df, macro, feature="beta_CMA",
                          macro_col="credit_spread")
    print("\nDone.")
