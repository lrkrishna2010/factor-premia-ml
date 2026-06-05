"""
evaluation.py
-------------
Computes all evaluation metrics and produces the results figures.

Statistical metrics:
    OOS R²              — Coefficient of determination on out-of-sample predictions
    IC (monthly)        — Spearman rank correlation between predicted and realised ranks
    ICIR                — IC / std(IC) — signal consistency measure

Economic metrics:
    Long-short Sharpe   — Long top quintile, short bottom quintile, monthly rebalance
    Gross vs Net Sharpe — Net of 10bps round-trip transaction cost per trade

Dynamic SHAP:
    Feature importance computed per out-of-sample year for XGBoost
    Visualised as heatmap over time
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

COST_BPS = 10 / 10_000  # 10bps round-trip per trade


# ── Statistical metrics ───────────────────────────────────────────────────────

def compute_ic_series(predictions: pd.Series, actuals: pd.Series) -> pd.Series:
    """Monthly Spearman IC between predicted and realised return ranks."""
    dates = predictions.index.get_level_values("Date").unique()
    ics = {}
    for d in dates:
        pred_d = predictions[predictions.index.get_level_values("Date") == d]
        act_d  = actuals[actuals.index.get_level_values("Date") == d]
        if len(pred_d) > 5:
            ic, _ = spearmanr(pred_d.values, act_d.values)
            ics[d] = ic
    return pd.Series(ics)


def statistical_summary(results: dict) -> pd.DataFrame:
    """Return OOS R², mean IC, ICIR for all models."""
    rows = []
    for name, res in results.items():
        preds   = res["predictions"]
        actuals = res["actuals"]

        oos_r2 = r2_score(actuals, preds)
        ic_series = compute_ic_series(preds, actuals)
        mean_ic = ic_series.mean()
        icir = mean_ic / ic_series.std() if ic_series.std() > 0 else np.nan

        rows.append({
            "Model": name,
            "OOS R²": round(oos_r2, 5),
            "Mean IC": round(mean_ic, 4),
            "ICIR": round(icir, 3),
            "IC > 0 (%)": round((ic_series > 0).mean() * 100, 1),
        })

    return pd.DataFrame(rows).set_index("Model")


# ── Portfolio construction ────────────────────────────────────────────────────

def build_long_short_portfolio(
    predictions: pd.Series,
    actuals: pd.Series,
    n_quantiles: int = 3,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """
    Long top tertile, short bottom tertile, equal-weighted within each leg.
    Uses tertiles (n_quantiles=3) by default — appropriate for small universes
    like 30 ETFs where quintiles produce only 6 assets per leg.
    Returns monthly gross and net returns.
    """
    dates = predictions.index.get_level_values("Date").unique().sort_values()
    monthly = []
    prev_long, prev_short = set(), set()

    for d in dates:
        pred_d = predictions[predictions.index.get_level_values("Date") == d]
        act_d  = actuals[actuals.index.get_level_values("Date") == d]

        q = pd.qcut(pred_d.values, n_quantiles, labels=False, duplicates="drop")
        assets = pred_d.index.get_level_values("Asset") if "Asset" in pred_d.index.names \
            else pred_d.index.get_level_values(1)

        long_assets  = set(assets[q == n_quantiles - 1])
        short_assets = set(assets[q == 0])

        if not long_assets or not short_assets:
            continue

        # Realised return for each leg
        long_mask  = act_d.index.get_level_values(-1).isin(long_assets)
        short_mask = act_d.index.get_level_values(-1).isin(short_assets)
        long_ret   = act_d[long_mask].mean()
        short_ret  = act_d[short_mask].mean()
        gross_ret  = long_ret - short_ret

        # Transaction costs: turnover on changed positions
        long_turnover  = len(long_assets.symmetric_difference(prev_long)) / max(len(long_assets), 1)
        short_turnover = len(short_assets.symmetric_difference(prev_short)) / max(len(short_assets), 1)
        total_cost = (long_turnover + short_turnover) * cost_bps / 2

        net_ret = gross_ret - total_cost

        monthly.append({
            "Date": d,
            "Long": long_ret,
            "Short": short_ret,
            "Gross": gross_ret,
            "Net": net_ret,
            "Turnover": (long_turnover + short_turnover) / 2,
        })

        prev_long, prev_short = long_assets, short_assets

    return pd.DataFrame(monthly).set_index("Date")


def portfolio_summary(port_returns: pd.DataFrame, annualise: float = 12) -> dict:
    """Annualised return, vol, Sharpe, max drawdown for gross and net."""
    results = {}
    for col in ["Gross", "Net"]:
        r = port_returns[col].dropna()
        ann_ret = r.mean() * annualise
        ann_vol = r.std() * np.sqrt(annualise)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum     = (1 + r).cumprod()
        drawdown = (cum / cum.cummax() - 1).min()
        results[col] = {
            "Ann. Return (%)": round(ann_ret * 100, 2),
            "Ann. Vol (%)":    round(ann_vol * 100, 2),
            "Sharpe":          round(sharpe, 3),
            "Max DD (%)":      round(drawdown * 100, 2),
        }
    return results


# ── Dynamic SHAP ──────────────────────────────────────────────────────────────

def compute_dynamic_shap(shap_inputs: list, feature_names: list) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per year for XGBoost.

    Parameters
    ----------
    shap_inputs  : list of dicts with keys 'year', 'X_test', 'model'
    feature_names: column names of X

    Returns
    -------
    DataFrame (year x feature) of mean |SHAP| values, normalised to sum to 1 per year
    """
    try:
        import shap
    except ImportError:
        print("shap not installed — skipping SHAP analysis")
        return pd.DataFrame()

    rows = []
    for entry in shap_inputs:
        year    = entry["year"]
        X_test  = entry["X_test"]
        model   = entry["model"]

        if not hasattr(model, "model") or model.model is None:
            continue

        try:
            explainer   = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(X_test)
            mean_abs    = np.abs(shap_values).mean(axis=0)
            # Normalise to get relative importance
            mean_abs    = mean_abs / mean_abs.sum()
            rows.append(dict(zip(["year"] + feature_names, [year] + list(mean_abs))))
        except Exception as e:
            print(f"  SHAP failed for year {year}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("year").sort_index()


# ── Plotting ──────────────────────────────────────────────────────────────────

COLOURS = {
    "OLS (FF5+Mom+STRev)": "#2166ac",
    "Lasso":               "#4dac26",
    "Random Forest":       "#d01c8b",
    "XGBoost":             "#f1a340",
}


def plot_ic_over_time(results: dict, save_path: str = "results/figures/ic_over_time.png"):
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, res in results.items():
        ic = compute_ic_series(res["predictions"], res["actuals"])
        ic_roll = ic.rolling(12, min_periods=6).mean()
        ax.plot(ic_roll.index, ic_roll.values,
                label=name, color=COLOURS.get(name), linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("12-Month Rolling Information Coefficient (IC) by Model", fontsize=13)
    ax.set_ylabel("IC (Spearman rank correlation)")
    ax.set_xlabel("")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_cumulative_returns(
    port_returns: dict,
    save_path: str = "results/figures/cumulative_returns.png"
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    for i, (name, port) in enumerate(port_returns.items()):
        ax = axes[i]
        gross_cum = (1 + port["Gross"]).cumprod()
        net_cum   = (1 + port["Net"]).cumprod()

        ax.plot(gross_cum.index, gross_cum.values, label="Gross",
                color=COLOURS.get(name, "steelblue"), linewidth=1.8)
        ax.plot(net_cum.index, net_cum.values, label="Net (−10bps)",
                color=COLOURS.get(name, "steelblue"), linewidth=1.8, linestyle="--")
        ax.axhline(1, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Cumulative Long-Short Portfolio Returns (Top vs Bottom Quintile)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_heatmap(
    shap_df: pd.DataFrame,
    save_path: str = "results/figures/shap_heatmap.png"
):
    if shap_df.empty:
        print("No SHAP data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(shap_df.T.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=shap_df.values.max())

    ax.set_xticks(range(len(shap_df.index)))
    ax.set_xticklabels(shap_df.index.astype(int), rotation=45, ha="right")
    ax.set_yticks(range(len(shap_df.columns)))
    ax.set_yticklabels(shap_df.columns)
    ax.set_title("Dynamic Feature Importance (XGBoost SHAP Values, Normalised)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Feature")
    plt.colorbar(im, ax=ax, label="Mean |SHAP| (relative)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_sharpe_comparison(
    portfolio_summaries: dict,
    save_path: str = "results/figures/sharpe_comparison.png"
):
    models   = list(portfolio_summaries.keys())
    gross_sr = [portfolio_summaries[m]["Gross"]["Sharpe"] for m in models]
    net_sr   = [portfolio_summaries[m]["Net"]["Sharpe"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, gross_sr, width, label="Gross Sharpe",
                   color=[COLOURS.get(m, "steelblue") for m in models], alpha=0.9)
    bars2 = ax.bar(x + width/2, net_sr,   width, label="Net Sharpe (−10bps)",
                   color=[COLOURS.get(m, "steelblue") for m in models], alpha=0.5,
                   edgecolor="black", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("Long-Short Portfolio Sharpe Ratio: Gross vs Net of Transaction Costs", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")