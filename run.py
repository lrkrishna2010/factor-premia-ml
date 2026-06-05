"""
run.py
------
Main entry point. Runs the full pipeline:
    1. Load data
    2. Build features
    3. Walk-forward validation
    4. Evaluate: statistical + economic metrics
    5. Dynamic SHAP analysis
    6. Save all results and figures

Usage:
    python run.py                    # synthetic data (default)
    python run.py --real-data        # real data from data/raw/

To use real data:
    1. Download from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html:
         - F-F_Research_Data_5_Factors_2x3.CSV
         - F-F_Momentum_Factor.CSV
         - F-F_ST_Reversal_Factor.CSV
    2. Place in data/raw/
    3. Build a monthly returns panel (CSV: Date index, ticker columns, simple returns)
       and place at data/raw/returns.csv
    4. Run: python run.py --real-data
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
sys.path.insert(0, str(Path(__file__).parent))

from src.data       import load_data
from src.macro      import load_macro
from src.features   import build_feature_matrix
from src.models     import OLS_FF5, LassoModel, RFModel, XGBModel, walk_forward
from src.evaluation import (
    statistical_summary,
    build_long_short_portfolio,
    portfolio_summary,
    compute_dynamic_shap,
    plot_ic_over_time,
    plot_cumulative_returns,
    plot_shap_heatmap,
    plot_sharpe_comparison,
)
from src.shap_regression import (
    run_shap_regression,
    plot_shap_vs_macro,
    plot_shap_time_series,
)

Path("results/figures").mkdir(parents=True, exist_ok=True)


def main(use_synthetic: bool = True):
    t0 = time.time()
    print("=" * 60)
    print("ML vs Factor Models: Predicting Equity Returns")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    factors, returns = load_data(use_synthetic=use_synthetic)
    macro = load_macro(use_synthetic=use_synthetic)

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("\n[2/6] Building feature matrix (with macro state variables)...")
    X, y = build_feature_matrix(returns, factors, macro=macro, beta_window=36)
    feature_names = list(X.columns)
    print(f"  Features: {feature_names}")

    # ── 3. Walk-forward validation ────────────────────────────────────────────
    print("\n[3/6] Running walk-forward validation...")
    models = [OLS_FF5(), LassoModel(), RFModel(), XGBModel()]

    # Build raw returns panel aligned to X/y index (for portfolio construction)
    fwd_raw = returns.shift(-1)
    raw_panel = fwd_raw.stack(future_stack=True).rename("raw_return")
    raw_panel.index.names = ["Date", "Asset"]
    raw_panel = raw_panel.reindex(y.index).dropna()
    # Align raw_panel to y index exactly
    common_idx = y.index.intersection(raw_panel.index)
    y_aligned = y.loc[common_idx]
    X_aligned = X.loc[common_idx]
    raw_aligned = raw_panel.loc[common_idx]

    results = walk_forward(X_aligned, y_aligned, models, raw_returns=raw_aligned,
                           train_years=5, val_years=1, test_years=1)

    # Save results for notebook use
    with open("results/walk_forward_results.pkl", "wb") as f:
        pickle.dump({"results": results, "feature_names": feature_names,
                     "X": X_aligned, "y": y_aligned,
                     "raw_aligned": raw_aligned}, f)
    print("  Saved: results/walk_forward_results.pkl")

    # ── 4. Statistical evaluation ─────────────────────────────────────────────
    print("\n[4/6] Evaluating results...")
    stats = statistical_summary(results)
    print("\n── Statistical Metrics ──────────────────────────────────")
    print(stats.to_string())

    # Portfolio construction
    port_returns  = {}
    port_summaries = {}

    print("\n── Portfolio Metrics ────────────────────────────────────")
    for name, res in results.items():
        port = build_long_short_portfolio(res["predictions"], res["raw_returns"])
        port_returns[name]   = port
        port_summaries[name] = portfolio_summary(port)

        print(f"\n{name}:")
        for leg in ["Gross", "Net"]:
            s = port_summaries[name][leg]
            print(f"  {leg:5s}  Sharpe={s['Sharpe']:+.3f}  "
                  f"Ann.Ret={s['Ann. Return (%)']:+.1f}%  "
                  f"Ann.Vol={s['Ann. Vol (%)']:.1f}%  "
                  f"MaxDD={s['Max DD (%)']:.1f}%")

    # ── 5. Dynamic SHAP ───────────────────────────────────────────────────────
    print("\n[5/6] Computing dynamic SHAP values (XGBoost)...")
    xgb_inputs = results.get("XGBoost", {}).get("shap_inputs", [])
    shap_df = compute_dynamic_shap(xgb_inputs, feature_names)

    if not shap_df.empty:
        print(f"\nSHAP computed for {len(shap_df)} years")
        print(shap_df.round(3))

    # ── 6. SHAP regression on macro ───────────────────────────────────────────
    print("\n[6/6] SHAP regression on macro state variables...")
    if not shap_df.empty:
        macro_aligned_annual = macro.copy()
        macro_aligned_annual.index = pd.to_datetime(macro_aligned_annual.index)

        shap_reg = run_shap_regression(shap_df, macro_aligned_annual)
        print("\n── SHAP ~ Macro regression (p < 0.15) ──────────────────")
        sig = shap_reg[shap_reg["p-value"] < 0.15]
        if len(sig):
            print(sig.to_string(index=False))
        else:
            print("  No significant relationships at p < 0.15")
            print(shap_reg.to_string(index=False))
        shap_reg.to_csv("results/shap_macro_regression.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_ic_over_time(results)
    plot_cumulative_returns(port_returns)
    plot_sharpe_comparison(port_summaries)
    if not shap_df.empty:
        plot_shap_heatmap(shap_df)
        plot_shap_vs_macro(shap_df, macro_aligned_annual)
        plot_shap_time_series(
            shap_df, macro_aligned_annual,
            feature="beta_CMA", macro_col="credit_spread"
        )

    # ── Save tables ───────────────────────────────────────────────────────────
    stats.to_csv("results/statistical_metrics.csv")
    port_rows = []
    for model_name, sums in port_summaries.items():
        for leg, metrics in sums.items():
            row = {"Model": model_name, "Leg": leg}
            row.update(metrics)
            port_rows.append(row)
    pd.DataFrame(port_rows).to_csv("results/portfolio_metrics.csv", index=False)
    if not shap_df.empty:
        shap_df.to_csv("results/shap_dynamic.csv")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print("Results saved to results/")
    print("Figures saved to results/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", action="store_true",
                        help="Use real data from data/raw/ instead of synthetic")
    args = parser.parse_args()
    main(use_synthetic=not args.real_data)