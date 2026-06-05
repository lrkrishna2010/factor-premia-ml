"""
generate_notebooks.py
---------------------
Generates all four project notebooks as .ipynb files.
Run once from the project root:
    python generate_notebooks.py
"""

import nbformat as nbf
from pathlib import Path

Path("notebooks").mkdir(exist_ok=True)

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    return n

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)


# ── 01_data.ipynb ─────────────────────────────────────────────────────────────

nb01 = nb([
    md("# 01 — Data\nLoad factors and returns, inspect the dataset, check for missing values and basic statistics."),

    code("""\
import sys, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})
"""),

    md("## Load data\nSet `use_synthetic=False` once `data/raw/` is populated by `fetch_data.py`."),

    code("""\
from src.data import load_data

factors, returns = load_data(use_synthetic=False)   # change to True for synthetic

print(f"Factors : {factors.shape}  {factors.index[0].date()} → {factors.index[-1].date()}")
print(f"Returns : {returns.shape}")
print()
print("Factor columns:", list(factors.columns))
print("Asset  columns:", list(returns.columns))
"""),

    md("## Factor summary statistics"),

    code("""\
factors.describe().round(4)
"""),

    md("## Returns summary statistics"),

    code("""\
returns.describe().round(4)
"""),

    md("## Missing values"),

    code("""\
missing = returns.isna().sum()
print("Tickers with missing months:")
print(missing[missing > 0].sort_values(ascending=False))
print(f"\\nTotal missing: {missing.sum()} / {returns.size:,} ({missing.sum()/returns.size*100:.2f}%)")
"""),

    md("## Factor time series"),

    code("""\
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
factor_cols = [c for c in factors.columns if c != 'RF']

for ax, col in zip(axes.flatten(), factor_cols):
    ax.plot(factors.index, factors[col], linewidth=0.8, color='steelblue')
    ax.axhline(0, linewidth=0.5, color='black', linestyle='--', alpha=0.4)
    ax.set_title(col, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.suptitle('Fama–French Factor Returns (Monthly)', fontsize=13)
plt.tight_layout()
plt.savefig('../results/figures/factor_series.png', bbox_inches='tight')
plt.show()
"""),

    md("## Return correlation heatmap"),

    code("""\
import numpy as np

corr = returns.corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
plt.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Cross-asset return correlation matrix', fontsize=12)
plt.tight_layout()
plt.savefig('../results/figures/return_corr.png', bbox_inches='tight')
plt.show()
print(f"Mean pairwise correlation: {corr.values[np.triu_indices(len(corr), k=1)].mean():.3f}")
"""),

    md("## Distribution of monthly returns"),

    code("""\
all_rets = returns.stack().dropna()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(all_rets, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
axes[0].set_title('Distribution of monthly returns (all assets)')
axes[0].set_xlabel('Monthly return')
axes[0].set_ylabel('Count')

axes[1].plot(returns.mean(axis=1).rolling(12).mean() * 12, label='Mean (ann.)', color='steelblue')
axes[1].fill_between(returns.index,
                     (returns.mean(axis=1) - returns.std(axis=1)).rolling(12).mean() * 12,
                     (returns.mean(axis=1) + returns.std(axis=1)).rolling(12).mean() * 12,
                     alpha=0.2, color='steelblue')
axes[1].axhline(0, linewidth=0.6, linestyle='--', color='black', alpha=0.4)
axes[1].set_title('Cross-sectional mean return (12-month rolling, ann.)')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].legend()

plt.tight_layout()
plt.savefig('../results/figures/return_dist.png', bbox_inches='tight')
plt.show()

print(f"Mean monthly return : {all_rets.mean():.4f}  ({all_rets.mean()*12:.3f} ann.)")
print(f"Std monthly return  : {all_rets.std():.4f}  ({all_rets.std()*12**0.5:.3f} ann.)")
print(f"Skewness            : {all_rets.skew():.3f}")
print(f"Kurtosis            : {all_rets.kurt():.3f}")
"""),
])


# ── 02_models.ipynb ───────────────────────────────────────────────────────────

nb02 = nb([
    md("# 02 — Models\nBuild the feature matrix and run walk-forward validation across all four models."),

    code("""\
import sys, warnings, time
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})
"""),

    md("## Build feature matrix"),

    code("""\
from src.data import load_data
from src.features import build_feature_matrix

factors, returns = load_data(use_synthetic=False)
X, y = build_feature_matrix(returns, factors, beta_window=36)

print("Feature matrix:", X.shape)
print("Features:", list(X.columns))
print("\\nSample:")
X.head(3)
"""),

    md("## Feature distributions\nAll features are cross-sectionally rank-normalised to [−0.5, +0.5] at each date."),

    code("""\
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax, col in zip(axes.flatten(), X.columns):
    ax.hist(X[col].dropna(), bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel('')
fig.suptitle('Feature distributions (cross-sectionally rank-normalised)', fontsize=12)
plt.tight_layout()
plt.savefig('../results/figures/feature_dists.png', bbox_inches='tight')
plt.show()
"""),

    md("## Walk-forward validation\n\nSchema: 5-year train → 1-year validation (hyperparameter tuning) → 1-year test. Roll forward 1 year at a time.\n\n> ⏱ This takes ~2–5 minutes depending on your machine."),

    code("""\
from src.models import OLS_FF5, LassoModel, RFModel, XGBModel, walk_forward

# Build raw returns panel for portfolio construction
fwd_raw   = returns.shift(-1)
raw_panel = fwd_raw.stack(future_stack=True).rename('raw_return')
raw_panel.index.names = ['Date', 'Asset']
raw_panel  = raw_panel.reindex(y.index).dropna()
common_idx = y.index.intersection(raw_panel.index)
y_aligned, X_aligned, raw_aligned = y.loc[common_idx], X.loc[common_idx], raw_panel.loc[common_idx]

models = [OLS_FF5(), LassoModel(), RFModel(), XGBModel()]

t0 = time.time()
results = walk_forward(X_aligned, y_aligned, models,
                       raw_returns=raw_aligned,
                       train_years=5, val_years=1, test_years=1)
print(f"\\nCompleted in {time.time()-t0:.0f}s")
"""),

    md("## Save results"),

    code("""\
import pickle
with open('../results/walk_forward_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'X': X_aligned, 'y': y_aligned,
                 'raw_aligned': raw_aligned, 'feature_names': list(X.columns)}, f)
print("Saved to results/walk_forward_results.pkl")
"""),
])


# ── 03_evaluation.ipynb ───────────────────────────────────────────────────────

nb03 = nb([
    md("# 03 — Evaluation\nStatistical and economic evaluation of all models. Load walk-forward results and compute IC, portfolio metrics, and cumulative returns."),

    code("""\
import sys, warnings, pickle
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})

with open('../results/walk_forward_results.pkl', 'rb') as f:
    saved = pickle.load(f)

results      = saved['results']
feature_names = saved['feature_names']
print("Models loaded:", list(results.keys()))
"""),

    md("## Statistical metrics"),

    code("""\
from src.evaluation import statistical_summary

stats = statistical_summary(results)
print(stats.to_string())
stats.to_csv('../results/statistical_metrics.csv')
"""),

    md("## IC time series"),

    code("""\
from src.evaluation import compute_ic_series

COLOURS = {
    'OLS (FF5+Mom+STRev)': '#2166ac',
    'Lasso':               '#4dac26',
    'Random Forest':       '#d01c8b',
    'XGBoost':             '#f1a340',
}

fig, ax = plt.subplots(figsize=(12, 4))
for name, res in results.items():
    ic = compute_ic_series(res['predictions'], res['actuals'])
    ic_roll = ic.rolling(12, min_periods=6).mean()
    ax.plot(ic_roll.index, ic_roll.values, label=name,
            color=COLOURS.get(name), linewidth=1.8)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_title('12-month rolling information coefficient (IC) by model', fontsize=12)
ax.set_ylabel('IC (Spearman rank correlation)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('../results/figures/ic_over_time.png', bbox_inches='tight')
plt.show()
"""),

    md("## Portfolio construction and metrics"),

    code("""\
from src.evaluation import build_long_short_portfolio, portfolio_summary

port_returns   = {}
port_summaries = {}

for name, res in results.items():
    port = build_long_short_portfolio(res['predictions'], res['raw_returns'])
    port_returns[name]   = port
    port_summaries[name] = portfolio_summary(port)
    for leg in ['Gross', 'Net']:
        s = port_summaries[name][leg]
        print(f"{name} [{leg}]  Sharpe={s['Sharpe']:+.3f}  "
              f"Ret={s['Ann. Return (%)']:+.1f}%  "
              f"Vol={s['Ann. Vol (%)']:.1f}%  MaxDD={s['Max DD (%)']:.1f}%")
    print()
"""),

    md("## Cumulative returns"),

    code("""\
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
axes = axes.flatten()

for i, (name, port) in enumerate(port_returns.items()):
    ax = axes[i]
    gross_cum = (1 + port['Gross']).cumprod()
    net_cum   = (1 + port['Net']).cumprod()
    ax.plot(gross_cum.index, gross_cum.values, label='Gross',
            color=COLOURS.get(name, 'steelblue'), linewidth=1.8)
    ax.plot(net_cum.index, net_cum.values, label='Net (−10bps)',
            color=COLOURS.get(name, 'steelblue'), linewidth=1.8, linestyle='--')
    ax.axhline(1, color='black', linewidth=0.5, linestyle=':', alpha=0.4)
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.suptitle('Cumulative long-short portfolio returns (top vs bottom tertile)', fontsize=12)
plt.tight_layout()
plt.savefig('../results/figures/cumulative_returns.png', bbox_inches='tight')
plt.show()
"""),

    md("## Sharpe comparison: gross vs net"),

    code("""\
from src.evaluation import plot_sharpe_comparison
plot_sharpe_comparison(port_summaries)

# Summary table
rows = []
for name, sums in port_summaries.items():
    for leg, m in sums.items():
        rows.append({'Model': name, 'Leg': leg, **m})
port_df = pd.DataFrame(rows)
port_df.to_csv('../results/portfolio_metrics.csv', index=False)
port_df
"""),
])


# ── 04_shap.ipynb ─────────────────────────────────────────────────────────────

nb04 = nb([
    md("# 04 — Dynamic SHAP analysis\nCompute SHAP values per out-of-sample year for XGBoost and visualise how feature importance evolves."),

    code("""\
import sys, warnings, pickle
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})

with open('../results/walk_forward_results.pkl', 'rb') as f:
    saved = pickle.load(f)

results       = saved['results']
feature_names = saved['feature_names']
print("Features:", feature_names)
"""),

    md("## Compute dynamic SHAP\n\n> ⏱ SHAP computation takes ~1–2 minutes."),

    code("""\
from src.evaluation import compute_dynamic_shap

xgb_inputs = results.get('XGBoost', {}).get('shap_inputs', [])
shap_df = compute_dynamic_shap(xgb_inputs, feature_names)

print(f"SHAP computed for {len(shap_df)} years")
shap_df.round(3)
"""),

    md("## Heatmap: feature importance by year"),

    code("""\
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(shap_df.T.values, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=shap_df.values.max())

ax.set_xticks(range(len(shap_df.index)))
ax.set_xticklabels(shap_df.index.astype(int), rotation=45, ha='right')
ax.set_yticks(range(len(shap_df.columns)))
ax.set_yticklabels(shap_df.columns)
ax.set_title('Dynamic feature importance — XGBoost SHAP values (normalised)', fontsize=12)
plt.colorbar(im, ax=ax, label='Mean |SHAP| (relative)')
plt.tight_layout()
plt.savefig('../results/figures/shap_heatmap.png', bbox_inches='tight')
plt.show()
"""),

    md("## Stacked area: importance over time"),

    code("""\
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#378ADD','#1D9E75','#BA7517','#D85A30','#D4537E','#7F77DD','#888780','#5DCAA5','#F0997B']

bottom = np.zeros(len(shap_df))
for i, col in enumerate(shap_df.columns):
    vals = shap_df[col].values * 100
    ax.fill_between(shap_df.index.astype(int), bottom, bottom + vals,
                    alpha=0.85, color=colors[i % len(colors)], label=col)
    bottom += vals

ax.set_ylim(0, 100)
ax.set_ylabel('Feature importance (%)')
ax.set_title('Stacked feature importance over time (XGBoost SHAP)', fontsize=12)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('../results/figures/shap_stacked.png', bbox_inches='tight')
plt.show()
"""),

    md("## Key findings\n\nInterpret what the model learns:"),

    code("""\
print("Top feature by year:")
print(shap_df.idxmax(axis=1).to_string())

print("\\nMean importance across full sample:")
print(shap_df.mean().sort_values(ascending=False).round(3).to_string())

print("\\nChange in CMA importance (2009 → last year):")
cma_trend = shap_df['beta_CMA'].iloc[-1] - shap_df['beta_CMA'].iloc[0]
print(f"  {cma_trend:+.3f} ({cma_trend/shap_df['beta_CMA'].iloc[0]*100:+.1f}%)")
"""),

    md("## Save SHAP results"),

    code("""\
shap_df.to_csv('../results/shap_dynamic.csv')
print("Saved to results/shap_dynamic.csv")
"""),
])


# ── Write all notebooks ───────────────────────────────────────────────────────

notebooks = {
    'notebooks/01_data.ipynb':       nb01,
    'notebooks/02_models.ipynb':     nb02,
    'notebooks/03_evaluation.ipynb': nb03,
    'notebooks/04_shap.ipynb':       nb04,
}

for path, notebook in notebooks.items():
    with open(path, 'w') as f:
        nbf.write(notebook, f)
    print(f"Created: {path}")

print("\nAll notebooks created. Open with: jupyter notebook notebooks/")
