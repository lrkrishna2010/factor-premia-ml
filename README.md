# Can Machine Learning Outperform Factor Models in Predicting Equity Returns?
### Evidence from a U.S. ETF Universe

**Radhesh Krishna Lalam** · University of Essex · 2026

---

## Overview

This project asks whether machine learning models can generate superior return predictions compared to classical factor models, using a clean, liquid universe of 30 U.S. sector ETFs. It is designed as an end-to-end empirical asset pricing pipeline — from data to portfolio construction to economic interpretation.

The full paper is in [`paper.tex`](paper.tex) (compile with `pdflatex`).

**Key findings:**
- XGBoost achieves mean IC of +0.017 and net Sharpe of 0.26 after 10bps transaction costs; OLS and Lasso produce near-zero or negative IC
- Random Forest achieves comparable IC to XGBoost but near-zero portfolio Sharpe — an IC-to-Sharpe disconnect attributable to prediction attenuation at the extremes
- Dynamic SHAP regression reveals that credit spread conditions predict SMB beta importance (R² = 0.63, p = 0.001) and momentum beta importance (R² = 0.55, p < 0.001)
- Macro VIX and inflation features show zero SHAP: yield curve and credit conditions subsume their information once factor betas are included

---

## Project Structure

```
ml_factor_comparison/
│
├── src/
│   ├── data.py              # Data loading (real + synthetic)
│   ├── macro.py             # Macro state variables (FRED / synthetic)
│   ├── features.py          # Feature engineering (rolling betas, momentum, vol)
│   ├── models.py            # OLS, Lasso, Random Forest, XGBoost + walk-forward engine
│   ├── evaluation.py        # IC, portfolio construction, Sharpe, SHAP heatmaps
│   └── shap_regression.py   # SHAP ~ macro OLS regression + scatter plots
│
├── notebooks/
│   ├── 01_data.ipynb        # Data inspection and descriptive statistics
│   ├── 02_models.ipynb      # Walk-forward validation (saves results pkl)
│   ├── 03_evaluation.ipynb  # Portfolio metrics and IC analysis
│   └── 04_shap.ipynb        # Dynamic SHAP and macro regression
│
├── data/
│   ├── raw/                 # factors.csv, returns.csv, macro.csv (user-supplied)
│   └── processed/           # intermediate files
│
├── results/
│   ├── figures/             # All output figures (PNG)
│   ├── statistical_metrics.csv
│   ├── portfolio_metrics.csv
│   ├── shap_dynamic.csv
│   └── shap_macro_regression.csv
│
├── run.py                   # Single entry point — runs full pipeline
├── fetch_data.py            # Pulls returns from Tiingo, factors from French library
├── generate_notebooks.py    # Generates .ipynb files
└── paper.tex                # Full LaTeX paper (32 pages)
```

---

## Methodology

### Walk-forward validation

```
|── Train (5 yrs) ──|── Val (1 yr) ──|── Test (1 yr) ──| → roll 1 year
```

- 15 non-overlapping test folds covering 2009–2023
- Hyperparameters tuned on validation fold only
- Model refit on train+val before generating test predictions
- **No look-ahead bias** — no future information enters any estimate

### Feature set (9 features)

| Feature | Description |
|---------|-------------|
| β_Mkt, β_SMB, β_HML, β_RMW, β_CMA, β_Mom | Rolling 36-month OLS betas to each FF5+Mom factor |
| Mom₁₂₋₁ | 12-1 month cumulative return (skip 1 month) |
| Rev₁ | Prior 1-month return (short-term reversal) |
| RVol₁₂ | Annualised rolling 12-month realised volatility |

Plus 4 macro state variables (term spread, credit spread, VIX, inflation) added as expanding z-scored features.

All features are cross-sectionally rank-normalised to [−0.5, +0.5] at each date.

### Models

| Model | Role |
|-------|------|
| OLS (FF5+Mom) | Linear baseline |
| Lasso | Regularised linear |
| Random Forest | Non-linear ensemble |
| XGBoost | Non-linear boosted trees (best performer) |

### Portfolio construction

Long top tertile, short bottom tertile, equal-weighted. Monthly rebalancing with 10bps round-trip transaction costs applied to turnover.

---

## Results

### Statistical metrics (2009–2023, out-of-sample)

| Model | OOS R² | Mean IC | ICIR | IC > 0 (%) |
|-------|--------|---------|------|------------|
| OLS (FF5+Mom) | −0.014 | −0.001 | −0.002 | 53.3 |
| Lasso | −0.001 | −0.020 | −0.055 | 17.8 |
| Random Forest | −0.007 | +0.007 | +0.021 | 52.2 |
| **XGBoost** | **−0.000** | **+0.017** | **+0.061** | **50.0** |

### Portfolio performance

| Model | Gross Sharpe | Net Sharpe | Ann. Return (net) | Max DD |
|-------|-------------|------------|-------------------|--------|
| OLS | +0.04 | −0.02 | −0.1% | −45.1% |
| Lasso | −0.16 | −0.20 | −1.9% | −22.6% |
| Random Forest | +0.01 | −0.06 | −0.5% | −41.4% |
| **XGBoost** | **+0.35** | **+0.26** | **+2.2%** | **−21.1%** |

### SHAP regression highlights

Credit spread → SMB beta importance: β = +0.020, p = 0.001, **R² = 0.63**

Momentum beta ~ credit spread: β = +0.030, p < 0.001, **R² = 0.55**

CMA beta ~ VIX: β = −0.006, p = 0.041 (importance falls during acute volatility shocks)

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with synthetic data (no downloads needed)

```bash
python run.py
```

### 3. Get real data

**Factors** (free, no signup):
```bash
pip install getFamaFrenchFactors lxml
# Factors are fetched automatically on first run with --real-data
```

**Returns** — sign up at [tiingo.com](https://tiingo.com) (free tier), then:
```bash
python fetch_data.py --api-key YOUR_TIINGO_KEY
```

**Macro variables** — get a free API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html), then:
```bash
python src/macro.py --api-key YOUR_FRED_KEY
```

### 4. Run with real data

```bash
python run.py --real-data
```

### 5. Open notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
```

Run in order: `01_data` → `02_models` → `03_evaluation` → `04_shap`

> ⏱ `02_models` takes 2–5 minutes (walk-forward validation across 4 models × 15 folds)

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
xgboost
shap
matplotlib
requests
getFamaFrenchFactors
lxml
tiingo
nbformat
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Paper

The full paper ([`paper.tex`](paper.tex)) is a 32-page LaTeX document structured as a journal submission:

1. Introduction
2. Literature Review (factor models, ML in asset pricing, SHAP, transaction costs)
3. Data
4. Methodology (model specs, walk-forward design, evaluation metrics)
5. Empirical Results (IC, annual breakdown, portfolio, dynamic SHAP, SHAP regression)
6. Discussion
7. Robustness Checks (alternative windows, subsamples, portfolio construction)
8. Conclusion
9. Appendices (ETF universe, feature correlation matrix, hyperparameter grid)

Compile with:
```bash
pdflatex paper.tex && pdflatex paper.tex
```

---

## Related Work

This project is a companion to a separate dissertation on time-varying stock–bond correlations and the 2022 inflation shock. The two papers together suggest a coherent picture in which macro regime shifts alter the cross-sectional pricing of risk factors — the stock–bond correlation paper establishes the regime structure; this paper shows that the regime structure predicts which factor exposures XGBoost relies on.

---

## Limitations

- Universe of 30 ETFs is small; results may not generalise to individual-stock cross-sections
- SHAP regression uses synthetic macro data by default (T=15 observations; results are indicative)
- No explicit regime-switching model; macro conditioning is implicit through XGBoost's tree splits
- SHAP decomposition is specific to tree ensembles; neural networks may learn different representations

---

## License

MIT
