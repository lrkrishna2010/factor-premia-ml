"""
models.py
---------
All model classes and the walk-forward validation engine.

Models
------
    OLS_FF5     : OLS regression on FF5 + Mom + ST_Rev betas (baseline)
    LassoModel  : Lasso with alpha tuned on validation fold
    RFModel     : Random Forest
    XGBModel    : XGBoost

Walk-Forward Schema
-------------------
    |--- Train (5 yrs) ---||-- Val (1 yr) --|| Test (1 yr) | → roll 1 year
    
    - Hyperparameters tuned on validation fold
    - Predictions made only on test fold (never seen during training or tuning)
    - Results concatenated across all test folds = full out-of-sample record
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


# ── Base model wrapper ────────────────────────────────────────────────────────

class BaseModel:
    name: str = "Base"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ── OLS baseline ──────────────────────────────────────────────────────────────

class OLS_FF5(BaseModel):
    """Standard OLS on all features — equivalent to FF5+Mom+STRev factor model."""
    name = "OLS (FF5+Mom+STRev)"

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ── Lasso ─────────────────────────────────────────────────────────────────────

class LassoModel(BaseModel):
    """Lasso with alpha selected by validation-fold MSE."""
    name = "Lasso"

    ALPHAS = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5]

    def __init__(self):
        self.best_alpha = None
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            best_mse = np.inf
            for alpha in self.ALPHAS:
                m = Lasso(alpha=alpha, max_iter=5000)
                m.fit(X_train, y_train)
                mse = np.mean((m.predict(X_val) - y_val) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    self.best_alpha = alpha
        else:
            self.best_alpha = 1e-3  # default

        self.model = Lasso(alpha=self.best_alpha, max_iter=5000)
        self.model.fit(np.vstack([X_train, X_val]) if X_val is not None else X_train,
                       np.concatenate([y_train, y_val]) if y_val is not None else y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ── Random Forest ─────────────────────────────────────────────────────────────

class RFModel(BaseModel):
    """Random Forest with light hyperparameter search on validation fold."""
    name = "Random Forest"

    PARAM_GRID = [
        {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 20},
        {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20},
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 10},
    ]

    def __init__(self):
        self.best_params = None
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            best_mse = np.inf
            for params in self.PARAM_GRID:
                m = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
                m.fit(X_train, y_train)
                mse = np.mean((m.predict(X_val) - y_val) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    self.best_params = params
        else:
            self.best_params = self.PARAM_GRID[0]

        X_fit = np.vstack([X_train, X_val]) if X_val is not None else X_train
        y_fit = np.concatenate([y_train, y_val]) if y_val is not None else y_train
        self.model = RandomForestRegressor(**self.best_params, n_jobs=-1, random_state=42)
        self.model.fit(X_fit, y_fit)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBModel(BaseModel):
    """XGBoost with validation-fold early stopping."""
    name = "XGBoost"

    def __init__(self):
        self.model = None
        self.best_n_rounds = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        if X_val is not None and y_val is not None:
            self.model = xgb.XGBRegressor(
                **params,
                early_stopping_rounds=30,
                eval_metric="rmse",
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.best_n_rounds = self.model.best_iteration
        else:
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train, verbose=False)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


# ── Walk-forward engine ───────────────────────────────────────────────────────

def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[BaseModel],
    raw_returns: pd.Series,
    train_years: int = 5,
    val_years: int = 1,
    test_years: int = 1,
) -> dict:
    """
    Run walk-forward validation for all models.

    Parameters
    ----------
    X, y    : feature matrix and target (MultiIndex: Date, Asset)
    models  : list of model instances
    train_years, val_years, test_years : window lengths in years

    Returns
    -------
    results : dict with keys = model names, values = dict containing:
        'predictions' : Series(Date, Asset) of out-of-sample predictions
        'actuals'     : Series(Date, Asset) of realised targets
        'shap_values' : dict(year -> array) for XGBModel (if applicable)
        'model_fitted': list of fitted models per fold (for SHAP)
    """
    dates = X.index.get_level_values("Date").unique().sort_values()
    years = dates.year.unique()

    train_months = train_years * 12
    val_months   = val_years  * 12

    results = {m.name: {
        "predictions": [],
        "actuals": [],
        "raw_returns": [],
        "dates": [],
        "shap_inputs": [],
    } for m in models}

    fold_count = 0

    # Slide forward one year at a time
    for fold_start in range(train_months, len(dates) - val_months - test_years * 12 + 1,
                             test_years * 12):
        train_end   = fold_start
        val_end     = fold_start + val_months
        test_end    = min(val_end + test_years * 12, len(dates))

        if test_end <= val_end:
            break

        train_dates = dates[:train_end]
        val_dates   = dates[train_end: val_end]
        test_dates  = dates[val_end: test_end]

        if len(test_dates) == 0:
            break

        # Filter panel
        X_train = X[X.index.get_level_values("Date").isin(train_dates)].values
        y_train = y[y.index.get_level_values("Date").isin(train_dates)].values
        X_val   = X[X.index.get_level_values("Date").isin(val_dates)].values
        y_val   = y[y.index.get_level_values("Date").isin(val_dates)].values

        test_mask = y.index.get_level_values("Date").isin(test_dates)
        X_test  = X[test_mask].values
        y_test  = y[test_mask]
        raw_test = raw_returns[test_mask] if raw_returns is not None else y_test

        fold_count += 1
        test_year = test_dates[0].year
        print(f"  Fold {fold_count}: train → {train_dates[-1].date()} | "
              f"val {val_dates[0].date()}–{val_dates[-1].date()} | "
              f"test {test_dates[0].date()}–{test_dates[-1].date()}")

        for model in models:
            model.fit(X_train, y_train, X_val, y_val)
            preds = model.predict(X_test)

            results[model.name]["predictions"].append(pd.Series(preds, index=y_test.index))
            results[model.name]["actuals"].append(y_test)
            results[model.name]["raw_returns"].append(raw_test)

            # Store for SHAP analysis
            if hasattr(model, "feature_importances"):
                results[model.name]["shap_inputs"].append({
                    "year": test_year,
                    "X_test": X_test,
                    "X_test_df": X[test_mask],
                    "model": model,
                })

    # Concatenate across folds
    for m_name in results:
        results[m_name]["predictions"] = pd.concat(results[m_name]["predictions"])
        results[m_name]["actuals"]     = pd.concat(results[m_name]["actuals"])
        results[m_name]["raw_returns"] = pd.concat(results[m_name]["raw_returns"])

    print(f"\nWalk-forward complete: {fold_count} folds, "
          f"{len(results[models[0].name]['predictions']):,} out-of-sample observations")
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data import load_data
    from src.features import build_feature_matrix

    factors, returns = load_data(use_synthetic=True)
    X, y = build_feature_matrix(returns, factors)

    models = [OLS_FF5(), LassoModel(), RFModel(), XGBModel()]
    print("\nRunning walk-forward validation...")
    results = walk_forward(X, y, models, train_years=5, val_years=1, test_years=1)

    for name, res in results.items():
        preds = res["predictions"]
        actuals = res["actuals"]
        r2 = r2_score(actuals, preds)
        ic, _ = spearmanr(actuals, preds)
        print(f"\n{name}:")
        print(f"  OOS R²: {r2:.4f}")
        print(f"  IC (Spearman): {ic:.4f}")