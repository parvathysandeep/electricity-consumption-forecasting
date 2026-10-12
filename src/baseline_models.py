"""
Baseline Forecasting Models for Electricity Demand Comparison.
Implements:
1. Persistence (Naive) Forecast (y_hat_t = y_{t-1})
2. Linear Regression (L2 Ridge / OLS)
3. Random Forest Regressor (ensemble decision trees on lag/calendar features)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


RF_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "random_forest_model.joblib")


class PersistenceForecaster:
    """
    Naive baseline model that predicts tomorrow's electricity consumption
    as being identical to today's observed consumption (lag-1 persistence).
    """
    def __init__(self, lag_col: str = "Consumption_lag_1"):
        self.lag_col = lag_col
        self.name = "Persistence (Naive)"

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lag_col in X.columns:
            return X[self.lag_col].values
        # Fallback if first column represents lag-1
        return X.iloc[:, 0].values


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 12,
    random_state: int = 42,
    save_path: str = RF_MODEL_PATH
) -> RandomForestRegressor:
    """
    Trains and serializes a Random Forest Regressor.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"[BaselineModels] Training Random Forest Regressor (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, save_path)
    print(f"[BaselineModels] Saved trained Random Forest model to: {save_path}")
    return rf


def train_linear_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    alpha: float = 1.0
) -> Ridge:
    """
    Trains a Ridge Linear Regression baseline.
    """
    print(f"[BaselineModels] Training Ridge Linear Regression baseline (alpha={alpha})...")
    lr = Ridge(alpha=alpha)
    lr.fit(X_train, y_train)
    return lr


def evaluate_baselines(tabular_data: dict) -> dict:
    """
    Fits and evaluates baseline models on held-out test data.
    """
    X_train, y_train = tabular_data["X_train"], tabular_data["y_train"]
    X_test, y_test = tabular_data["X_test"], tabular_data["y_test"]

    # 1. Persistence
    persistence = PersistenceForecaster()
    persistence_preds = persistence.predict(X_test)

    # 2. Linear Regression (Ridge)
    ridge_model = train_linear_model(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)

    # 3. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    return {
        "Persistence": persistence_preds,
        "Linear Regression": ridge_preds,
        "Random Forest": rf_preds,
        "actual": y_test,
        "models": {
            "Persistence": persistence,
            "Linear Regression": ridge_model,
            "Random Forest": rf_model
        }
    }
