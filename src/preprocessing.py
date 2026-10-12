"""
Data Preprocessing Module for Electricity Consumption Forecasting.
Handles chronological splitting, scaling without data leakage,
and sliding-window sequence creation for LSTM.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DEFAULT_SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scaler.joblib")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    """
    Splits time-series chronologically into Train, Validation, and Test sets.
    Preserves temporal ordering to prevent lookahead bias.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"[Preprocessing] Chronological Split:")
    print(f"  - Train:      {len(train_df)} records ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  - Validation: {len(val_df)} records ({val_df.index.min().date()} to {val_df.index.max().date()})")
    print(f"  - Test:       {len(test_df)} records ({test_df.index.min().date()} to {test_df.index.max().date()})")

    return train_df, val_df, test_df


def fit_and_save_scaler(train_values: np.ndarray, scaler_path: str = DEFAULT_SCALER_PATH) -> MinMaxScaler:
    """
    Fits a MinMaxScaler strictly on training target values and saves it.
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values.reshape(-1, 1))
    joblib.dump(scaler, scaler_path)
    print(f"[Preprocessing] Fitted and saved target scaler to {scaler_path}")
    return scaler


def load_scaler(scaler_path: str = DEFAULT_SCALER_PATH) -> MinMaxScaler:
    """
    Loads saved MinMaxScaler artifact.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
    return joblib.load(scaler_path)


def create_sliding_sequences(
    series: np.ndarray,
    n_steps: int = 14
):
    """
    Transforms a 1D continuous array into sliding window input/output pairs for LSTM.
    
    Given sequence [x_0, x_1, ..., x_t, ...]:
    X[i] = [x_i, x_{i+1}, ..., x_{i+n_steps-1}]
    y[i] = x_{i+n_steps}
    
    Returns:
        X (np.ndarray): Shape (samples, n_steps, 1)
        y (np.ndarray): Shape (samples, 1)
    """
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i : i + n_steps])
        y.append(series[i + n_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to 3D: [samples, time_steps, features]
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=-1)
        
    return X, y.reshape(-1, 1)


def prepare_lstm_data(
    df: pd.DataFrame,
    target_col: str = "Consumption",
    n_steps: int = 14,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    scaler_path: str = DEFAULT_SCALER_PATH
):
    """
    Full pipeline to prepare scaled 3D sequence tensors for LSTM training & testing.
    Uses continuous series to create sequences at boundary without data leakage.
    """
    train_df, val_df, test_df = chronological_split(df, train_ratio, val_ratio)
    
    # Fit scaler strictly on training target
    scaler = fit_and_save_scaler(train_df[target_col].values, scaler_path)
    
    # Transform entire series with the train-fitted scaler
    all_scaled = scaler.transform(df[[target_col]].values).flatten()
    
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    
    # Build sequences for each partition with lookback buffer
    # Train: slices from 0 to n_train
    train_series = all_scaled[:n_train]
    X_train, y_train = create_sliding_sequences(train_series, n_steps)
    
    # Validation: slice including buffer of n_steps before validation starts
    val_series = all_scaled[n_train - n_steps : n_train + n_val]
    X_val, y_val = create_sliding_sequences(val_series, n_steps)
    
    # Test: slice including buffer of n_steps before test starts
    test_series = all_scaled[n_train + n_val - n_steps : ]
    X_test, y_test = create_sliding_sequences(test_series, n_steps)
    
    # Dates corresponding to test targets
    test_dates = test_df.index
    
    print(f"[Preprocessing] LSTM Tensors Prepared (lookback window={n_steps}):")
    print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  - X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  - X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "test_dates": test_dates,
        "test_actual_unscaled": test_df[target_col].values
    }


def prepare_tabular_data(
    feature_df: pd.DataFrame,
    target_col: str = "Consumption",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    """
    Prepares tabular feature matrices and targets for baseline ML models.
    """
    train_df, val_df, test_df = chronological_split(feature_df, train_ratio, val_ratio)
    
    drop_cols = [target_col]
    # Drop any other raw renewable columns if present to focus strictly on consumption features
    for c in ["Wind", "Solar", "Wind+Solar"]:
        if c in feature_df.columns:
            drop_cols.append(c)
            
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_col].values
    
    X_val = val_df.drop(columns=drop_cols, errors="ignore")
    y_val = val_df[target_col].values
    
    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[target_col].values
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "test_dates": test_df.index,
        "feature_names": list(X_train.columns)
    }


def inverse_transform_predictions(scaler: MinMaxScaler, scaled_preds: np.ndarray) -> np.ndarray:
    """
    Inverse-transforms scaled predictions back to physical units (GWh).
    """
    return scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
