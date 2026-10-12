"""
Feature Engineering Module for Time-Series Electricity Forecasting.
Extracts calendar features, lag observations, and rolling statistics.
Strictly avoids data leakage by shifting rolling windows.
"""

import numpy as np
import pandas as pd


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar and cyclical temporal features from DatetimeIndex.
    """
    data = df.copy()
    
    # Calendar identifiers
    data["day_of_week"] = data.index.dayofweek
    data["day_of_month"] = data.index.day
    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["day_of_year"] = data.index.dayofyear
    data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

    # Cyclical sin/cos encodings for periodic continuity
    data["sin_day_of_week"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cos_day_of_week"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12.0)

    return data


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Consumption",
    lags: list = [1, 2, 7, 14],
    rolling_windows: list = [7, 14]
) -> pd.DataFrame:
    """
    Creates historical lag features and rolling aggregations.
    
    Note on preventing data leakage:
        All rolling windows use .shift(1) so current day target observation is NEVER
        included in the calculation of predictors for that day.
    """
    data = df.copy()
    new_feature_cols = []
    
    # Lag features
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        data[col_name] = data[target_col].shift(lag)
        new_feature_cols.append(col_name)

    # Rolling window statistics (strictly shifted by 1)
    for window in rolling_windows:
        shifted_series = data[target_col].shift(1)
        col_mean = f"{target_col}_roll_mean_{window}"
        col_std = f"{target_col}_roll_std_{window}"
        data[col_mean] = shifted_series.rolling(window=window).mean()
        data[col_std] = shifted_series.rolling(window=window).std()
        new_feature_cols.extend([col_mean, col_std])

    # Drop only early rows with NaNs introduced by the longest lag/rolling window
    # Do NOT drop rows due to missing values in unrelated columns like Solar/Wind
    data = data.dropna(subset=new_feature_cols).copy()
    return data


def build_feature_pipeline(df: pd.DataFrame, target_col: str = "Consumption") -> pd.DataFrame:
    """
    Executes full feature engineering pipeline.
    """
    data = create_calendar_features(df)
    data = create_lag_and_rolling_features(data, target_col=target_col)
    print(f"[FeatureEngineering] Generated {data.shape[1]} total features across {len(data)} rows.")
    return data


if __name__ == "__main__":
    from data_loader import load_electricity_data
    raw = load_electricity_data()
    feat = build_feature_pipeline(raw)
    print(feat.columns.tolist())
    print(f"Shape: {feat.shape}")
