"""
Data Loader Module for Electricity Consumption Forecasting.
Fetches, caches, validates, and cleans historical electricity consumption data.
"""

import os
import urllib.request
import pandas as pd
import numpy as np


OPSD_DATASET_URL = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "electricity_consumption.csv")


def download_dataset(target_path: str = DEFAULT_DATA_PATH, url: str = OPSD_DATASET_URL) -> str:
    """
    Downloads the real public electricity dataset if not already present.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        print(f"[DataLoader] Using cached dataset at: {target_path}")
        return target_path

    print(f"[DataLoader] Downloading public dataset from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"[DataLoader] Successfully saved dataset to: {target_path}")
    except Exception as e:
        print(f"[DataLoader] Warning: Could not download directly ({e}). Checking local fallback...")
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Dataset could not be downloaded and was not found at {target_path}. "
                f"Please ensure an internet connection or place a valid CSV at {target_path}."
            )
    return target_path


def load_electricity_data(
    file_path: str = None,
    target_col: str = "Consumption",
    date_col: str = "Date",
    force_download: bool = False
) -> pd.DataFrame:
    """
    Loads, validates, and cleans the electricity consumption dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataframe indexed chronologically by Datetime.
    """
    if file_path is None:
        file_path = DEFAULT_DATA_PATH
    
    if force_download or not os.path.exists(file_path):
        download_dataset(target_path=file_path)

    print(f"[DataLoader] Loading data from {file_path} ...")
    df = pd.read_csv(file_path)

    # Validate column presence (case-insensitive search if needed)
    cols_lower = {c.lower(): c for c in df.columns}
    
    actual_date_col = None
    for cand in [date_col, "date", "datetime", "timestamp"]:
        if cand.lower() in cols_lower:
            actual_date_col = cols_lower[cand.lower()]
            break
            
    actual_target_col = None
    for cand in [target_col, "consumption", "load", "demand", "pjme_mw"]:
        if cand.lower() in cols_lower:
            actual_target_col = cols_lower[cand.lower()]
            break

    if actual_date_col is None:
        raise ValueError(f"Could not find a valid timestamp column in dataset. Available: {list(df.columns)}")
    if actual_target_col is None:
        raise ValueError(f"Could not find a target consumption column in dataset. Available: {list(df.columns)}")

    # Parse timestamps and sort chronologically
    df[actual_date_col] = pd.to_datetime(df[actual_date_col], errors="coerce")
    df = df.dropna(subset=[actual_date_col])
    df = df.sort_values(by=actual_date_col).reset_index(drop=True)

    # Check for duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[actual_date_col])
    if len(df) < initial_rows:
        print(f"[DataLoader] Removed {initial_rows - len(df)} duplicate timestamps.")

    # Target variable validation
    df[actual_target_col] = pd.to_numeric(df[actual_target_col], errors="coerce")
    null_target_count = df[actual_target_col].isnull().sum()
    if null_target_count > 0:
        print(f"[DataLoader] Imputing {null_target_count} missing target values using forward-fill.")
        df[actual_target_col] = df[actual_target_col].ffill().bfill()

    # Standardize column names
    df = df.rename(columns={actual_date_col: "Date", actual_target_col: "Consumption"})
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    print(f"[DataLoader] Loaded {len(df)} records from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}.")
    print(f"[DataLoader] Target stats: Mean={df['Consumption'].mean():.2f}, Std={df['Consumption'].std():.2f}, Min={df['Consumption'].min():.2f}, Max={df['Consumption'].max():.2f}")
    
    return df


if __name__ == "__main__":
    df = load_electricity_data()
    print(df.head())
