"""
Evaluation Metrics Module for Electricity Demand Forecasting.
Computes MAE, RMSE, MAPE, and R2 on original demand scale.
Generates structured comparison summaries.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Computes regression evaluation metrics in original physical units (GWh).
    Handles potential divide-by-zero for MAPE using epsilon.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    
    # Safe MAPE calculation
    epsilon = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100.0)
    
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "Model": model_name,
        "MAE (GWh)": round(mae, 3),
        "RMSE (GWh)": round(rmse, 3),
        "MAPE (%)": round(mape, 3),
        "R2 Score": round(r2, 4)
    }
    return metrics


def compare_models(results_dict: dict, save_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Builds a consolidated comparison DataFrame for all models and saves to disk.
    
    Args:
        results_dict: Dictionary mapping model names to tuples/dicts of (y_true, y_pred).
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    
    for model_name, preds_info in results_dict.items():
        y_true = preds_info["actual"]
        y_pred = preds_info["predicted"]
        m = calculate_metrics(y_true, y_pred, model_name=model_name)
        rows.append(m)

    df_summary = pd.DataFrame(rows)
    # Sort models by RMSE ascending
    df_summary = df_summary.sort_values(by="RMSE (GWh)", ascending=True).reset_index(drop=True)

    # Save as CSV and JSON
    csv_path = os.path.join(save_dir, "metrics_summary.csv")
    json_path = os.path.join(save_dir, "metrics_summary.json")
    
    df_summary.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df_summary.to_dict(orient="records"), f, indent=4)

    print("\n" + "=" * 65)
    print("           MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 65)
    print(df_summary.to_string(index=False))
    print("=" * 65 + "\n")
    print(f"[Evaluation] Metrics saved to {csv_path} and {json_path}")
    
    return df_summary
