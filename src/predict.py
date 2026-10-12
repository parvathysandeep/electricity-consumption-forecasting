"""
Inference & Prediction Module for Electricity Demand Forecasting.
Loads trained LSTM model and scaler to produce future forecasts from recent inputs.
Includes CLI interface and input validation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Union, List

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import load_scaler, inverse_transform_predictions, DEFAULT_SCALER_PATH
from src.lstm_model import load_trained_lstm, DEFAULT_LSTM_PATH
from src.data_loader import load_electricity_data, DEFAULT_DATA_PATH


def predict_next_step(
    recent_observations: Union[List[float], np.ndarray],
    model=None,
    scaler=None,
    model_path: str = DEFAULT_LSTM_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    n_steps: int = 14
) -> dict:
    """
    Generates a single-step electricity demand forecast from the most recent n_steps observations.
    """
    obs = np.asarray(recent_observations, dtype=np.float32).flatten()

    # Input validation
    if len(obs) < n_steps:
        raise ValueError(
            f"Insufficient observations provided. Expected at least {n_steps} historical "
            f"timesteps, but received {len(obs)}."
        )

    # Use the most recent n_steps
    seq_input = obs[-n_steps:]

    if np.any(np.isnan(seq_input)):
        raise ValueError("Input sequence contains NaN / missing values. Please clean before prediction.")

    # Lazy-load artifacts if not provided
    if scaler is None:
        scaler = load_scaler(scaler_path)
    if model is None:
        model = load_trained_lstm(model_path)

    # Scale input using train scaler
    seq_scaled = scaler.transform(seq_input.reshape(-1, 1)).flatten()

    # Reshape to (1, n_steps, 1) for LSTM input
    X_input = seq_scaled.reshape(1, n_steps, 1)

    # Predict in scaled space
    y_scaled_pred = model(X_input, training=False).numpy()

    # Inverse transform to original GWh
    predicted_demand = float(inverse_transform_predictions(scaler, y_scaled_pred)[0])

    # Out-of-bounds warning check based on typical grid distribution
    min_bound, max_bound = scaler.data_min_[0] * 0.7, scaler.data_max_[0] * 1.3
    warning = None
    if predicted_demand < min_bound or predicted_demand > max_bound:
        warning = f"Prediction {predicted_demand:.2f} GWh is outside normal historical envelope ({min_bound:.1f} - {max_bound:.1f} GWh)."

    return {
        "predicted_demand_gwh": round(predicted_demand, 2),
        "recent_input_mean_gwh": round(float(np.mean(seq_input)), 2),
        "lookback_steps_used": n_steps,
        "warning": warning
    }


def predict_multi_step_autoregressive(
    recent_observations: Union[List[float], np.ndarray],
    horizon: int = 7,
    model_path: str = DEFAULT_LSTM_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    n_steps: int = 14
) -> List[float]:
    """
    Generates multi-step forecasts using iterative auto-regressive rollout.
    Loads model and scaler once for high throughput.
    """
    scaler = load_scaler(scaler_path)
    model = load_trained_lstm(model_path)
    
    obs = list(np.asarray(recent_observations, dtype=np.float32).flatten())
    predictions = []

    for step in range(horizon):
        res = predict_next_step(
            recent_observations=obs,
            model=model,
            scaler=scaler,
            n_steps=n_steps
        )
        pred_val = res["predicted_demand_gwh"]
        predictions.append(pred_val)
        obs.append(pred_val)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast electricity consumption using trained LSTM.")
    parser.add_argument("--steps", type=int, default=14, help="Lookback window size.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecasting horizon (days).")
    parser.add_argument("--use-test-sample", action="store_true", default=True, help="Load most recent values from dataset.")
    args = parser.parse_args()

    # Load dataset to extract recent sample
    df = load_electricity_data()
    recent_vals = df["Consumption"].values[-args.steps:]

    print(f"\n[CLI Prediction] Forecasting next {args.horizon} step(s) from last {args.steps} days...")
    print(f"Recent {args.steps} days observed demand (GWh): {[round(float(x), 1) for x in recent_vals]}")

    if args.horizon == 1:
        res = predict_next_step(recent_vals, n_steps=args.steps)
        print("\n" + "=" * 45)
        print(f" FORECAST RESULT: {res['predicted_demand_gwh']} GWh")
        print(f" Input Mean:     {res['recent_input_mean_gwh']} GWh")
        if res["warning"]:
            print(f" Warning:        {res['warning']}")
        print("=" * 45)
    else:
        multi_preds = predict_multi_step_autoregressive(recent_vals, horizon=args.horizon, n_steps=args.steps)
        print("\n" + "=" * 45)
        print(f" {args.horizon}-DAY AHEAD FORECAST (GWh):")
        for day, val in enumerate(multi_preds, 1):
            print(f" Day +{day}: {val} GWh")
        print("=" * 45)
